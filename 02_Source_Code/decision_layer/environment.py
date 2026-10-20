"""GraphSAGE -> PPO railway dispatch environment.

The environment deliberately exposes PPO only to *controllable* operational
situations. Direct injected train delays are excluded from the objective.
Temporary infrastructure disruptions remain because they create genuine
resource contention.
"""
from __future__ import annotations
import os, sys, copy
from typing import Dict, List, Optional
import gymnasium as gym
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gymnasium import spaces
from data_layer.data_layer import load_all
from simulation_layer.simulation_layer import SimulationLayer
from graph_construction_layer.dorg_builder import DORGBuilder
from representation_layer.state_builder import RepresentationLayer

HOLD, GRANT_PLATFORM, GRANT_LOOP = 0, 1, 2
MAX_DECISIONS = 5
SLOT_FEATS = 6
GRAPH_SLOT_FEATS = 4   # graph-only: neighbourhood features gathered from DORG edges
from representation_layer.flat_state import build_flat_state, FLAT_BASE_DIM
ACTION_NAMES = {HOLD:"HOLD", GRANT_PLATFORM:"GRANT PLATFORM", GRANT_LOOP:"GRANT LOOP"}

class RailwayPPOEnv(gym.Env):
    metadata={"render_modes":["human"]}
    def __init__(self,start_time_min=480.0,end_time_min=1320.0,tick_min=1.0,
                 max_ticks=1440,seed=42,stress_mode=False,weekday=None,late_fraction=None,late_mean_min=25.0,obs_mode='graph'):
        super().__init__()
        self.start_time_min=float(start_time_min); self.end_time_min=float(end_time_min)
        self.tick_min=float(tick_min); self.max_ticks=int(max_ticks); self.seed_value=int(seed)
        self._timetable,self._stations,self._platforms,self._priority,self._disruptions=load_all(demo_mode=stress_mode,seed=seed)
        # The source generator can create direct delay events; simulation ignores them.
        self._sim=SimulationLayer(self._timetable,self._stations,self._platforms,self._priority,self._disruptions,tick_min=tick_min)
        self.obs_mode=obs_mode   # 'graph' = DORG + GraphSAGE, 'flat' = plain feature vector (no graph)
        self._dorg=DORGBuilder(self._stations,self._platforms)
        if obs_mode=='graph':
            self._rep=RepresentationLayer(hidden_dim=64,num_layers=2,dropout=0.0); base_dim=self._rep.state_dim+GRAPH_SLOT_FEATS*MAX_DECISIONS
        else:
            self._rep=None; base_dim=FLAT_BASE_DIM
        self.state_dim=base_dim+SLOT_FEATS*MAX_DECISIONS
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(self.state_dim,),dtype=np.float32)
        self.action_space=spaces.MultiDiscrete([3]*MAX_DECISIONS)
        self.stress_mode=bool(stress_mode); self._rs=None; self._obs=np.zeros(self.state_dim,np.float32)
        self._candidates=[]; self._last_reward=0.0; self._tick=0; self._total_decisions=0
        self._prev_delay=0.0
        self.late_fraction=late_fraction; self.late_mean_min=float(late_mean_min); self._initial_delay={}
        self.fixed_weekday=weekday; self.weekday=0 if weekday is None else int(weekday)%7; self._base_tt=self._timetable

    def _snapshot(self):
        self._candidates=self._decision_candidates(self._rs)
        if self.obs_mode=='graph':
            self._dorg.build(self._rs); base=np.concatenate([self._rep.encode(self._dorg.graph,self._rs),self._graph_slot_features()])
        else:
            base=build_flat_state(self._rs)
        self._obs=np.concatenate([base,self._slot_features()]).astype(np.float32)
        return self._obs

    def graph_context(self):
        """Return the current DORG and railway snapshot for differentiable GraphSAGE.

        The PPO rollout stores this context so the graph encoder can be recomputed
        during policy updates instead of training on permanently frozen embeddings.
        """
        if self.obs_mode != 'graph' or self._rs is None:
            return None
        return self._dorg.graph.copy(), copy.deepcopy(self._rs)

    def _graph_slot_features(self):
        """Message passing over the DORG: for each decision slot, aggregate its neighbourhood.
        Uses 'precedes' edges (higher-priority train -> this train) and station->loop edges."""
        g=self._dorg.graph; f=np.zeros((MAX_DECISIONS,GRAPH_SLOT_FEATS),np.float32)
        w=lambda p:(7-p)/2.0
        for i,c in enumerate(self._candidates[:MAX_DECISIONS]):
            node=f"T:{c['train_number']}"
            if node not in g: continue
            pl=self._train_priority(c['train_number']); blocked={}
            for u,_,d in g.in_edges(node,data=True):
                if d.get('edge_type')=='precedes': blocked[u]=g.nodes[u].get('priority',3)
            B=sum(0.25*(pl-p)*w(p) for p in blocked.values() if pl>p); H=w(pl)
            ls=c.get('loop_station') or c.get('station'); sn=f"S:{ls}"
            loops=[v for _,v,d in g.out_edges(sn,data=True) if d.get('edge_type')=='has_loop'] if sn in g else []
            free=sum(1 for l in loops if not any(d.get('edge_type')=='occupies_loop' for _,_,d in g.in_edges(l,data=True)))
            f[i]=[min(len(blocked),4)/4.0,min(B,10.0)/10.0,min(B/max(H,1e-6),5.0)/5.0,(free/len(loops)) if loops else 0.0]
        return f.reshape(-1)

    def _slot_features(self):
        """Explicit per-decision-slot features so PPO knows WHICH train/choice each slot is."""
        f=np.zeros((MAX_DECISIONS,SLOT_FEATS),np.float32)
        trains={int(t['train_number']):t for t in self._rs.get('trains',[])}
        for i,c in enumerate(self._candidates[:MAX_DECISIONS]):
            p=self._train_priority(c['train_number']); t=trains.get(c['train_number'],{})
            opp=c.get('opponent'); gap=abs(p-self._train_priority(opp)) if opp is not None else 0
            st=c.get('station')
            loop_ok=1.0 if self._sim.loop_line_manager.capacity_of(c.get('loop_station') or st)>0 else 0.0
            f[i]=[1.0 if c['kind']=='waiting' else 0.0,(p-1)/5.0,gap/5.0,min(float(t.get('delay_min',0)),60.0)/60.0,loop_ok,1.0]
        return f.reshape(-1)

    def reset(self,seed:Optional[int]=None,options=None):
        if seed is not None:
            self.seed_value=int(seed); self._disruptions.rng.seed(self.seed_value)
        self._sim.set_demo_mode(self.stress_mode)
        # real calendar day: fixed weekday, or one derived from the seed so training covers all 7 days
        self.weekday=int(self.fixed_weekday)%7 if self.fixed_weekday is not None else self.seed_value%7
        tt=self._base_tt.resolve(self.weekday,self.start_time_min,self.end_time_min)
        self._timetable=tt; self._sim.timetable=tt; self._sim.position_updater.timetable=tt
        self._sim.reset(self.start_time_min,self.end_time_min)
        self._inject_initial_lateness()
        self._tick=0; self._total_decisions=0; self._prev_delay=0.0; self._last_reward=0.0
        self._rs=self._sim.step().as_dict()
        self._prev_delay=self.total_delay()
        self._snapshot()
        return self._obs.copy(),{}

    def _inject_initial_lateness(self):
        """Stress scenarios: a share of trains start the window running late (exponential, mean ~25 min).
        This is scenario input, not agent cost: only delay ADDED after this point counts as controllable."""
        frac=self.late_fraction if self.late_fraction is not None else (0.4 if self.stress_mode else 0.0)
        rng=np.random.default_rng(self.seed_value+7); self._initial_delay={}
        for tn,st in self._sim.states.items():
            if frac>0 and rng.random()<frac:
                d=float(min(120.0,round(rng.exponential(self.late_mean_min),1)))
                st.delay_min=d; self._initial_delay[int(tn)]=d

    def set_stress_mode(self,on:bool):
        self.stress_mode=bool(on); self._sim.set_demo_mode(on)

    def total_delay(self):
        """Controllable delay = delay added since the scenario started (initial lateness excluded)."""
        return float(sum(max(float(t.get('delay_min',0))-self._initial_delay.get(int(t['train_number']),0.0),0.0) for t in (self._rs or {}).get('trains',[])))

    def weighted_delay(self):
        """Total delay weighted by train importance: priority 1 -> 3.0 ... priority 6 -> 0.5."""
        return float(sum(float(t.get('delay_min',0))*(7-self._train_priority(t['train_number']))/2.0 for t in (self._rs or {}).get('trains',[])))

    def weighted_controllable_delay(self):
        """Priority-weighted delay added after reset; excludes injected initial lateness."""
        total = 0.0
        for t in (self._rs or {}).get('trains', []):
            tn = int(t['train_number'])
            added = max(float(t.get('delay_min', 0.0)) - self._initial_delay.get(tn, 0.0), 0.0)
            total += added * (7 - self._train_priority(tn)) / 2.0
        return float(total)

    def decision_candidates(self): return list(self._candidates)

    def slot_valid(self):
        v=np.zeros(MAX_DECISIONS,dtype=np.bool_); v[:len(self._candidates[:MAX_DECISIONS])]=True; return v

    def action_mask(self):
        mask=np.zeros((MAX_DECISIONS,3),dtype=np.bool_)
        mask[:,HOLD]=True  # unused slots stay valid (HOLD only) so logits never go all -inf
        for i,c in enumerate(self._candidates[:MAX_DECISIONS]):
            mask[i,HOLD]=True
            if c['kind']=='waiting':
                st=c['station']
                if self._sim.platform_occupancy_manager.capacity_of(st)>0: mask[i,GRANT_PLATFORM]=True
                if self._sim.loop_line_manager.capacity_of(st)>0: mask[i,GRANT_LOOP]=True
            else:
                mask[i,1]=True  # PROCEED uses action id 1 for non-station conflict
                ls=c.get('loop_station')
                if ls and self._sim.loop_line_manager.capacity_of(ls)>0: mask[i,GRANT_LOOP]=True  # loop-and-overtake
        return mask

    def _decision_candidates(self,rs):
        trains={int(t['train_number']):t for t in rs.get('trains',[])}
        out=[]; seen=set()
        # Waiting/resource contention has the richest 3-way decision.
        for tn,st in rs.get('waiting_trains',[]):
            key=('waiting',int(tn),st)
            if key in seen: continue
            out.append({'kind':'waiting','train_number':int(tn),'station':st,'conflict':'platform/loop','opponent':None}); seen.add(key)
        # If there are no waiting trains, expose one train from each live
        # headway/track conflict as a hold/proceed decision. This makes the
        # policy meaningful before a platform queue forms.
        if len(out)<MAX_DECISIONS:
            for c in rs.get('conflicts',[]):
                if c.get('type') not in ('headway','track'): continue
                ranked=sorted([int(x) for x in c.get('trains',[])],key=lambda n:self._train_priority(n))
                if len(ranked)<2: continue
                tn=ranked[-1]
                key=('section',tn,c.get('location'))
                if key in seen: continue
                t=trains.get(tn)
                if not t or t.get('status') in ('TERMINATED','NOT_STARTED'): continue
                out.append({'kind':'section','train_number':tn,'station':t.get('station'),'conflict':c.get('type'),'location':c.get('location'),'opponent':ranked[0],'blocked':[int(x) for x in ranked[:-1]],'loop_station':t.get('to_station') or t.get('station')}); seen.add(key)
                if len(out)>=MAX_DECISIONS: break
        return out

    def _train_priority(self,tn): return self._priority.get(int(tn))

    def step(self,action):
        action=np.asarray(action,dtype=np.int64).reshape(-1)
        candidates=self._candidates[:MAX_DECISIONS]
        platform_req=set(); loop_req=set(); hold_req=set(); pass_req=set(); applied=[]
        mask=self.action_mask()
        for i,c in enumerate(candidates):
            a=int(action[i]) if i<len(action) else HOLD
            if not mask[i,a]: a=HOLD
            tn=int(c['train_number'])
            if c['kind']=='waiting':
                if a==GRANT_PLATFORM: platform_req.add((tn,c['station']))
                elif a==GRANT_LOOP: loop_req.add((tn,c['station']))
            else:
                if a==HOLD: hold_req.add(tn)
                elif a==GRANT_LOOP: pass_req.add(tn)
            applied.append({'slot':i,'kind':c['kind'],'opponent':c.get('opponent'),'blocked':[int(x) for x in c.get('blocked',[])],'loop_station':c.get('loop_station'),'priority':self._train_priority(tn),'time':float(self._rs['simulation_time_min']),'train_number':tn,'station':c.get('station'),'conflict':c.get('conflict'),'location':c.get('location'),'action':a,'action_label':('LOOP & OVERTAKE' if a==GRANT_LOOP else 'HOLD' if a==HOLD else 'PROCEED') if c['kind']=='section' else ACTION_NAMES[a]})
        self._sim.set_agent_requests(platform_req,loop_req,hold_req); self._sim.request_loop_pass(pass_req)
        old_delay=self.weighted_controllable_delay()
        old_waiting=len(self._rs.get('waiting_trains',[]))
        old_conflicts=len(self._rs.get('conflicts',[]))
        self._rs=self._sim.step().as_dict(); self._tick+=1
        self._snapshot()
        new_delay=self.total_delay()
        delay_delta=self.weighted_controllable_delay()-old_delay
        waiting=len(self._rs.get('waiting_trains',[])); conflicts=len(self._rs.get('conflicts',[]))
        # Main objective is added priority-weighted delay. Small potential terms
        # give earlier feedback when a dispatch decision clears queues/conflicts.
        reward=-delay_delta + 0.10*(old_waiting-waiting) + 0.10*(old_conflicts-conflicts)
        self._last_reward=float(reward); self._total_decisions+=len(applied)
        terminated=self._sim.clock.is_finished() and all(t.get('status')=='TERMINATED' for t in self._rs.get('trains',[]))
        truncated=self._tick>=self.max_ticks
        info={'sim_time_min':self._rs['simulation_time_min'],'n_waiting':waiting,'n_conflicts':conflicts,'total_delay':new_delay,'decisions':applied,'decision_required':bool(self._candidates),'slot_valid':self.slot_valid().tolist()}
        return self._obs.copy(),self._last_reward,terminated,truncated,info

    def advance_to_decision(self,max_ticks=100):
        """Advance without fabricating PPO actions until a controllable decision exists."""
        for _ in range(max_ticks):
            if self._candidates: return self._obs.copy(),False
            self._sim.clear_agent_requests(); self._rs=self._sim.step().as_dict(); self._tick+=1; self._snapshot()
            if self._sim.clock.is_finished(): return self._obs.copy(),True
        return self._obs.copy(),False

    def render(self):
        print(f"t={self._rs['simulation_time_min']:.0f} waiting={len(self._rs.get('waiting_trains',[]))} conflicts={len(self._rs.get('conflicts',[]))} delay={self.total_delay():.1f} decisions={len(self._candidates)}")
