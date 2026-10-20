"""Proper PPO (GAE, minibatches, entropy bonus, slot masking) for the railway env."""
from __future__ import annotations
import argparse,csv,os,sys,time,shutil
from pathlib import Path
import numpy as np, torch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import project_paths as P
from decision_layer.environment import RailwayPPOEnv
from decision_layer.ppo_agent import GraphPPOActorCritic
from decision_layer.baselines import priority_rule, conflict_aware_rule
from decision_layer.environment import MAX_DECISIONS

def collect(env,agent,seed,steps):
    obs,_=env.reset(seed=seed); B={k:[] for k in 'S A M V LP R SV C'.split()}; ep_r=0.; dec=0; delay=0.
    n_dec=0
    while n_dec<steps:
        obs,done=env.advance_to_decision()
        if done: break
        if not env._candidates: continue
        n_dec+=1
        m=torch.as_tensor(env.action_mask()).unsqueeze(0); sv=torch.as_tensor(env.slot_valid()).unsqueeze(0)
        st=torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0)
        ctx=env.graph_context() if agent.use_graph else None
        with torch.no_grad(): a,lp,v=agent.act(st,m,slot_valid=sv,graph_contexts=[ctx] if agent.use_graph else None)
        _,r,term,trunc,info=env.step(a.squeeze(0).numpy())
        for k,x in zip('S A M V LP R SV'.split(),[st[0],a[0],m[0],v[0],lp[0],r,sv[0]]): B[k].append(x)
        B['C'].append(ctx)
        ep_r+=r; dec+=int(sv.sum()); delay=info['total_delay']
        if term or trunc: break
    return B,ep_r,dec,delay

def gae(R,V,gamma,lam):
    adv=np.zeros(len(R),np.float32); last=0.; 
    for t in reversed(range(len(R))):
        nv=V[t+1] if t+1<len(R) else 0.
        d=R[t]+gamma*nv-V[t]; last=d+gamma*lam*last; adv[t]=last
    return adv,adv+np.array(V,np.float32)

def evaluate(env,agent,seeds,steps):
    out=[]
    for s in seeds:
        obs,_=env.reset(seed=s); tot=0.; n_dec=0
        while n_dec<steps:
            obs,done=env.advance_to_decision()
            if done: break
            if not env._candidates: continue
            n_dec+=1
            m=torch.as_tensor(env.action_mask()).unsqueeze(0); sv=torch.as_tensor(env.slot_valid()).unsqueeze(0)
            ctx=env.graph_context() if agent.use_graph else None
            with torch.no_grad(): a,_,_=agent.act(torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0),m,deterministic=True,slot_valid=sv,graph_contexts=[ctx] if agent.use_graph else None)
            _,r,te,tr,_=env.step(a[0].numpy()); tot+=r
            if te or tr: break
        out.append(tot)
    return float(np.mean(out))

def behaviour_clone(env,agent,episodes,steps,seed0,epochs=30,lr=1e-3,teacher=priority_rule):
    """Warm start: imitate the priority rule, then PPO improves on it."""
    S,M,A,V,C=[],[],[],[],[]
    for ep in range(episodes):
        env.reset(seed=seed0+ep)
        n_dec=0
        while n_dec<steps:
            obs,done=env.advance_to_decision()
            if done: break
            if not env._candidates: continue
            n_dec+=1; a=teacher(env); S.append(torch.as_tensor(obs,dtype=torch.float32)); M.append(torch.as_tensor(env.action_mask()))
            A.append(torch.as_tensor(a)); V.append(torch.as_tensor(env.slot_valid())); C.append(env.graph_context() if agent.use_graph else None); env.step(a)
    S,M,A,V=map(torch.stack,(S,M,A,V)); opt=torch.optim.Adam(agent.parameters(),lr=lr); n=len(S)
    for e in range(epochs):
        perm=torch.randperm(n); tot=0.
        for i in range(0,n,128):
            ix=perm[i:i+128]; contexts=[C[j] for j in ix.tolist()] if agent.use_graph else None; dists,_=agent.forward(S[ix],M[ix],graph_contexts=contexts)
            ce=sum(-(dists[k].log_prob(A[ix][:,k])*V[ix][:,k].float()).sum() for k in range(MAX_DECISIONS))/V[ix].sum().clamp(min=1)
            opt.zero_grad(); ce.backward(); opt.step(); tot+=float(ce)*len(ix)
        if e%10==9: print(f'  BC epoch {e+1} loss={tot/n:.4f} (samples={n})',flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--episodes',type=int,default=60); p.add_argument('--decision-steps',type=int,default=200)
    p.add_argument('--start',type=float,default=480); p.add_argument('--end',type=float,default=720)
    p.add_argument('--stress',action='store_true'); p.add_argument('--save',default=str(P.MODELS / 'policy_final.pt'))
    p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--gamma',type=float,default=0.99); p.add_argument('--lam',type=float,default=0.95)
    p.add_argument('--epochs',type=int,default=6); p.add_argument('--minibatch',type=int,default=64)
    p.add_argument('--clip',type=float,default=0.2); p.add_argument('--ent',type=float,default=0.01); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--eval-every',type=int,default=10); p.add_argument('--init',default=''); p.add_argument('--teacher',default='priority',choices=['priority','conflict_aware']); p.add_argument('--obs-mode',default='graph',choices=['graph','flat'],help='flat = plain RL without graph'); p.add_argument('--weekday',type=int,default=-1,help='0=Mon..6=Sun, -1 = all days (default)'); p.add_argument('--bc-episodes',type=int,default=0); p.add_argument('--seed-offset',type=int,default=0)
    p.add_argument('--validation-seeds',default='1000,1001,1002,1003,1004',help='fixed comma-separated validation seeds')
    a=p.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    env=RailwayPPOEnv(a.start,a.end,stress_mode=a.stress,weekday=None if a.weekday<0 else a.weekday,obs_mode=a.obs_mode); agent=GraphPPOActorCritic(env.state_dim,use_graph=(a.obs_mode=='graph'))
    if a.init and os.path.exists(a.init):
        ck=torch.load(a.init,weights_only=False); missing,unexpected=agent.load_state_dict(ck['actor_critic'],strict=False); print('resumed from',a.init,'missing keys:',len(missing),'unexpected keys:',len(unexpected))
    if a.bc_episodes: print('behaviour-cloning warm start'); behaviour_clone(env,agent,a.bc_episodes,a.decision_steps,5000,teacher=conflict_aware_rule if a.teacher=='conflict_aware' else priority_rule)
    opt=torch.optim.Adam(agent.parameters(),lr=a.lr,eps=1e-5); rows=[]; best=-1e18
    Path(a.save).parent.mkdir(parents=True,exist_ok=True); P.OUT_SYN.mkdir(parents=True,exist_ok=True); (P.OUT_SYN/'training_logs').mkdir(exist_ok=True)
    def save(path): torch.save({'actor_critic':agent.state_dict(),'state_dim':env.state_dim,'obs_mode':a.obs_mode,'use_graph':agent.use_graph,'episodes':a.episodes,'seed':a.seed,'validation_seeds':a.validation_seeds},path)
    for ep in range(1,a.episodes+1):
        for g in opt.param_groups: g['lr']=a.lr*(1-(ep-1)/a.episodes)  # linear decay
        B,er,dec,delay=collect(env,agent,seed=ep+a.seed_offset,steps=a.decision_steps)
        if len(B['R'])<2: print(f'episode {ep:03d} too few decisions, skipped'); continue
        S=torch.stack(B['S']); A=torch.stack(B['A']); M=torch.stack(B['M']); SV=torch.stack(B['SV']); OLD=torch.stack(B['LP'])
        V=[float(x) for x in B['V']]; adv,ret=gae(B['R'],V,a.gamma,a.lam)
        adv=torch.as_tensor(adv); ret=torch.as_tensor(ret); n=len(adv)
        for _ in range(a.epochs):
            perm=torch.randperm(n)
            for i in range(0,n,a.minibatch):
                ix=perm[i:i+a.minibatch]; ad=adv[ix]; ad=(ad-ad.mean())/(ad.std(unbiased=False)+1e-8) if len(ix)>1 else ad
                contexts=[B['C'][j] for j in ix.tolist()] if agent.use_graph else None
                lp,ent,v=agent.evaluate(S[ix],M[ix],A[ix],SV[ix],graph_contexts=contexts); ratio=(lp-OLD[ix]).exp()
                pl=-torch.min(ratio*ad,ratio.clamp(1-a.clip,1+a.clip)*ad).mean()
                loss=pl+0.5*(v-ret[ix]).pow(2).mean()-a.ent*ent.mean()
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(agent.parameters(),.5); opt.step()
        ev=None
        if ep%a.eval_every==0:
            agent.eval(); val_seeds=[int(x.strip()) for x in a.validation_seeds.split(',') if x.strip()]; ev=evaluate(env,agent,val_seeds,a.decision_steps); agent.train()
            if ev>best: best=ev; save(str(Path(a.save).with_name('policy_best.pt' if Path(a.save).name=='policy_final.pt' else Path(a.save).stem+'_best.pt')))
        rows.append((ep,er,dec,delay,ev)); print(f'episode {ep:03d} reward={er:8.2f} decisions={dec} delay={delay:.1f}'+(f' eval={ev:.2f}' if ev is not None else ''),flush=True)
    # Keep the final training weights for debugging, but make the requested
    # checkpoint the BEST validation policy. Previously this final save
    # overwrote the checkpoint path with the last (not necessarily best) epoch.
    requested = Path(a.save)
    last_path = requested.with_name(requested.stem + '_last' + requested.suffix)
    save(str(last_path))
    best_path = requested.with_name('policy_best.pt' if requested.name == 'policy_final.pt' else requested.stem + '_best' + requested.suffix)
    if best_path.exists():
        shutil.copy2(best_path, requested)
        print('saved best validation policy to', requested)
        print('saved final-episode weights to', last_path)
    else:
        shutil.copy2(last_path, requested)
        print('no validation checkpoint was produced; saved final weights to', requested)
    with open(f"{P.OUT_SYN}/training_logs/training_log_{a.obs_mode}.csv",'w',newline='') as f: w=csv.writer(f); w.writerow(['episode','reward','decisions','delay','eval']); w.writerows(rows)
if __name__=='__main__': main()
