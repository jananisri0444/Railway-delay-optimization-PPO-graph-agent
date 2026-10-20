"""Railway Dispatch dashboard  (DORG -> GraphSAGE -> PPO).

Single-file app: Flask API + the whole dashboard UI live in this file.


Run:  python run_live_dashboard.py   ->  http://127.0.0.1:5000
"""
from __future__ import annotations
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / '02_Source_Code'))
import project_paths as P   # folder locations (also hides a harmless OpenMP warning)
import math
import threading
from pathlib import Path

import torch
from flask import Flask, Response, jsonify, request

from decision_layer.environment import RailwayPPOEnv
from decision_layer.ppo_agent import GraphPPOActorCritic

CKPT_DIR = P.MODELS
app = Flask(__name__)
lock = threading.Lock()
S = {"env": None, "agent": None, "obs": None, "tick": 0, "decisions": [],
     "history": [], "checkpoint": "none", "stress": False, "finished": False, "trace": {}, "log": []}

MAX_ITEMS = 22      # max columns (stations + gaps) drawn in the route diagram
MAX_LANES = 3       # label lanes above / below the track
LANE_SEP = 1.05     # min horizontal separation (in columns) of trains sharing a lane


# --------------------------------------------------------------------------- #
# Agent / state helpers
# --------------------------------------------------------------------------- #
def load_agent(env):
    agent = GraphPPOActorCritic(env.state_dim, use_graph=(env.obs_mode == 'graph'))
    names = ("policy_flat.pt", "policy_flat_best.pt") if env.obs_mode == "flat" else ("policy_final.pt", "policy_best.pt")
    for name in names:
        path = CKPT_DIR / name
        if not path.exists():
            continue
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            use_graph = ck.get('use_graph', any(k.startswith('graph_encoder.') for k in ck['actor_critic']))
            if use_graph != agent.use_graph:
                agent = GraphPPOActorCritic(env.state_dim, use_graph=use_graph)
            agent.load_state_dict(ck["actor_critic"], strict=False)
            agent.eval()
            return agent, name
        except Exception as exc:  # corrupt / architecture mismatch -> do not crash the UI
            print(f"[warn] could not load {name}: {exc}")
    agent.eval()
    return agent, "untrained (random policy)"


def payload():
    e = S["env"]
    rs = (e._rs if e else None) or {}
    trains = [t for t in rs.get("trains", []) if t.get("status") not in ("NOT_STARTED", "TERMINATED")]
    trains.sort(key=lambda t: (-float(t.get("delay_min", 0)), t["train_number"]))
    h = S["history"]
    return {
        "ready": e is not None, "model": getattr(e, "obs_mode", "graph"), "weekday": getattr(e, "weekday", 0), "tick": S["tick"], "finished": S["finished"],
        "sim_time_min": rs.get("simulation_time_min", 0), "mode": rs.get("mode"),
        "stress_mode": S["stress"], "checkpoint": S["checkpoint"],
        "trains": trains, "waiting": rs.get("waiting_trains", []),
        "conflicts": rs.get("conflicts", []),
        "active_disruptions": rs.get("active_disruptions", []),
        "decisions": S["decisions"], "metrics": h[-1] if h else None,
        "history": h[-120:],
    }


def need_env():
    if S["env"] is None:
        return jsonify({"error": "not initialised - press Realistic or Stress Test first"}), 400
    return None


# --------------------------------------------------------------------------- #
# Clean route-diagram model (all layout work is done here, UI only draws)
# --------------------------------------------------------------------------- #
def _split_section(loc):
    if isinstance(loc, (list, tuple)) and len(loc) == 2:
        return str(loc[0]), str(loc[1])
    if isinstance(loc, str) and "-" in loc:
        a, b = loc.split("-", 1)
        return a, b
    return None


def build_route_view(e, full=False):
    rs = e._rs or {}
    db = e._stations
    codes = list(db.ordered_codes)
    idx = {c: i for i, c in enumerate(codes)}
    trains = [t for t in rs.get("trains", []) if t.get("status") not in ("NOT_STARTED", "TERMINATED")]
    conflicts = rs.get("conflicts", [])
    waiting = {int(tn) for tn, _ in rs.get("waiting_trains", [])}
    decided = {int(c["train_number"]) for c in e._candidates}
    in_conflict = {int(x) for c in conflicts for x in c.get("trains", [])}
    disruptions = rs.get("active_disruptions", [])

    def real_index(t):
        """Fractional position of a train along the ordered station list."""
        if t.get("station") in idx:
            return float(idx[t["station"]])
        a, b = t.get("from_station"), t.get("to_station")
        if a in idx and b in idx:
            da, dbb = db.distance_km(a), db.distance_km(b)
            pos = t.get("position_km")
            f = 0.5
            if pos is not None and da != dbb and min(da, dbb) <= pos <= max(da, dbb):
                f = (pos - da) / (dbb - da)
            return idx[a] + f * (idx[b] - idx[a])
        return None

    # trains shown: everything in 'full' mode, else only the ones that matter
    shown = []
    for t in trains:
        tn = int(t["train_number"])
        role = ("decision" if tn in decided else "conflict" if tn in in_conflict
                else "waiting" if tn in waiting else "delayed" if float(t.get("delay_min", 0)) > 0 else "normal")
        r = real_index(t)
        if r is None:
            continue
        if full or role != "normal":
            shown.append((t, role, r))
    if not shown:   # nothing interesting -> show up to 8 active trains for context
        shown = [(t, "normal", real_index(t)) for t in trains if real_index(t) is not None][:8]

    # --- which stations get a column -------------------------------------------------
    keep = {0, len(codes) - 1}
    for _, _, r in shown:
        keep.update({int(math.floor(r)), int(math.ceil(r))})
    for c in conflicts:
        sec = _split_section(c.get("location"))
        for code in (sec or (c.get("location"),)):
            if code in idx:
                keep.add(idx[code])
    for d in disruptions:
        sec = _split_section(d.get("section")) if d.get("section") else None
        for code in (sec or (d.get("station"),)):
            if code in idx:
                keep.add(idx[code])
    keep = {k for k in keep if 0 <= k < len(codes)}

    def compress(ks):
        out, prev = [], None
        for k in sorted(ks):
            if prev is not None and k - prev > 1:
                out.append(("gap", prev + 1, k - 1))
            out.append(("st", k, k))
            prev = k
        return out

    cols = compress(keep)
    while len(cols) > MAX_ITEMS and len(keep) > 2:     # drop least important stations
        important = {0, len(codes) - 1}
        for _, _, r in shown:
            important.add(int(round(r)))
        droppable = sorted(k for k in keep if k not in important) or sorted(k for k in keep if k not in {0, len(codes) - 1})
        keep.discard(droppable[len(droppable) // 2])
        cols = compress(keep)

    xof = {}
    items = []
    for x, (kind, a, b) in enumerate(cols):
        if kind == "st":
            xof[a] = x
            c = codes[a]
            items.append({"x": x, "type": "station", "code": c, "name": db.get(c).name,
                          "km": round(db.distance_km(c), 1)})
        else:
            for k in range(a, b + 1):
                xof[k] = x
            items.append({"x": x, "type": "gap", "n": b - a + 1,
                          "from": codes[a], "to": codes[b]})

    def xmap(r):
        lo = int(math.floor(r)); hi = min(lo + 1, len(codes) - 1)
        xl, xh = xof.get(lo), xof.get(hi)
        if xl is None and xh is None:
            return None
        if xl is None: return float(xh)
        if xh is None: return float(xl)
        return xl + (r - lo) * (xh - xl)

    # --- train markers with collision-free lanes ---------------------------------------
    marks = []
    for t, role, r in shown:
        x = xmap(r)
        if x is None:
            continue
        a, b = t.get("from_station"), t.get("to_station")
        if a in idx and b in idx:
            side = "up" if idx[b] > idx[a] else "down"
        else:
            side = "up" if int(t["train_number"]) % 2 else "down"
        marks.append({"id": int(t["train_number"]), "x": round(x, 3), "role": role, "side": side,
                      "status": t.get("status"), "delay": float(t.get("delay_min", 0)),
                      "priority": t.get("priority"), "at": t.get("station"),
                      "from": a, "to": b})
    rank = {"decision": 0, "conflict": 1, "waiting": 2, "delayed": 3, "normal": 4}
    marks.sort(key=lambda m: (rank[m["role"]], m["id"]))
    lanes = {"up": [[] for _ in range(MAX_LANES)], "down": [[] for _ in range(MAX_LANES)]}
    placed, hidden = [], 0
    for m in marks:
        done = False
        for li in range(MAX_LANES):
            if all(abs(m["x"] - ox) >= LANE_SEP for ox in lanes[m["side"]][li]):
                lanes[m["side"]][li].append(m["x"]); m["lane"] = li; done = True; break
        if not done:                    # try the other side before giving up
            other = "down" if m["side"] == "up" else "up"
            for li in range(MAX_LANES):
                if all(abs(m["x"] - ox) >= LANE_SEP for ox in lanes[other][li]):
                    lanes[other][li].append(m["x"]); m["side"] = other; m["lane"] = li; done = True; break
        if done: placed.append(m)
        else: hidden += 1
    pos = {m["id"]: m for m in placed}

    # --- conflicts / disruptions as overlays ---------------------------------------------
    links = []
    for c in conflicts:
        ids = [int(x) for x in c.get("trains", []) if int(x) in pos]
        for i in range(len(ids) - 1):
            links.append({"a": ids[i], "b": ids[i + 1], "type": c.get("type")})
    spans = []
    for c in conflicts:
        sec = _split_section(c.get("location"))
        pts = [xof[idx[s]] for s in (sec or (c.get("location"),)) if s in idx and idx[s] in xof]
        if pts:
            spans.append({"x1": min(pts), "x2": max(pts), "kind": "conflict:" + str(c.get("type"))})
    for d in disruptions:
        sec = _split_section(d.get("section")) if d.get("section") else None
        pts = [xof[idx[s]] for s in (sec or (d.get("station"),)) if s in idx and idx[s] in xof]
        if pts:
            spans.append({"x1": min(pts), "x2": max(pts), "kind": d.get("kind")})

    return {"items": items, "trains": placed, "links": links, "spans": spans,
            "hidden": hidden, "total_active": len(trains), "shown": len(placed),
            "full": bool(full), "max_lanes": MAX_LANES}


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.get("/api/state")
def state():
    with lock:
        return jsonify(payload())



# --------------------------------------------------------------------------- #
# Per-train view: every decision, every station, loop-and-overtake events
# --------------------------------------------------------------------------- #
_W = lambda p: (7 - p) / 2.0          # importance weight of one delay-minute (priority 1 -> 3.0 ... 6 -> 0.5)


def _hook_trace(env):
    orig = env._sim.step
    def step():
        rs = orig(); t = float(getattr(rs, "simulation_time_min", 0.0))
        for tn, st in env._sim.states.items():
            if st.status.name in ("NOT_STARTED", "TERMINATED"): continue
            S["trace"].setdefault(int(tn), []).append((t, st.status.name, st.station_code or "", float(st.delay_min)))
        return rs
    env._sim.step = step


def _name(e, tn):
    s = e._timetable.trains.get(int(tn)); return s.train_name if s else ""


def _log_record(e, d, rid, tick, t_dec):
    r = dict(d); r["id"] = rid; r["tick"] = tick; r["time"] = float(d.get("time", t_dec)); tn = int(d["train_number"]); pl = e._train_priority(tn)
    blocked = [int(x) for x in d.get("blocked", []) if e._train_priority(x) < pl]
    r["blocked_info"] = [{"train": x, "name": _name(e, x), "priority": e._train_priority(x)} for x in blocked]
    r["name"] = _name(e, tn); r["hold_cost"] = round(_W(pl), 2)
    r["blocking_cost"] = round(sum(0.25 * (pl - e._train_priority(x)) * _W(e._train_priority(x)) for x in blocked), 2)
    r["n_blocked"] = len(blocked)
    r["loop_station_name"] = e._stations.get(d["loop_station"]).name if d.get("loop_station") else None
    return r


def _eff_time(tn, A):
    """Simulated clock time when the train reached the point it was scheduled to reach at minute A (delay shifts the timetable)."""
    for t, s, st, dl in S["trace"].get(int(tn), []):
        if t - dl >= A - 1e-6: return t, dl
    return None


def _stop_at(e, tn, code):
    sch = e._timetable.trains.get(int(tn))
    for p in (sch.stops if sch else []):
        if p.station_code == code:
            a = p.scheduled_arrival_min; a = p.scheduled_departure_min if a != a else a
            return p, float(a)
    return None, None


def _loop_event(e, r):
    code = r["loop_station"]; ev = {"id": r["id"], "time": r["time"], "train": r["train_number"], "name": r["name"], "priority": r["priority"],
        "station": code, "station_name": r["loop_station_name"], "until": r["time"] + e._sim.LOOP_PASS_MIN, "cost_min": e._sim.LOOP_PASS_COST_MIN, "blockers": []}
    for b in r["blocked_info"]:
        p, A = _stop_at(e, b["train"], code); pt = None
        est = False
        if A is not None:
            x = _eff_time(b["train"], A); pt = round(x[0], 1) if x else None
            if pt is None and S["trace"].get(b["train"]):          # not passed yet: expected time = scheduled + current delay
                pt = round(A + S["trace"][b["train"]][-1][3], 1); est = True
        ev["blockers"].append({"train": b["train"], "name": b["name"], "priority": b["priority"], "pass_time": pt, "scheduled": A, "estimated": est})
    return ev


@app.get("/api/trains")
def trains_list():
    with lock:
        bad = need_env()
        if bad: return bad
        e = S["env"]; cnt = {}
        for r in S["log"]:
            cnt.setdefault(r["train_number"], [0, 0])[0] += 1
            for b in r["blocked_info"]: cnt.setdefault(b["train"], [0, 0])[1] += 1
        out = []; now = [r for r in S["log"] if r["tick"] == S["tick"] and r.get("n_options", 2) > 1]
        deciding = {r["train_number"] for r in now}; blocked = {b["train"] for r in now for b in r["blocked_info"]}
        for tn, tr in S["trace"].items():
            last = tr[-1]; c = cnt.get(tn, [0, 0])
            out.append({"train": tn, "name": _name(e, tn), "priority": e._train_priority(tn), "status": last[1], "station": last[2], "delay": round(last[3], 1), "decided": c[0], "involved": c[1], "deciding_now": tn in deciding, "blocked_now": tn in blocked})
        out.sort(key=lambda x: (not x["deciding_now"], not x["blocked_now"], -(x["decided"] + x["involved"]), x["train"]))
        return jsonify({"trains": out})


@app.get("/api/train/<int:tn>")
def train_story(tn):
    with lock:
        bad = need_env()
        if bad: return bad
        e = S["env"]; sch = e._timetable.trains.get(tn)
        if sch is None and tn not in S["trace"]: return jsonify({"error": "train not in this simulation"}), 404
        decisions = [r for r in S["log"] if r["train_number"] == tn or any(b["train"] == tn for b in r["blocked_info"])]
        loops = [_loop_event(e, r) for r in S["log"] if r["kind"] == "section" and r["action"] == 2 and (r["train_number"] == tn or any(b["train"] == tn for b in r["blocked_info"]))]
        loop_at = {(ev["station"]) for ev in loops if ev["train"] == tn}; journey = []
        for p in (sch.stops if sch else []):
            a = p.scheduled_arrival_min; a = p.scheduled_departure_min if a != a else a; x = _eff_time(tn, float(a))
            journey.append({"station": p.station_code, "name": p.station_name, "halt": bool(p.is_halt), "scheduled": float(a), "sim_time": round(x[0], 1) if x else None,
                            "delay": round(x[1], 1) if x else None, "loop": p.station_code in loop_at})
        tr = S["trace"].get(tn, []); last = tr[-1] if tr else None
        return jsonify({"train": tn, "name": _name(e, tn), "priority": e._train_priority(tn), "status": last[1] if last else "NOT_STARTED", "delay": round(last[3], 1) if last else 0.0,
                        "decisions": decisions, "loop_events": loops, "journey": journey, "now": float(e._rs["simulation_time_min"]), "loop_pass_min": e._sim.LOOP_PASS_MIN})


@app.get("/api/decisions")
def decisions_since():
    since = int(request.args.get("since", 0))
    with lock: return jsonify({"decisions": [r for r in S["log"] if r["id"] > since][:200]})


def _run_full(env, agent, seed, label):
    env.reset(seed=seed); decs = []; reward = 0.0; n = 0
    while n < 4000:
        obs, done = env.advance_to_decision()
        if done: break
        if not env._candidates: continue
        mask = torch.as_tensor(env.action_mask(), dtype=torch.bool).unsqueeze(0); sv = torch.as_tensor(env.slot_valid()).unsqueeze(0)
        st = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        ctx = env.graph_context() if agent.use_graph else None
        with torch.no_grad(): acts, _, _ = agent.act(st, mask, deterministic=True, slot_valid=sv, graph_contexts=[ctx] if agent.use_graph else None)
        _, r, te, tr, info = env.step(acts.squeeze(0).numpy()); reward += r; n += 1
        for i, d in enumerate(info["decisions"]):
            decs.append({"time": round(d["time"]), "train": d["train_number"], "kind": d["kind"], "label": d["action_label"], "options": int(mask[0, i].sum())})
        if te or tr: break
    return {"checkpoint": label, "decisions": decs, "reward": round(reward, 2), "added_delay": round(float(env.total_delay()), 1), "steps": n}


@app.post("/api/compare")
def compare():
    """Run Graph PPO and Flat PPO on the SAME scenario from scratch and report what really differs (nothing is adjusted)."""
    b = request.get_json(silent=True) or {}
    try:
        start = float(b.get("start", 480)); end = float(b.get("end", 1320)); seed = int(b.get("seed", 42)); stress = bool(b.get("stress", True)); wd = int(b.get("weekday", 0)) % 7
    except (TypeError, ValueError):
        return jsonify({"error": "bad parameters"}), 400
    if end <= start: end += 1440
    res = {}
    with lock:
        for model in ("graph", "flat"):
            env = RailwayPPOEnv(start, end, seed=seed, stress_mode=stress, weekday=wd, obs_mode=model); agent, label = load_agent(env); res[model] = _run_full(env, agent, seed, label)
    real = {m: {(d["time"], d["train"]): d["label"] for d in res[m]["decisions"] if d["options"] > 1} for m in res}
    diff = [{"time": k[0], "train": k[1], "graph": real["graph"][k], "flat": real["flat"][k]} for k in sorted(real["graph"]) if k in real["flat"] and real["graph"][k] != real["flat"][k]]
    summ = {}
    for m in res:
        lab = [d["label"] for d in res[m]["decisions"] if d["options"] > 1]
        summ[m] = {"checkpoint": res[m]["checkpoint"], "real_choices": len(lab), "hold": lab.count("HOLD"), "proceed": lab.count("PROCEED"), "loop": lab.count("LOOP & OVERTAKE"),
                   "added_delay": res[m]["added_delay"], "reward": res[m]["reward"]}
    return jsonify({"summary": summ, "differing_decisions": diff[:40], "n_differing": len(diff), "only_graph": len(set(real["graph"]) - set(real["flat"])), "only_flat": len(set(real["flat"]) - set(real["graph"]))})


@app.get("/follow")
def follow():
    return Response(FOLLOW_PAGE, mimetype="text/html")


@app.post("/api/reset")
def reset():
    b = request.get_json(silent=True) or {}
    try:
        start = float(b.get("start", 480)); end = float(b.get("end", 1320))
        seed = int(b.get("seed", 42)); stress = bool(b.get("stress", False))
        weekday = int(b.get("weekday", 0)) % 7            # 0=Mon .. 6=Sun
        model = "flat" if b.get("model") == "flat" else "graph"
    except (TypeError, ValueError):
        return jsonify({"error": "bad parameters"}), 400
    if end <= start:                                       # overnight window, e.g. 22:00 -> 02:00
        end += 1440
    if end - start > 1440:
        return jsonify({"error": "Window must be 24 h or shorter"}), 400
    with lock:
        env = RailwayPPOEnv(start, end, seed=seed, stress_mode=stress, weekday=weekday, obs_mode=model)
        S["trace"] = {}; S["log"] = []; _hook_trace(env)          # record every train, every minute (for the per-train view)
        obs, _ = env.reset(seed=seed)
        agent, label = load_agent(env)
        S.update(env=env, agent=agent, obs=obs, tick=0, decisions=[], history=[],
                 checkpoint=label, stress=stress, finished=False)
        return jsonify(payload())


@app.post("/api/step")
def step():
    with lock:
        bad = need_env()
        if bad: return bad
        e, a = S["env"], S["agent"]
        if S["finished"]:
            return jsonify(payload())
        obs, done = e.advance_to_decision()
        if done:
            S["decisions"] = []; S["finished"] = True
            return jsonify(payload())
        mask = torch.as_tensor(e.action_mask(), dtype=torch.bool).unsqueeze(0)
        sv = torch.as_tensor(e.slot_valid()).unsqueeze(0)
        st = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            ctx = e.graph_context() if a.use_graph else None
            acts, _, val = a.act(st, mask, deterministic=True, slot_valid=sv, graph_contexts=[ctx] if a.use_graph else None)
            dists, _ = a.forward(st, mask, graph_contexts=[ctx] if a.use_graph else None)
        probs = [d.probs.squeeze(0).tolist() for d in dists]
        t_dec = float(e._rs["simulation_time_min"])
        _, reward, term, trunc, info = e.step(acts.squeeze(0).numpy())
        S["obs"] = e._obs; S["tick"] += 1
        S["finished"] = bool(term or trunc)
        S["decisions"] = []
        for i, d in enumerate(info["decisions"]):
            d = dict(d)
            d["probs"] = {"hold": probs[i][0], "grant_platform": probs[i][1], "grant_loop": probs[i][2]}
            S["decisions"].append(d)
            rec = _log_record(e, d, len(S["log"]) + 1, S["tick"], t_dec); rec["n_options"] = int(mask[0, i].sum()); S["log"].append(rec)
        S["history"].append({"tick": S["tick"], "sim_time_min": info["sim_time_min"],
                             "waiting": info["n_waiting"], "conflicts": info["n_conflicts"],
                             "delay": info["total_delay"], "reward": float(reward),
                             "value": float(val.item())})
        return jsonify(payload())


@app.post("/api/mode")
def mode():
    with lock:
        on = bool((request.get_json(silent=True) or {}).get("stress", False))
        S["stress"] = on
        if S["env"]:
            S["env"].set_stress_mode(on)
        return jsonify(payload())


@app.get("/api/graph")
def graph():
    """Clean, layout-ready route diagram. ?full=1 shows every active train."""
    with lock:
        bad = need_env()
        if bad: return bad
        return jsonify(build_route_view(S["env"], full=request.args.get("full") == "1"))


# --------------------------------------------------------------------------- #
# Dashboard UI
# --------------------------------------------------------------------------- #
FOLLOW_PAGE = (Path(__file__).with_name("follow_page.html")).read_text(encoding="utf-8")   # the auto-following "decision follower" page

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Railway Dispatch · DORG → GraphSAGE → PPO</title>
<style>
:root{--bg:#f3f5f9;--card:#fff;--ink:#16202f;--mute:#6b7688;--line:#e4e8ef;--blue:#2f6fed;--red:#e5484d;--amber:#f59f00;--green:#12a150;--violet:#7c4dff;--track:#9aa5b6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
header{position:sticky;top:0;z-index:5;background:rgba(255,255,255,.92);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:10px 22px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
header h1{font-size:16px;margin:0 6px 0 0}header .sp{flex:1}
button,input{font:inherit;padding:7px 12px;border:1px solid #cfd6e2;border-radius:8px;background:#fff;color:var(--ink)}
button{cursor:pointer;transition:.12s}button:hover{background:#f0f4fb}button.primary{background:var(--blue);border-color:var(--blue);color:#fff}button.primary:hover{background:#2459c6}
button.on{background:#e8f0ff;border-color:var(--blue);color:var(--blue)}
label{color:var(--mute);font-size:12px;display:flex;gap:6px;align-items:center}
.pill{padding:4px 10px;border-radius:999px;background:#e8f0ff;color:var(--blue);font-size:12px;font-weight:600}.pill.stress{background:#fff1dc;color:#b86e00}.pill.done{background:#e3f7ec;color:var(--green)}
.wrap{max-width:1500px;margin:0 auto;padding:18px 22px 40px}
.metrics{display:grid;grid-template-columns:repeat(7,1fr);gap:10px;margin-bottom:14px}
.m{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:11px 14px}.m span{color:var(--mute);font-size:12px}.m b{display:block;font-size:22px;margin-top:2px;font-variant-numeric:tabular-nums}
.m.bad b{color:var(--red)}.m.warn b{color:var(--amber)}
.grid{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(0,1fr);gap:14px}.full{grid-column:1/-1}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px}
.card h3{margin:0 0 4px;font-size:15px}.sub{color:var(--mute);font-size:12px;margin-bottom:10px}
.head{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:6px}.head h3{margin:0}
table{width:100%;border-collapse:collapse}th,td{padding:8px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}
th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--mute);font-weight:600}tbody tr:last-child td{border-bottom:0}
.mono{font-family:ui-monospace,Consolas,monospace}.empty{text-align:center;color:var(--mute);padding:22px}
.chip{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:700;background:#eef1f6;color:#44506a}
.chip.HOLD{background:#fdecec;color:var(--red)}.chip.KEEPPLATFORM,.chip.PROCEED,.chip.GRANTPLATFORM{background:#e3f7ec;color:var(--green)}.chip.YIELDTOLOOP,.chip.GRANTLOOP{background:#f0eaff;color:var(--violet)}
.chip.IN_TRANSIT{background:#e8f0ff;color:var(--blue)}.chip.WAITING{background:#fff1dc;color:#b86e00}.chip.HALTED{background:#eef1f6}.chip.HELD_LOOP{background:#f0eaff;color:var(--violet)}
.prob{min-width:92px;font-size:11px;color:var(--mute)}.bar{height:5px;border-radius:3px;background:#edf0f5;margin-top:3px;overflow:hidden}.bar i{display:block;height:100%;background:#b9c7e6}.bar i.best{background:var(--blue)}
.dec td:first-child{border-left:3px solid var(--blue)}
svg.route{width:100%;display:block;background:linear-gradient(#fbfcfe,#f6f8fb);border-radius:10px;border:1px solid var(--line)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--mute);margin-top:8px}.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.note{font-size:12px;color:var(--mute);margin-top:6px}
.spark{width:100%;height:110px}
.scroll{max-height:340px;overflow:auto}
@media(max-width:1100px){.metrics{grid-template-columns:repeat(4,1fr)}.grid{grid-template-columns:1fr}}
.tfgrid{display:grid;grid-template-columns:1fr 1fr;gap:14px}h4{margin:6px 0;font-size:13px}.dcard{border:1px solid var(--line);border-left:4px solid #9aa5b6;border-radius:8px;padding:8px 10px;margin-bottom:8px;background:#fcfdff}
.dcard.loop{border-left-color:#7c4dff}.dcard.hold{border-left-color:#e5484d}.dcard.go{border-left-color:#12a150}.dcard.forced{opacity:.55}.dtime{font-weight:700;font-variant-numeric:tabular-nums;margin-right:8px}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;font-weight:700;background:#eef1f6;color:#44506a;margin-left:6px}.badge.me{background:#e8f0ff;color:#2f6fed}.badge.other{background:#fff1dc;color:#b86e00}
.why{color:#44506a;font-size:12px;margin-top:3px}.pb{display:flex;gap:4px;margin-top:4px;font-size:10px;color:#6b7688}.pb span{flex:1}.pb i{display:block;height:4px;border-radius:2px;background:#d9e1f2;margin-top:2px}.pb i.best{background:#2f6fed}
@media(max-width:1000px){.tfgrid{grid-template-columns:1fr}}
</style></head><body>
<header>
 <h1>Railway Dispatch</h1><a href="/follow" style="font-size:12px;color:#2f6fed;text-decoration:none;border:1px solid #cfd6e2;border-radius:8px;padding:6px 10px">▶ Decision follower (auto)</a><span id="mode" class="pill">NOT INITIALISED</span>
 <button onclick="reset(false)">Realistic</button><button onclick="reset(true)">Stress test</button>
 <button class="primary" onclick="step()">Step ▸</button><button id="play" onclick="play()">▶ Play</button>
 <label>Speed <input id="speed" type="range" min="1" max="8" value="3" style="padding:0;width:80px"></label>
 <span class="sp"></span>
 <label>Model <select id="model" style="padding:7px 8px;border:1px solid #cfd6e2;border-radius:8px"><option value="graph">Graph PPO (GraphSAGE)</option><option value="flat">Flat PPO (no graph)</option></select></label>
 <label>Day <select id="wd" style="padding:7px 8px;border:1px solid #cfd6e2;border-radius:8px"><option value="0">Mon</option><option value="1">Tue</option><option value="2">Wed</option><option value="3">Thu</option><option value="4">Fri</option><option value="5">Sat</option><option value="6">Sun</option></select></label>
 <label>Start <input id="start" type="time" value="08:00"></label>
 <label>End <input id="end" type="time" value="12:00"></label>
 <label>Seed <input id="seed" type="number" value="42" style="width:70px"></label>
</header>
<div class="wrap">
 <div class="metrics">
  <div class="m"><span>Clock</span><b id="clock">--:--</b></div><div class="m"><span>Tick</span><b id="tick">0</b></div>
  <div class="m"><span>Active trains</span><b id="active">0</b></div><div class="m" id="mw"><span>Waiting</span><b id="waiting">0</b></div>
  <div class="m" id="mc"><span>Conflicts</span><b id="conflicts">0</b></div><div class="m" id="md"><span>Controllable delay (min)</span><b id="delay">0</b></div>
  <div class="m"><span>Last reward</span><b id="reward">0.00</b></div>
 </div>
 <div class="grid">
  <div class="card full"><h3>PPO policy — this decision tick</h3>
   <div class="sub">Rows only appear when a real controllable decision exists. Probabilities come from the live action mask.</div>
   <table><thead><tr><th>Train</th><th>Station / section</th><th>Conflict</th><th>Action</th><th>Hold</th><th>Proceed / platform</th><th>Loop</th></tr></thead><tbody id="dec"></tbody></table></div>

  <div class="card full" id="tf-card"><h3>Train focus: every decision, station by station</h3>
  <div class="sub">Pick a train (or click any train in the tables). You see each decision that involved it, why it was taken, and how a loop &amp; overtake plays out at the loop station.</div>
  <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px"><select id="tf-train" style="padding:7px 10px;border:1px solid #cfd6e2;border-radius:8px;min-width:360px" onchange="focusTrain(this.value)"></select>
   <label><input type="checkbox" id="tf-real" checked onchange="renderFocus()"> hide forced decisions (only one option)</label><span id="tf-head" style="font-weight:600"></span></div>
  <div class="tfgrid"><div><h4>Decisions about this train</h4><div id="tf-decs" class="scroll" style="max-height:430px"></div></div>
   <div><h4>Loop &amp; overtake, as it happens</h4><div id="tf-ev" style="margin-bottom:6px"></div><svg id="tf-anim" height="250" style="background:#f8fafc;border:1px solid var(--line);border-radius:10px"></svg>
    <div style="display:flex;gap:8px;align-items:center;margin-top:6px"><button id="tf-play" onclick="tfPlay()">▶ Play</button><input type="range" id="tf-t" style="flex:1;padding:0" oninput="tfDraw()"><b id="tf-clock" style="min-width:60px;text-align:right"></b></div><div id="tf-log" class="sub" style="margin-top:6px"></div></div></div>
  <h4 style="margin:12px 0 4px">Station by station (this train)</h4><div class="scroll" style="max-height:300px"><table><thead><tr><th>Station</th><th>Stops?</th><th>Scheduled</th><th>Simulated</th><th>Delay (min)</th><th>Note</th></tr></thead><tbody id="tf-j"></tbody></table></div></div>
  <div class="card"><div class="head"><h3>Route diagram</h3><span class="sp" style="flex:1"></span>
    <button id="bFocus" class="on" onclick="setFull(false)">Focus</button><button id="bFull" onclick="setFull(true)">All trains</button></div>
   <div class="sub" id="gsub">Only trains involved in conflicts, waits, decisions or delays are drawn; quiet stretches are collapsed.</div>
   <svg id="graph" class="route" height="300"></svg>
   <div class="legend"><span><i style="background:var(--blue)"></i>normal</span><span><i style="background:var(--amber)"></i>delayed</span><span><i style="background:var(--violet)"></i>waiting</span><span><i style="background:var(--red)"></i>conflict</span><span><i style="background:#fff;border:2px solid var(--blue)"></i>PPO decision</span><span>▲ up direction ▼ down</span><span><i style="background:repeating-linear-gradient(45deg,#f59f00,#f59f00 3px,#fff 3px,#fff 6px)"></i>disruption</span></div>
   <div class="note" id="gnote"></div></div>

  <div style="display:grid;gap:14px;align-content:start;min-width:0">
   <div class="card"><h3>Live conflicts</h3><div class="scroll" style="max-height:150px"><table><thead><tr><th>Type</th><th>Location</th><th>Trains</th></tr></thead><tbody id="conf"></tbody></table></div></div>
   <div class="card"><h3>Trend</h3><div class="sub">delay (blue) · conflicts (red) · waiting (violet)</div><svg id="spark" class="spark" viewBox="0 0 300 110" preserveAspectRatio="none"></svg></div>
  </div>

  <div class="card full"><h3>Active trains <span class="sub" id="tcount"></span></h3><div class="scroll"><table><thead><tr><th>Train</th><th>Status</th><th>Station / section</th><th>Delay</th><th>Priority</th></tr></thead><tbody id="trains"></tbody></table></div></div>
 </div>
</div>
<script>
const $=id=>document.getElementById(id);let timer=null,busy=false,fullMode=false,lastP=null;
const NS='http://www.w3.org/2000/svg';
async function api(u,o){const r=await fetch(u,o);const j=await r.json();if(!r.ok)throw new Error(j.error||r.statusText);return j}
const DN=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];const dlabel=(t,wd)=>DN[(wd+Math.floor(t/1440))%7]+' ';
const mins=v=>String(Math.floor(v/60)%24).padStart(2,'0')+':'+String(Math.floor(v)%60).padStart(2,'0');
const tm=s=>{const[h,m]=s.split(':').map(Number);return h*60+m};
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

async function reset(stress){stop();try{const p=await api('/api/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({start:tm($('start').value),end:tm($('end').value),seed:+$('seed').value,weekday:+$('wd').value,model:$('model').value,stress})});render(p);FT=null;FDATA=null;$('tf-decs').innerHTML='';$('tf-head').textContent='';$('tf-j').innerHTML='';tfClear();loadTrains();await drawGraph()}catch(e){alert(e.message)}}
async function step(){if(busy)return;busy=true;try{const p=await api('/api/step',{method:'POST'});render(p);refreshFocus();await drawGraph();if(p.finished)stop()}catch(e){stop();console.error(e)}finally{busy=false}}
function stop(){if(timer){clearTimeout(timer);timer=null}$('play').textContent='▶ Play'}
function play(){if(timer){stop();return}$('play').textContent='■ Stop';const loop=async()=>{await step();if(timer)timer=setTimeout(loop,1100-$('speed').value*120)};timer=setTimeout(loop,0)}
function setFull(f){fullMode=f;$('bFocus').classList.toggle('on',!f);$('bFull').classList.toggle('on',f);drawGraph()}

function render(p){lastP=p;
 const m=$('mode');m.textContent=(p.finished?'FINISHED · ':'')+(p.ready!==false?DN[p.weekday||0]+' · ':'')+(p.mode||'').replace(/_/g,' ').toUpperCase()+(p.checkpoint?' · '+p.checkpoint:'');m.className='pill'+(p.finished?' done':p.stress_mode?' stress':'');
 $('clock').textContent=dlabel(p.sim_time_min||0,p.weekday||0)+mins(p.sim_time_min||0);$('tick').textContent=p.tick;$('active').textContent=p.trains.length;
 $('waiting').textContent=p.waiting.length;$('conflicts').textContent=p.conflicts.length;
 $('delay').textContent=p.metrics?(+p.metrics.delay).toFixed(1):'0';$('reward').textContent=p.metrics?(+p.metrics.reward).toFixed(2):'0.00';
 $('mw').className='m'+(p.waiting.length?' warn':'');$('mc').className='m'+(p.conflicts.length?' bad':'');$('md').className='m'+(p.metrics&&p.metrics.delay>0?' warn':'');
 $('dec').innerHTML=p.decisions.length?p.decisions.map(d=>{const pr=[d.probs.hold,d.probs.grant_platform,d.probs.grant_loop],mx=Math.max(...pr);
  const lab=d.action_label.replace(/\s/g,'');const cell=(v,i)=>`<td class="prob">${(100*v).toFixed(1)}%<div class="bar"><i class="${v===mx?'best':''}" style="width:${100*v}%"></i></div></td>`;
  return `<tr class="dec" style="cursor:pointer" title="Click to follow this train" onclick="focusTrain(${d.train_number})"><td class="mono">${d.train_number}</td><td>${esc(d.station||d.location||'—')}</td><td>${esc(d.conflict)}</td><td><span class="chip ${lab}">${esc(d.action_label)}</span></td>${pr.map(cell).join('')}</tr>`}).join('')
  :`<tr><td colspan="7" class="empty">${p.ready===false?'Press Realistic or Stress test to begin.':p.finished?'Simulation finished.':'No controllable decision on this tick.'}</td></tr>`;
 $('conf').innerHTML=p.conflicts.length?p.conflicts.map(c=>`<tr><td><span class="chip">${esc(c.type)}</span></td><td>${esc(c.location)}</td><td class="mono">${c.trains.join(', ')}</td></tr>`).join(''):'<tr><td colspan="3" class="empty">No active conflicts</td></tr>';
 $('tcount').textContent='('+p.trains.length+', most delayed first)';
 $('trains').innerHTML=p.trains.map(t=>`<tr style="cursor:pointer" title="Click to follow this train" onclick="focusTrain(${t.train_number})"><td class="mono">${t.train_number}</td><td><span class="chip ${t.status}">${esc(t.status)}</span></td><td>${esc(t.station||((t.from_station||'')+' → '+(t.to_station||'')))}</td><td>${(+t.delay_min).toFixed(1)}</td><td>${t.priority}</td></tr>`).join('');
 spark(p.history||[])}

function spark(h){const s=$('spark');s.innerHTML='';if(h.length<2)return;
 const line=(key,col)=>{const v=h.map(x=>+x[key]),mx=Math.max(...v,1);const pts=v.map((y,i)=>`${(i/(v.length-1))*298+1},${106-(y/mx)*98}`).join(' ');
  const pl=document.createElementNS(NS,'polyline');pl.setAttribute('points',pts);pl.setAttribute('fill','none');pl.setAttribute('stroke',col);pl.setAttribute('stroke-width','1.8');pl.setAttribute('vector-effect','non-scaling-stroke');s.appendChild(pl)};
 line('delay','#2f6fed');line('conflicts','#e5484d');line('waiting','#7c4dff')}

/* ---------------- route diagram ---------------- */
const ROLE={decision:'#2f6fed',conflict:'#e5484d',waiting:'#7c4dff',delayed:'#f59f00',normal:'#2f6fed'};
function el(n,a,txt,parent){const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);if(txt!=null)e.textContent=txt;(parent||svg).appendChild(e);return e}
let svg;
async function drawGraph(){svg=$('graph');let g;try{g=await api('/api/graph'+(fullMode?'?full=1':''))}catch(e){return}
 const W=svg.clientWidth||900,L=g.max_lanes,laneH=26,pad=30,top=18+L*laneH,mid=top+8,H=mid+L*laneH+58;
 svg.setAttribute('height',H);svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.innerHTML='';
 const n=g.items.length,step=(W-2*pad)/Math.max(n-1,1),X=x=>pad+x*step;
 // disruption / conflict spans (behind the track)
 g.spans.forEach(s=>{const c=s.kind&&s.kind.startsWith('conflict')?'#e5484d':'#f59f00';
  const x1=X(s.x1),x2=Math.max(X(s.x2),x1+10);const r=el('rect',{x:x1-6,y:mid-9,width:x2-x1+12,height:18,rx:9,fill:c,opacity:.16});
  el('title',{},(s.kind||'')+'',r)});
 // track
 el('line',{x1:pad-14,x2:W-pad+14,y1:mid,y2:mid,stroke:'#9aa5b6','stroke-width':3,'stroke-linecap':'round'});
 // stations
 const showEvery=Math.ceil(n/Math.max(Math.floor((W-2*pad)/62),1));
 g.items.forEach((it,i)=>{const x=X(it.x);
  if(it.type==='station'){const c=el('circle',{cx:x,cy:mid,r:5,fill:'#fff',stroke:'#59657a','stroke-width':2});el('title',{},it.name+' ('+it.code+') · '+it.km+' km',c);
   const t=el('text',{x:x,y:mid+22+(i%2)*13,'text-anchor':'middle','font-size':10.5,fill:'#3a455a','font-weight':600},it.code)}
  else{el('text',{x:x,y:mid+4,'text-anchor':'middle','font-size':11,fill:'#8a95a8'},'⋯');el('text',{x:x,y:mid+22,'text-anchor':'middle','font-size':9.5,fill:'#8a95a8'},it.n+' stops')}});
 // trains
 const P={};g.trains.forEach(t=>{const up=t.side==='up',y=up?mid-18-t.lane*laneH:mid+44+t.lane*laneH;P[t.id]={x:X(t.x),y,up}});
 g.links.forEach(l=>{const a=P[l.a],b=P[l.b];if(!a||!b)return;const mx=(a.x+b.x)/2,my=(a.y+b.y)/2+(a.up===b.up?(a.up?-14:14):0);
  el('path',{d:`M${a.x},${a.y} Q${mx},${my} ${b.x},${b.y}`,fill:'none',stroke:'#e5484d','stroke-width':1.6,'stroke-dasharray':'4 3',opacity:.85})});
 g.trains.forEach(t=>{const p=P[t.id],col=ROLE[t.role];
  el('line',{x1:p.x,x2:p.x,y1:p.y+(p.up?10:-10),y2:mid+(p.up?-4:4),stroke:col,'stroke-width':1.2,opacity:.55});
  const grp=el('g',{});const w=44;
  el('rect',{x:p.x-w/2,y:p.y-10,width:w,height:20,rx:10,fill:t.role==='decision'?'#fff':col,stroke:col,'stroke-width':t.role==='decision'?2.4:0},null,grp);
  el('text',{x:p.x,y:p.y+4,'text-anchor':'middle','font-size':10.5,'font-weight':700,fill:t.role==='decision'?col:'#fff'},t.id,grp);
  const tip=el('title',{},`Train ${t.id}\n${t.status}${t.at?' at '+t.at:' '+t.from+' → '+t.to}\ndelay ${t.delay.toFixed(1)} min · priority ${t.priority}\nrole: ${t.role}`,grp);
  el('circle',{cx:p.x,cy:mid,r:3.2,fill:col,stroke:'#fff','stroke-width':1.2})});
 el('text',{x:pad-14,y:16,'font-size':10.5,fill:'#8a95a8'},'▲ up direction');el('text',{x:pad-14,y:H-8,'font-size':10.5,fill:'#8a95a8'},'▼ down direction');
 $('gnote').textContent=`Showing ${g.shown} of ${g.total_active} active trains`+(g.hidden?` · ${g.hidden} more hidden (too crowded)`:'')+(g.full?'':' · switch to “All trains” for the full picture');
}
window.addEventListener('resize',()=>{if(lastP&&lastP.ready!==false)drawGraph()});
$('wd').value=(new Date().getDay()+6)%7;

/* ================= Train focus: per-train decisions, journey, loop & overtake animation ================= */
let FT=null,FDATA=null,EVI=0,tfTimer=null;
async function loadTrains(){try{const j=await api('/api/trains'),s=$('tf-train'),cur=s.value;s.innerHTML='<option value="">- pick a train -</option>'+j.trains.map(t=>`<option value="${t.train}">${t.deciding_now?'★ DECIDING NOW · ':t.blocked_now?'⚠ blocked now · ':''}${t.train} · ${esc(t.name)} · P${t.priority}${t.decided+t.involved?' · '+t.decided+' decided / '+t.involved+' blocked-by':''}</option>`).join('');if(cur)s.value=cur}catch(e){}}
function focusTrain(tn){if(!tn)return;FT=+tn;const s=$('tf-train');if(![...s.options].some(o=>+o.value===FT)){s.add(new Option(String(FT),FT))}s.value=FT;EVI=0;refreshFocus(true);const c=$('tf-card');if(c)c.scrollIntoView({behavior:'smooth',block:'start'})}
async function refreshFocus(scroll){if(!FT)return;try{FDATA=await api('/api/train/'+FT);renderFocus()}catch(e){$('tf-decs').innerHTML='<div class="empty">'+esc(e.message)+'</div>'}loadTrains()}
const pct=v=>(100*v).toFixed(0)+'%';
function reasonText(r,me){const nm=x=>x.train+' (P'+x.priority+')',bl=(r.blocked_info||[]).map(nm).join(', ');
 if(r.kind==='waiting')return r.n_options>1?'Train waits at '+esc(r.station||'')+' for a platform; the agent chose among '+r.n_options+' options.':'Train waits at '+esc(r.station||'')+'. Only one option was possible here (forced), so no real choice was made.';
 const base=r.train_number===me?'Train '+r.train_number+' (P'+r.priority+') is the least important in the conflict at '+esc(r.location||'the section')+' and blocks '+(r.n_blocked)+' more important train(s): '+bl+'.':'Train '+me+' was blocked by '+r.train_number+' (P'+r.priority+') at '+esc(r.location||'the section')+'.';
 const cost=' Blocking costs them about '+r.blocking_cost+' points per minute; holding '+r.train_number+' costs '+r.hold_cost+' per minute.';
 const out=r.action===0?' Decision: HOLD: it waits 1 minute, which removes the blocking for that minute.':r.action===2?' Decision: LOOP &amp; OVERTAKE: it goes into the loop at '+esc(r.loop_station_name||r.loop_station||'')+' (one-off '+(FDATA?1.5:1.5)+' min cost) and the conflict is cleared for '+(FDATA?FDATA.loop_pass_min:10)+' minutes.':' Decision: PROCEED: nobody is held, the blocked trains keep losing time while the conflict lasts.';
 return base+cost+out}
function renderFocus(){if(!FDATA)return;const D=FDATA,hide=$('tf-real').checked;
 $('tf-head').innerHTML='Train '+D.train+' · '+esc(D.name)+' · priority '+D.priority+' · <span class="chip '+D.status+'">'+D.status+'</span> · delay '+D.delay+' min';
 const ds=D.decisions.filter(r=>!(hide&&r.n_options===1)),mx=r=>Math.max(r.probs.hold,r.probs.grant_platform,r.probs.grant_loop);
 $('tf-decs').innerHTML=ds.length?ds.slice().reverse().map(r=>{const cls=r.n_options===1?'forced':r.action===2?'loop':r.action===0?'hold':'go',me=r.train_number===D.train;
  const pb=[['Hold',r.probs.hold],[r.kind==='waiting'?'Platform':'Proceed',r.probs.grant_platform],['Loop',r.probs.grant_loop]].map(([l,v])=>'<span>'+l+' '+pct(v)+'<i class="'+(v===mx(r)?'best':'')+'" style="width:'+Math.max(2,100*v)+'%"></i></span>').join('');
  return '<div class="dcard '+cls+'"><span class="dtime">'+mins(r.time)+'</span><span class="chip">'+esc(r.action_label)+'</span><span class="badge '+(me?'me':'other')+'">'+(me?'decided about this train':'this train was blocked')+'</span>'+(r.n_options===1?'<span class="badge">forced</span>':'')+'<div class="why">'+reasonText(r,D.train)+'</div><div class="pb">'+pb+'</div></div>'}).join(''):'<div class="empty">No real decision involved this train yet. Press Step / Play, or untick "hide forced decisions".</div>';
 $('tf-j').innerHTML=D.journey.map(j=>'<tr><td><b>'+esc(j.station)+'</b> <span style="color:#6b7688">'+esc(j.name)+'</span></td><td>'+(j.halt?'stops':'passes')+'</td><td>'+mins(j.scheduled)+'</td><td>'+(j.sim_time==null?'-':mins(j.sim_time))+'</td><td>'+(j.delay==null?'-':j.delay)+'</td><td>'+(j.loop?'<span class="badge" style="background:#f0eaff;color:#7c4dff">waited in loop here</span>':'')+'</td></tr>').join('');
 const E=D.loop_events;if(EVI>=E.length)EVI=0;
 $('tf-ev').innerHTML=E.length?'Loop events for this train: '+E.map((e,i)=>'<button class="'+(i===EVI?'on':'')+'" onclick="EVI='+i+';renderFocus()">'+mins(e.time)+' at '+esc(e.station)+(e.train===D.train?' (this train waits)':' (this train overtakes)')+'</button>').join(' '):'<span class="sub">No loop &amp; overtake involved this train yet. Try Monday + seed 5, or Friday + seed 0 in Stress test.</span>';
 tfSetup()}
function tfClear(){const s=$('tf-anim');if(s)s.innerHTML='';$('tf-log').textContent='';$('tf-ev').innerHTML='';$('tf-clock').textContent=''}
function tfEv(){return FDATA&&FDATA.loop_events.length?FDATA.loop_events[Math.min(EVI,FDATA.loop_events.length-1)]:null}
function tfLeave(e){const ps=e.blockers.map(b=>b.pass_time).filter(x=>x!=null);return Math.max(e.time+3,(ps.length?Math.max(...ps):e.until)+0.5)}
function tfSetup(){const e=tfEv(),s=$('tf-anim'),r=$('tf-t');if(!e){s.innerHTML='';$('tf-log').textContent='';$('tf-clock').textContent='';return}
 const lo=Math.floor(e.time-4),hi=Math.ceil(tfLeave(e)+6);const keep=(+r.min===lo&&+r.max===hi);r.min=lo;r.max=hi;if(!keep)r.value=lo;tfDraw()}
function tfDraw(){const e=tfEv(),s=$('tf-anim');if(!e)return;s.innerHTML='';const W=s.clientWidth||620,H=250,t=+$('tf-t').value,cx=W/2,main=70,loop=160,sw=Math.min(110,W*.18),k=Math.min(16,W/45),kb=Math.min(34,W/20);$('tf-clock').textContent=mins(t);
 const L=el('line',{},null,s);s.removeChild(L);
 el('text',{x:12,y:20,'font-size':12,'font-weight':700,fill:'#16202f'},esc(e.station_name||e.station)+' ('+e.station+')',s);el('text',{x:12,y:36,'font-size':10.5,fill:'#6b7688'},'main line above, loop line below',s);
 el('line',{x1:0,x2:W,y1:main,y2:main,stroke:'#9aa5b6','stroke-width':4},null,s);
 el('path',{d:'M'+(cx-sw)+','+main+' L'+(cx-sw+30)+','+loop+' L'+(cx+sw-30)+','+loop+' L'+(cx+sw)+','+main,fill:'none',stroke:'#7c4dff','stroke-width':4,opacity:.55},null,s);
 el('text',{x:cx,y:loop+22,'text-anchor':'middle','font-size':10.5,fill:'#7c4dff'},'loop line',s);
 const tl=tfLeave(e),t0=e.time,lerp=(a,b,f)=>[a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f];let p,st;
 const A=[cx-sw,main],B=[cx-sw+30,loop],C=[cx,loop],D2=[cx+sw-30,loop],E2=[cx+sw,main];
 if(t<t0){p=[A[0]-(t0-t)*k,main];st='approaching the loop station'}else if(t<t0+1){p=lerp(A,B,t-t0);st='entering the loop'}else if(t<t0+3){p=lerp(B,C,(t-t0-1)/2);st='in the loop'}
 else if(t<tl){p=C;st='waiting in the loop'}else if(t<tl+2){p=lerp(C,D2,(t-tl)/2);st='leaving the loop'}else if(t<tl+3){p=lerp(D2,E2,t-tl-2);st='back on the main line'}else{p=[E2[0]+(t-tl-3)*k,main];st='continuing'}
 const dot=(x,y,c,lab,sub)=>{if(x<-30||x>W+30)return;el('circle',{cx:x,cy:y,r:11,fill:c,stroke:'#fff','stroke-width':2},null,s);el('text',{x:x,y:y-17,'text-anchor':'middle','font-size':10.5,'font-weight':700,fill:c},lab,s);if(sub)el('text',{x:x,y:y+27,'text-anchor':'middle','font-size':9.5,fill:'#6b7688'},sub,s)};
 dot(p[0],p[1],'#7c4dff',e.train+' (P'+e.priority+')',st);
 e.blockers.forEach((b,i)=>{if(b.pass_time!=null)dot(cx+(t-b.pass_time)*kb,main,'#12a150',b.train+' (P'+b.priority+')',t>=b.pass_time?'passed '+mins(b.pass_time):'due '+mins(b.pass_time));else if(b.scheduled!=null)el('text',{x:12,y:210+i*14,'font-size':10.5,fill:'#6b7688'},b.train+' (P'+b.priority+') has not reached this station yet (scheduled '+mins(b.scheduled)+')',s)});
 const lines=['<b>'+mins(e.time)+'</b> decision: LOOP &amp; OVERTAKE for train '+e.train+' (P'+e.priority+'), one-off cost '+e.cost_min+' min'].concat(e.blockers.map(b=>b.pass_time!=null?'<b>'+mins(b.pass_time)+'</b> train '+b.train+' (P'+b.priority+') passes on the main line':'train '+b.train+' (P'+b.priority+') has not passed yet')).concat(['<b>'+mins(tl)+'</b> train '+e.train+' leaves the loop and continues']);
 $('tf-log').innerHTML=lines.join('<br>')+'<br><i>As modelled: the loop pass costs the waiting train '+e.cost_min+' min and clears the conflict for '+(FDATA?FDATA.loop_pass_min:10)+' min; this picture shows what that means at the station.</i>'}
function tfPlay(){if(tfTimer){clearInterval(tfTimer);tfTimer=null;$('tf-play').textContent='▶ Play';return}$('tf-play').textContent='■ Pause';tfTimer=setInterval(()=>{const r=$('tf-t');let v=+r.value+0.12;if(v>+r.max)v=+r.min;r.value=v;tfDraw()},60)}

api('/api/state').then(p=>{if(p.ready){$('wd').value=p.weekday;$('model').value=p.model}render(p);if(p.ready){drawGraph();loadTrains()}else reset(false)}).catch(()=>reset(false));
</script></body></html>"""

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
