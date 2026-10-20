"""Flat (NO graph) state for the plain-RL baseline.

Same operational information as the graph model's explicit features, but with
no DORG, no GraphSAGE, no node embeddings, no neighbourhood/topology context:
    5  global scalars
   30  per-waiting-train features (5 x 6)
   40  conflict descriptors (8 x 5)
= 75 numbers (+30 decision-slot features added by the environment -> 105).
"""
from __future__ import annotations
import numpy as np
from .state_builder import (MAX_WAITING, MAX_CONFLICTS, _MAX_TIME_MIN, _MAX_TRAINS,
                            _MAX_DELAY_MIN, _conflict_onehot)

FLAT_BASE_DIM = 5 + MAX_WAITING * 6 + MAX_CONFLICTS * 5


def build_flat_state(rs: dict) -> np.ndarray:
    s = np.zeros(FLAT_BASE_DIM, np.float32)
    trains = rs.get("trains", []); waiting = rs.get("waiting_trains", []); conflicts = rs.get("conflicts", [])
    n_active = sum(1 for t in trains if t.get("status") not in ("NOT_STARTED", "TERMINATED"))
    tot = sum(t.get("delay_min", 0.0) for t in trains)
    s[0] = (rs.get("simulation_time_min", 0.0) % _MAX_TIME_MIN) / _MAX_TIME_MIN
    s[1] = n_active / _MAX_TRAINS; s[2] = len(waiting) / MAX_WAITING; s[3] = len(conflicts) / MAX_CONFLICTS
    s[4] = min(tot, _MAX_DELAY_MIN * _MAX_TRAINS) / (_MAX_DELAY_MIN * _MAX_TRAINS)
    by = {t["train_number"]: t for t in trains}
    pocc = rs.get("platform_occupancy", {}); locc = rs.get("loop_line_occupancy", {}); p = 5
    for tn, st in waiting[:MAX_WAITING]:
        t = by.get(tn, {}); status = t.get("status", "WAITING")
        s[p:p + 6] = [min(float(t.get("delay_min", 0.0)), _MAX_DELAY_MIN) / _MAX_DELAY_MIN,
                      min(float(t.get("priority", 3)), 6.0) / 6.0, float(status == "WAITING"), float(status == "HELD_LOOP"),
                      min(len(pocc.get(st, {})), 10) / 10.0, min(len(locc.get(st, {})), 5) / 5.0]
        p += 6
    p = 5 + MAX_WAITING * 6
    prio = {t["train_number"]: t.get("priority", 3) for t in trains}
    for c in conflicts[:MAX_CONFLICTS]:
        pv = [prio.get(x, 3) for x in c.get("trains", [])]
        s[p:p + 4] = _conflict_onehot(c.get("type", "track"))
        s[p + 4] = (max(pv) - min(pv)) / 6.0 if len(pv) >= 2 else 0.0
        p += 5
    return s
