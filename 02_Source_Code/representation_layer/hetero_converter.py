"""
Heterogeneous Graph Converter
==============================
Converts a tick's DORG (networkx.MultiDiGraph produced by DORGBuilder)
into a PyTorch Geometric HeteroData object ready for the GraphSAGE encoder.

Node feature vectors
---------------------
Every node type gets a fixed-size float32 feature vector.  Absent /
unknown attributes are replaced by 0.

    station  [5]:  distance_km_norm, platforms_norm, loop_line_count_norm,
                   is_junction, is_terminus
    platform [2]:  platform_id_norm, is_occupied
    loop     [2]:  loop_id_norm, is_occupied
    train    [11]: 6×status_onehot, delay_min_norm, speed_kmph_norm,
                   position_km_norm, priority_norm, is_conflicting

Edge index tensors
-------------------
One edge_index (2, E) tensor per relation type.  The converter maps node
string IDs to integer indices per type, returning the index maps so
downstream code can look up which node corresponds to which index.

Relations exported (10 total):
    (station, connects,          station)
    (station, has_platform,      platform)
    (station, has_loop,          loop)
    (train,   at,                station)
    (train,   transiting,        station)
    (train,   waiting,           station)
    (train,   occupies_platform, platform)
    (train,   occupies_loop,     loop)
    (train,   conflict,          train)
    (train,   precedes,          train)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData

# Normalisation constants (rough corridor-scale values)
_MAX_DIST_KM   = 400.0   # Chennai–Bangalore corridor
_MAX_PLATFORMS = 10.0
_MAX_LOOPS     = 6.0
_MAX_PLATFORM_ID = 10.0
_MAX_LOOP_ID     = 6.0
_MAX_DELAY_MIN   = 120.0
_MAX_SPEED_KMPH  = 160.0
_MAX_POS_KM      = 400.0
_MAX_PRIORITY    = 6.0

_STATUS_ORDER = ["NOT_STARTED", "IN_TRANSIT", "HALTED", "HELD_LOOP", "WAITING", "TERMINATED"]
_STATUS_IDX   = {s: i for i, s in enumerate(_STATUS_ORDER)}

_JUNCTIONS = {"KPD", "JTJ", "BWT", "SA"}   # known junction codes on the corridor


def _station_features(data: dict) -> List[float]:
    code = data.get("code", "")
    return [
        data.get("distance_km", 0.0) / _MAX_DIST_KM,
        data.get("platforms", 0) / _MAX_PLATFORMS,
        data.get("loop_line_count", 0) / _MAX_LOOPS,
        float(code in _JUNCTIONS),
        0.0,   # is_terminus placeholder (could be filled from timetable)
    ]


def _platform_features(data: dict, occupied: bool) -> List[float]:
    return [
        data.get("platform_id", 1) / _MAX_PLATFORM_ID,
        float(occupied),
    ]


def _loop_features(data: dict, occupied: bool) -> List[float]:
    return [
        data.get("loop_id", 1) / _MAX_LOOP_ID,
        float(occupied),
    ]


def _train_features(data: dict, is_conflicting: bool) -> List[float]:
    onehot = [0.0] * len(_STATUS_ORDER)
    status = data.get("status", "NOT_STARTED")
    if isinstance(status, str):
        onehot[_STATUS_IDX.get(status, 0)] = 1.0
    else:
        onehot[_STATUS_IDX.get(str(status), 0)] = 1.0
    return onehot + [
        min(data.get("delay_min", 0.0), _MAX_DELAY_MIN) / _MAX_DELAY_MIN,
        data.get("speed_kmph", 0.0) / _MAX_SPEED_KMPH,
        data.get("position_km", 0.0) / _MAX_POS_KM,
        data.get("priority", 3) / _MAX_PRIORITY,
        float(is_conflicting),
    ]


class HeteroConverter:
    """
    Stateless converter: call ``convert(graph)`` once per tick to get a
    fresh HeteroData object.

    Returns
    -------
    data : HeteroData
    node_maps : dict[str, dict[str, int]]
        Maps node-type → {node_id_string → integer_index}
        e.g. node_maps["train"]["T:12345"] == 3
    """

    RELATIONS: List[Tuple[str, str, str]] = [
        ("station", "connects",          "station"),
        ("station", "has_platform",      "platform"),
        ("station", "has_loop",          "loop"),
        ("train",   "at",                "station"),
        ("train",   "transiting",        "station"),
        ("train",   "waiting",           "station"),
        ("train",   "occupies_platform", "platform"),
        ("train",   "occupies_loop",     "loop"),
        ("train",   "conflict",          "train"),
        ("train",   "precedes",          "train"),
    ]

    def convert(
        self,
        graph: nx.MultiDiGraph,
    ) -> Tuple[HeteroData, Dict[str, Dict[str, int]]]:

        # ── 1. collect nodes per type ──────────────────────────────────────
        node_lists: Dict[str, List[str]] = {
            "station": [], "platform": [], "loop": [], "train": []
        }
        for nid, ndata in graph.nodes(data=True):
            nt = ndata.get("node_type", "station")
            if nt in node_lists:
                node_lists[nt].append(nid)

        node_maps: Dict[str, Dict[str, int]] = {
            nt: {nid: i for i, nid in enumerate(ids)}
            for nt, ids in node_lists.items()
        }

        # ── 2. find occupied platforms / loops ─────────────────────────────
        occupied_platforms: set = set()
        occupied_loops: set     = set()
        conflicting_trains: set = set()
        for u, v, d in graph.edges(data=True):
            et = d.get("edge_type", "")
            if et == "occupies_platform":
                occupied_platforms.add(v)
            elif et == "occupies_loop":
                occupied_loops.add(v)
            elif et == "conflict":
                conflicting_trains.add(u)
                conflicting_trains.add(v)

        # ── 3. build node feature tensors ──────────────────────────────────
        data = HeteroData()

        def _feat(nt: str, fn) -> torch.Tensor:
            rows = []
            for nid in node_lists[nt]:
                nd = graph.nodes[nid]
                rows.append(fn(nid, nd))
            if not rows:
                # empty placeholder so PyG doesn't crash on missing node type
                n_feat = {"station": 5, "platform": 2, "loop": 2, "train": 11}[nt]
                return torch.zeros((0, n_feat), dtype=torch.float)
            return torch.tensor(rows, dtype=torch.float)

        data["station"].x = _feat(
            "station",
            lambda nid, nd: _station_features(nd),
        )
        data["platform"].x = _feat(
            "platform",
            lambda nid, nd: _platform_features(nd, nid in occupied_platforms),
        )
        data["loop"].x = _feat(
            "loop",
            lambda nid, nd: _loop_features(nd, nid in occupied_loops),
        )
        data["train"].x = _feat(
            "train",
            lambda nid, nd: _train_features(nd, nid in conflicting_trains),
        )

        # ── 4. build edge index tensors per relation ───────────────────────
        edge_buckets: Dict[Tuple[str, str, str], Tuple[List[int], List[int]]] = {
            rel: ([], []) for rel in self.RELATIONS
        }

        for u, v, ed in graph.edges(data=True):
            et = ed.get("edge_type", "")
            u_type = graph.nodes[u].get("node_type", "station")
            v_type = graph.nodes[v].get("node_type", "station")
            key = (u_type, et, v_type)
            if key in edge_buckets and u in node_maps[u_type] and v in node_maps[v_type]:
                edge_buckets[key][0].append(node_maps[u_type][u])
                edge_buckets[key][1].append(node_maps[v_type][v])

        for (src_type, rel, dst_type), (srcs, dsts) in edge_buckets.items():
            if srcs:
                data[src_type, rel, dst_type].edge_index = torch.tensor(
                    [srcs, dsts], dtype=torch.long
                )
            else:
                data[src_type, rel, dst_type].edge_index = torch.zeros(
                    (2, 0), dtype=torch.long
                )

        return data, node_maps
