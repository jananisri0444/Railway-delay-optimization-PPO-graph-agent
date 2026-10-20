"""
Graph Construction Layer -- DORG Builder
=========================================
Builds the **Dynamic Operational Railway Graph (DORG)** shown in the
architecture diagram, on top of the `RailwayState` produced each tick by
the Simulation Layer (`simulation_layer.simulation_layer.SimulationLayer`).

Node types
----------
    station   S:<code>            static, one per corridor station
    platform  P:<code>:<id>       static, one per physical platform
    loop      L:<code>:<id>       static, one per physical loop line
    train     T:<train_number>    dynamic, added while a train is running
                                   (NOT_STARTED / TERMINATED trains have no node)

Keeping station/platform/loop nodes **static** (built once, never removed)
matters for the downstream GraphSAGE encoder -- only train nodes and edges
churn tick to tick, so the infrastructure part of the node index never
shifts under the model.

Edge types
----------
    connects          station  -> station   static corridor adjacency
    has_platform      station  -> platform  static
    has_loop          station  -> loop      static
    at                train    -> station   train is halted at this station
    transiting        train    -> station   train is running toward this
                                             station (from its current section)
    waiting           train    -> station   train wants a platform/loop here
                                             but none is free right now
    occupies_platform train    -> platform  train currently holds this platform
    occupies_loop     train    -> loop      train currently holds this loop line
    conflict          train   <-> train     both directions; trains contending
                                             for the same resource this tick
                                             (attr `conflict_type`: platform |
                                             loop_line | track | junction | headway)
    precedes          train    -> train     directed, higher-priority train ->
                                             lower-priority train, for every
                                             conflicting pair (dispatch-precedence
                                             signal for the RL agent)

This module's only job is graph construction -- it does not do any
representation learning (that's GraphSAGE, the next layer downstream) and
does not know about PPO.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import networkx as nx

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_layer.data_layer import StationDB, PlatformInfo
from simulation_layer.simulation_layer import RailwayState


def sid(code: str) -> str:
    return f"S:{code}"


def pid(station: str, platform_id: int) -> str:
    return f"P:{station}:{platform_id}"


def lid(station: str, loop_id: int) -> str:
    return f"L:{station}:{loop_id}"


def tid(train_number: int) -> str:
    return f"T:{train_number}"


class DORGBuilder:
    """
    Builds/maintains one `networkx.MultiDiGraph` representing the Dynamic
    Operational Railway Graph. Static infrastructure (stations, platforms,
    loop lines, corridor adjacency) is built once in `__init__`. Call
    `build(railway_state)` every tick to refresh the dynamic (train-related)
    nodes and edges in place and get the current graph back.
    """

    STATIC_EDGE_TYPES = {"connects", "has_platform", "has_loop"}
    DYNAMIC_EDGE_TYPES = {
        "at", "transiting", "waiting",
        "occupies_platform", "occupies_loop",
        "conflict", "precedes",
    }

    def __init__(self, stations: StationDB, platform_info: PlatformInfo):
        self.stations = stations
        self.platform_info = platform_info
        self.graph = nx.MultiDiGraph()
        self._build_static_topology()

    # ------------------------------------------------------------------
    # Static infrastructure (built once)
    # ------------------------------------------------------------------
    def _build_static_topology(self) -> None:
        g = self.graph

        for code in self.stations.ordered_codes:
            station = self.stations.get(code)
            cap = self.platform_info.capacity.get(code)
            g.add_node(
                sid(code), node_type="station", code=code, name=station.name,
                distance_km=station.distance_km,
                platforms=cap.platforms if cap else station.platforms,
                loop_line_count=cap.loop_line_count if cap else 0,
                tracks=cap.tracks if cap else None,
            )

            n_platforms = cap.platforms if cap else station.platforms
            for i in range(1, n_platforms + 1):
                g.add_node(pid(code, i), node_type="platform", station=code, platform_id=i)
                g.add_edge(sid(code), pid(code, i), edge_type="has_platform")

            n_loops = cap.loop_line_count if cap else 0
            for i in range(1, n_loops + 1):
                g.add_node(lid(code, i), node_type="loop", station=code, loop_id=i)
                g.add_edge(sid(code), lid(code, i), edge_type="has_loop")

        # corridor adjacency, both directions, with the live distance between
        # each pair of adjacent stations
        codes = self.stations.ordered_codes
        for a, b in zip(codes[:-1], codes[1:]):
            dist = abs(self.stations.distance_km(b) - self.stations.distance_km(a))
            g.add_edge(sid(a), sid(b), edge_type="connects", distance_km=dist)
            g.add_edge(sid(b), sid(a), edge_type="connects", distance_km=dist)

    # ------------------------------------------------------------------
    # Dynamic refresh (called every tick)
    # ------------------------------------------------------------------
    def _clear_dynamic(self) -> None:
        g = self.graph
        # drop dynamic edges
        drop_edges = [
            (u, v, k) for u, v, k, d in g.edges(keys=True, data=True)
            if d.get("edge_type") in self.DYNAMIC_EDGE_TYPES
        ]
        g.remove_edges_from(drop_edges)
        # drop train nodes (rebuilt fresh each tick -- infra nodes untouched)
        drop_nodes = [n for n, d in g.nodes(data=True) if d.get("node_type") == "train"]
        g.remove_nodes_from(drop_nodes)

    def build(self, railway_state: Union[RailwayState, dict]) -> nx.MultiDiGraph:
        """Refresh the dynamic portion of the graph from a RailwayState
        (or its `.as_dict()` form) and return the current graph."""
        state = railway_state.as_dict() if hasattr(railway_state, "as_dict") else railway_state
        g = self.graph
        self._clear_dynamic()

        priority_by_train = {t["train_number"]: t["priority"] for t in state["trains"]}

        # -- train nodes + "at" / "transiting" edges
        for t in state["trains"]:
            tn = t["train_number"]
            g.add_node(
                tid(tn), node_type="train", train_number=tn,
                status=t["status"], delay_min=t["delay_min"],
                speed_kmph=t["speed_kmph"], position_km=t["position_km"],
                priority=t["priority"],
            )
            if t["status"] in ("HALTED",) and t["station"]:
                g.add_edge(tid(tn), sid(t["station"]), edge_type="at")
            elif t["status"] in ("WAITING", "HELD_LOOP") and t["station"]:
                # still physically "at" the station even though it hasn't
                # been granted a platform/loop yet
                g.add_edge(tid(tn), sid(t["station"]), edge_type="at")
            elif t["status"] == "IN_TRANSIT" and t["to_station"]:
                g.add_edge(
                    tid(tn), sid(t["to_station"]), edge_type="transiting",
                    from_station=t["from_station"],
                )

        # -- waiting edges (train wants a platform/loop at this station,
        #    none free yet)
        for tn, station in state["waiting_trains"]:
            if tid(tn) in g:
                g.add_edge(tid(tn), sid(station), edge_type="waiting")

        # -- occupies_platform / occupies_loop edges
        for station, table in state["platform_occupancy"].items():
            for platform_id, tn in table.items():
                if tid(tn) in g and pid(station, platform_id) in g:
                    g.add_edge(tid(tn), pid(station, platform_id), edge_type="occupies_platform")

        for station, table in state["loop_line_occupancy"].items():
            for loop_id, tn in table.items():
                if tid(tn) in g and lid(station, loop_id) in g:
                    g.add_edge(tid(tn), lid(station, loop_id), edge_type="occupies_loop")

        # -- conflict + precedes edges (all pairwise combinations per conflict)
        for c in state["conflicts"]:
            trains = [tn for tn in c["trains"] if tid(tn) in g]
            for i in range(len(trains)):
                for j in range(i + 1, len(trains)):
                    a, b = trains[i], trains[j]
                    g.add_edge(tid(a), tid(b), edge_type="conflict",
                               conflict_type=c["type"], location=c["location"])
                    g.add_edge(tid(b), tid(a), edge_type="conflict",
                               conflict_type=c["type"], location=c["location"])

                    pa, pb = priority_by_train.get(a), priority_by_train.get(b)
                    if pa is not None and pb is not None and pa != pb:
                        higher, lower = (a, b) if pa < pb else (b, a)
                        g.add_edge(tid(higher), tid(lower), edge_type="precedes",
                                   location=c["location"])

        return g

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def node_type_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _, d in self.graph.nodes(data=True):
            counts[d["node_type"]] = counts.get(d["node_type"], 0) + 1
        return counts

    def edge_type_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _, _, d in self.graph.edges(data=True):
            counts[d["edge_type"]] = counts.get(d["edge_type"], 0) + 1
        return counts

    def subgraph_for_train(self, train_number: int, hops: int = 1) -> nx.MultiDiGraph:
        """Ego-graph around one train node -- handy for debugging/plotting
        a single train's local neighborhood instead of the whole corridor."""
        node = tid(train_number)
        if node not in self.graph:
            return nx.MultiDiGraph()
        return nx.ego_graph(self.graph.to_undirected(as_view=False), node, radius=hops)

    def export_graphml(self, path: str) -> None:
        """Export the current snapshot to GraphML (Gephi / Cytoscape / etc.)."""
        g = nx.MultiDiGraph()
        for n, d in self.graph.nodes(data=True):
            g.add_node(n, **{k: ("" if v is None else v) for k, v in d.items()})
        for u, v, d in self.graph.edges(data=True):
            g.add_edge(u, v, **{k: ("" if v2 is None else v2) for k, v2 in d.items()})
        nx.write_graphml(g, path)
