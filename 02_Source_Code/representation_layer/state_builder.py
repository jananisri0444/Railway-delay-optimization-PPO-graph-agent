"""
State Builder
=============
Converts per-node embeddings from DORGEncoder into a flat 1-D numpy array
that the PPO agent receives as its observation at each tick.

State vector layout (total = STATE_DIM = 5 + hidden_dim + MAX_TRAINS*hidden_dim +
                                          MAX_WAITING*6 + MAX_CONFLICTS*5)

    [0:5]                  global scalars
    [5:5+H]                mean-pooled embedding over ALL train nodes
    [5+H : 5+H+W*H]        per-waiting-train embeddings (zero-padded to MAX_WAITING)
    [5+H+W*H : ...]        conflict descriptors (zero-padded to MAX_CONFLICTS)

Global scalars (5):
    sim_time_norm, n_active_norm, n_waiting_norm, n_conflicts_norm, total_delay_norm

Per-waiting-train (H each, up to MAX_WAITING = 5):
    The GraphSAGE embedding of that train node.

Conflict descriptor (5 each, up to MAX_CONFLICTS = 8):
    [conflict_type_oh(4), priority_diff_norm]
    conflict type: 0=platform, 1=loop, 2=track/junction, 3=headway

RepresentationLayer
-------------------
The convenience class that owns a HeteroConverter, DORGEncoder, and
StateBuilder and exposes a single ``encode(graph, railway_state_dict)``
call.  This is what the RL environment imports.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from .hetero_converter import HeteroConverter
from .graphsage_encoder import DORGEncoder

MAX_WAITING   = 5
MAX_CONFLICTS = 8
_MAX_TIME_MIN  = 1440.0   # 24 h in minutes
_MAX_TRAINS    = 60.0
_MAX_DELAY_MIN = 120.0

_CONFLICT_TYPE_MAP = {
    "platform": 0, "loop_line": 1, "loop": 1,
    "track": 2, "junction": 2, "headway": 3,
}


def _conflict_onehot(ctype: str) -> List[float]:
    oh = [0.0, 0.0, 0.0, 0.0]
    oh[_CONFLICT_TYPE_MAP.get(ctype, 2)] = 1.0
    return oh


class StateBuilder:
    """
    Stateless helper that assembles the flat state array from encoder output
    and the raw railway state dict.

    Parameters
    ----------
    hidden_dim : int
        Must match DORGEncoder.hidden_dim.
    """

    def __init__(self, hidden_dim: int = 64):
        self.hidden_dim = hidden_dim
        H = hidden_dim
        self.state_dim: int = (
            5                        # global scalars
            + H                      # mean train pool
            + MAX_WAITING * H        # per-waiting-train embeddings
            + MAX_WAITING * 6        # explicit operational features
            + MAX_CONFLICTS * 5      # conflict descriptors
        )

    # ------------------------------------------------------------------
    def build(
        self,
        embeddings: Dict[str, "torch.Tensor"],
        node_maps:  Dict[str, Dict[str, int]],
        railway_state_dict: Dict[str, Any],
    ) -> np.ndarray:
        """
        Parameters
        ----------
        embeddings : dict from DORGEncoder.forward()
        node_maps  : dict from HeteroConverter.convert()
        railway_state_dict : the tick's railway_state.as_dict()

        Returns
        -------
        np.ndarray of shape (state_dim,), dtype float32
        """
        H = self.hidden_dim
        state = np.zeros(self.state_dim, dtype=np.float32)

        train_embs: torch.Tensor = embeddings.get(
            "train", torch.zeros((0, H), dtype=torch.float)
        )
        train_map: Dict[str, int] = node_maps.get("train", {})

        # ── Global scalars ───────────────────────────────────────────────
        sim_time = railway_state_dict.get("simulation_time_min", 0.0)
        trains   = railway_state_dict.get("trains", [])
        waiting  = railway_state_dict.get("waiting_trains", [])
        conflicts = railway_state_dict.get("conflicts", [])
        total_delay = sum(t.get("delay_min", 0.0) for t in trains)
        n_active = sum(
            1 for t in trains
            if t.get("status") not in ("NOT_STARTED", "TERMINATED")
        )

        state[0] = (sim_time % _MAX_TIME_MIN) / _MAX_TIME_MIN
        state[1] = n_active / _MAX_TRAINS
        state[2] = len(waiting) / MAX_WAITING
        state[3] = len(conflicts) / MAX_CONFLICTS
        state[4] = min(total_delay, _MAX_DELAY_MIN * _MAX_TRAINS) / (_MAX_DELAY_MIN * _MAX_TRAINS)

        ptr = 5

        # ── Mean-pooled train embedding ───────────────────────────────────
        if train_embs.size(0) > 0:
            mean_emb = train_embs.mean(dim=0).detach().cpu().numpy()
        else:
            mean_emb = np.zeros(H, dtype=np.float32)
        state[ptr: ptr + H] = mean_emb
        ptr += H

        # ── Per-waiting-train embeddings (up to MAX_WAITING) ─────────────
        # waiting_trains: list of (train_number, station_code)
        for slot, (tn, _) in enumerate(waiting[:MAX_WAITING]):
            node_id = f"T:{tn}"
            if node_id in train_map and train_embs.size(0) > 0:
                idx = train_map[node_id]
                state[ptr: ptr + H] = train_embs[idx].detach().cpu().numpy()
            ptr += H

        # pad remaining slots
        ptr += max(0, MAX_WAITING - len(waiting)) * H

        # ── Explicit operational features per waiting train ───────────────
        # These are deliberately simple and directly tied to dispatching.
        # The GNN can provide topology context, while PPO no longer has to
        # infer delay/priority/resource pressure from a random embedding.
        train_by_number = {t["train_number"]: t for t in trains}
        platform_occ = railway_state_dict.get("platform_occupancy", {})
        loop_occ = railway_state_dict.get("loop_line_occupancy", {})
        for tn, station in waiting[:MAX_WAITING]:
            t = train_by_number.get(tn, {})
            status = t.get("status", "WAITING")
            features = np.array([
                min(float(t.get("delay_min", 0.0)), _MAX_DELAY_MIN) / _MAX_DELAY_MIN,
                min(float(t.get("priority", 3)), 6.0) / 6.0,
                1.0 if status == "WAITING" else 0.0,
                1.0 if status == "HELD_LOOP" else 0.0,
                min(len(platform_occ.get(station, {})), 10) / 10.0,
                min(len(loop_occ.get(station, {})), 5) / 5.0,
            ], dtype=np.float32)
            state[ptr: ptr + 6] = features
            ptr += 6
        ptr += max(0, MAX_WAITING - len(waiting)) * 6

        # ── Conflict descriptors ──────────────────────────────────────────
        priority_by_train = {t["train_number"]: t.get("priority", 3) for t in trains}
        for slot, c in enumerate(conflicts[:MAX_CONFLICTS]):
            oh = _conflict_onehot(c.get("type", "track"))
            c_trains = c.get("trains", [])
            pvals = [priority_by_train.get(t, 3) for t in c_trains]
            pdiff = (max(pvals) - min(pvals)) / 6.0 if len(pvals) >= 2 else 0.0
            state[ptr: ptr + 4] = oh
            state[ptr + 4]      = pdiff
            ptr += 5

        # pad remaining conflict slots
        ptr += max(0, MAX_CONFLICTS - len(conflicts)) * 5

        return state


# ===========================================================================
# Convenience wrapper
# ===========================================================================
class RepresentationLayer:
    """
    Owns the full representation pipeline and exposes a single
    ``encode()`` call for the RL environment.

    Usage
    -----
    ::

        rep_layer = RepresentationLayer(hidden_dim=64)
        state_vec = rep_layer.encode(graph, railway_state_dict)
        # state_vec.shape == (rep_layer.state_dim,)

    Parameters
    ----------
    hidden_dim : int
    num_layers : int
    dropout : float
    device : str | torch.device
    """

    def __init__(
        self,
        hidden_dim:  int   = 64,
        num_layers:  int   = 2,
        dropout:     float = 0.1,
        device: str | torch.device = "cpu",
    ):
        self.device   = torch.device(device)
        self.converter = HeteroConverter()
        # Fixed seed (RNG forked so global state is untouched): the encoder is a
        # frozen feature extractor, so every env/process must get IDENTICAL
        # weights, otherwise trained PPO checkpoints do not match the dashboard.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)
            self.encoder   = DORGEncoder(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            ).to(self.device)
        self.encoder.eval()   # inference mode by default; call .train() during PPO updates
        self.state_builder = StateBuilder(hidden_dim=hidden_dim)
        self.state_dim     = self.state_builder.state_dim

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(
        self,
        graph: nx.MultiDiGraph,
        railway_state_dict: Dict[str, Any],
    ) -> np.ndarray:
        """
        Parameters
        ----------
        graph : nx.MultiDiGraph
            Output of DORGBuilder.build() for this tick.
        railway_state_dict : dict
            Output of railway_state.as_dict() for this tick.

        Returns
        -------
        np.ndarray of shape (self.state_dim,), dtype float32
        """
        pyg_data, node_maps = self.converter.convert(graph)
        pyg_data = pyg_data.to(self.device)
        embeddings = self.encoder(pyg_data)
        state = self.state_builder.build(embeddings, node_maps, railway_state_dict)
        return state

    # ------------------------------------------------------------------
    def parameters(self):
        """Expose encoder parameters for the PPO optimiser."""
        return self.encoder.parameters()

    def named_parameters(self):
        return self.encoder.named_parameters()

    def state_dict(self):
        return self.encoder.state_dict()

    def load_state_dict(self, sd):
        self.encoder.load_state_dict(sd)

    def train(self, mode: bool = True):
        self.encoder.train(mode)
        return self

    def eval(self):
        self.encoder.eval()
        return self

    def encode_with_grad(
        self,
        graph: nx.MultiDiGraph,
        railway_state_dict: Dict[str, Any],
    ) -> Tuple["torch.Tensor", Dict[str, Any]]:
        """
        Same as encode() but returns a differentiable Tensor (for the
        PPO actor-critic forward pass during training) plus the node_maps.
        """
        pyg_data, node_maps = self.converter.convert(graph)
        pyg_data = pyg_data.to(self.device)
        embeddings = self.encoder(pyg_data)

        H = self.encoder.hidden_dim
        train_embs: torch.Tensor = embeddings.get(
            "train", torch.zeros((0, H), device=self.device)
        )

        # Mean-pool train embeddings (differentiable)
        if train_embs.size(0) > 0:
            mean_emb = train_embs.mean(dim=0)
        else:
            mean_emb = torch.zeros(H, device=self.device)

        # Build full state as Tensor (parallel to StateBuilder.build but differentiable)
        state_np = self.state_builder.build(embeddings, node_maps, railway_state_dict)
        # Replace mean-pool slice with differentiable version
        state_t  = torch.tensor(state_np, dtype=torch.float, device=self.device)
        state_t[5: 5 + H] = mean_emb   # differentiable slot

        return state_t, node_maps
