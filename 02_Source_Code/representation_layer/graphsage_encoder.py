"""
DORG GraphSAGE Encoder
=======================
A heterogeneous GraphSAGE encoder that reads a HeteroData produced by
HeteroConverter and outputs per-node embeddings for every node type.

Architecture
------------
    Input node features (per type)
        │
        ▼
    Linear input projection  →  hidden_dim
        │
        ▼
    HeteroConv(SAGEConv) × num_layers      (message-passing on all relations)
        │
        ▼
    Per-node embeddings  (dict[node_type → Tensor[N, hidden_dim]])

The SAGEConv layers aggregate neighbour messages into each node's embedding,
propagating information across the heterogeneous graph edges.  Using
`aggr="mean"` per the original GraphSAGE paper; switch to "max" or "lstm"
if you want to experiment.

Node feature dimensions (must match HeteroConverter output):
    station  : 5
    platform : 2
    loop     : 2
    train    : 11
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


# Input feature dimensions per node type
NODE_IN_DIM: Dict[str, int] = {
    "station":  5,
    "platform": 2,
    "loop":     2,
    "train":    11,
}


class DORGEncoder(nn.Module):
    """
    Heterogeneous GraphSAGE encoder for the DORG.

    Parameters
    ----------
    hidden_dim : int
        Output embedding dimension for every node type (default 64).
    num_layers : int
        Number of message-passing rounds (default 2).
    dropout : float
        Dropout probability applied between layers (default 0.1).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout    = dropout

        # ── Input projections: raw features → hidden_dim ──────────────────
        # nn.ModuleDict reserves "train" as a method name, so we prefix
        # all keys with "nt_" (node-type) to avoid the collision.
        self.input_proj = nn.ModuleDict({
            f"nt_{nt}": nn.Linear(in_dim, hidden_dim)
            for nt, in_dim in NODE_IN_DIM.items()
        })

        # ── Message-passing layers ─────────────────────────────────────────
        # Each HeteroConv layer covers all 10 relations.
        # For relations whose src/dst share the same type (e.g. train→train),
        # SAGEConv still gets (in_channels, out_channels) = (hidden_dim, hidden_dim).
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                ("station", "connects",          "station"):  SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
                ("station", "has_platform",      "platform"): SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
                ("station", "has_loop",          "loop"):     SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
                ("train",   "at",                "station"):  SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
                ("train",   "transiting",        "station"):  SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
                ("train",   "waiting",           "station"):  SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
                ("train",   "occupies_platform", "platform"): SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
                ("train",   "occupies_loop",     "loop"):     SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
                ("train",   "conflict",          "train"):    SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
                ("train",   "precedes",          "train"):    SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # ── Layer norms (stabilise training) ──────────────────────────────
        self.norms = nn.ModuleList([
            nn.ModuleDict({f"nt_{nt}": nn.LayerNorm(hidden_dim) for nt in NODE_IN_DIM})
            for _ in range(num_layers)
        ])

    # ------------------------------------------------------------------
    def forward(self, data: HeteroData) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        data : HeteroData
            Output of HeteroConverter.convert().  Must contain .x tensors
            for every node type and .edge_index tensors for every relation
            (even empty ones).

        Returns
        -------
        embeddings : dict[str, Tensor]
            { node_type : Tensor[N_type, hidden_dim] }
        """
        # ── Project inputs ────────────────────────────────────────────────
        # self.input_proj keys are "nt_{node_type}"; x_dict uses bare node_type
        x_dict: Dict[str, Tensor] = {}
        for prefixed_key, proj in self.input_proj.items():
            nt = prefixed_key[3:]   # strip "nt_"
            feat = data[nt].x
            if feat.size(0) == 0:
                x_dict[nt] = torch.zeros((0, self.hidden_dim), dtype=torch.float,
                                          device=feat.device)
            else:
                x_dict[nt] = F.relu(proj(feat))

        # ── Build edge_index_dict ─────────────────────────────────────────
        edge_index_dict = {}
        for store in data.edge_stores:
            if store.edge_index is not None and store.edge_index.numel() > 0:
                edge_index_dict[store._key] = store.edge_index

        # ── Message passing ───────────────────────────────────────────────
        for layer_idx, conv in enumerate(self.convs):
            # filter to relations that have edges *and* non-empty src/dst
            active_edict = {
                k: ei for k, ei in edge_index_dict.items()
                if x_dict.get(k[0], torch.empty(0)).size(0) > 0
                and x_dict.get(k[2], torch.empty(0)).size(0) > 0
                and ei.size(1) > 0
            }
            if not active_edict:
                break

            out_dict = conv(x_dict, active_edict)

            norm_layer = self.norms[layer_idx]
            new_x: Dict[str, Tensor] = {}
            for nt in x_dict:
                if nt in out_dict and out_dict[nt].size(0) > 0:
                    new_x[nt] = F.dropout(
                        F.relu(norm_layer[f"nt_{nt}"](out_dict[nt])),
                        p=self.dropout, training=self.training,
                    )
                else:
                    new_x[nt] = x_dict[nt]  # no messages arrived → carry over
            x_dict = new_x

        return x_dict

    # ------------------------------------------------------------------
    @property
    def out_dim(self) -> int:
        return self.hidden_dim
