from __future__ import annotations

from typing import Optional, Sequence, Tuple, Any
import torch
import torch.nn as nn
from torch.distributions import Categorical
from .environment import MAX_DECISIONS


class GraphPPOActorCritic(nn.Module):
    """PPO actor-critic; Graph mode recomputes trainable GraphSAGE embeddings per state."""
    def __init__(self, state_dim: int, hidden_dim: int = 256, use_graph: bool = True):
        super().__init__()
        self.state_dim = int(state_dim)
        self.use_graph = bool(use_graph)
        self.graph_hidden_dim = 64
        if self.use_graph:
            from representation_layer.graphsage_encoder import DORGEncoder
            from representation_layer.hetero_converter import HeteroConverter
            self.graph_encoder = DORGEncoder(hidden_dim=self.graph_hidden_dim, num_layers=2, dropout=0.0)
            self.graph_converter = HeteroConverter()
        self.norm = nn.LayerNorm(state_dim)
        def trunk():
            return nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.actor_body = trunk()
        self.critic_body = trunk()
        self.actor = nn.Linear(hidden_dim, MAX_DECISIONS * 3)
        self.critic = nn.Linear(hidden_dim, 1)
        for m in list(self.actor_body) + list(self.critic_body):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def _graph_state(self, state: torch.Tensor, contexts: Optional[Sequence[Any]]) -> torch.Tensor:
        """Replace frozen graph embedding slices with differentiable GraphSAGE output."""
        if not self.use_graph or contexts is None:
            return state
        if len(contexts) != state.shape[0]:
            raise ValueError(f'Expected {state.shape[0]} graph contexts, got {len(contexts)}')
        H = self.graph_hidden_dim
        result = []
        for row, context in zip(state, contexts):
            if context is None:
                result.append(row)
                continue
            graph, railway_state = context
            data, node_maps = self.graph_converter.convert(graph)
            data = data.to(row.device)
            embeddings = self.graph_encoder(data)
            train_embs = embeddings.get('train', torch.zeros((0, H), device=row.device))
            x = row.clone()
            # StateBuilder layout: globals[0:5], mean train pool[5:69], then
            # five waiting-train embeddings[69:389]. These assignments retain autograd.
            if train_embs.size(0):
                x[5:5 + H] = train_embs.mean(dim=0)
            waiting = railway_state.get('waiting_trains', [])[:5]
            train_map = node_maps.get('train', {})
            for slot in range(5):
                lo = 5 + H + slot * H
                hi = lo + H
                if slot < len(waiting):
                    tn = waiting[slot][0]
                    idx = train_map.get(f'T:{tn}')
                    if idx is not None and idx < train_embs.size(0):
                        x[lo:hi] = train_embs[idx]
                    else:
                        x[lo:hi] = 0.0
                else:
                    x[lo:hi] = 0.0
            result.append(x)
        return torch.stack(result, dim=0)

    def forward(self, state, action_masks=None, graph_contexts=None):
        state = self._graph_state(state, graph_contexts)
        x = self.norm(state)
        logits = self.actor(self.actor_body(x)).view(-1, MAX_DECISIONS, 3)
        if action_masks is not None:
            logits = logits.masked_fill(~action_masks.bool(), -1e9)
        dists = [Categorical(logits=logits[:, i, :]) for i in range(MAX_DECISIONS)]
        value = self.critic(self.critic_body(x)).squeeze(-1)
        return dists, value

    def act(self, state, mask, deterministic=False, slot_valid=None, graph_contexts=None):
        dists, value = self.forward(state, mask, graph_contexts=graph_contexts)
        acts, lps = [], []
        for d in dists:
            a = d.probs.argmax(-1) if deterministic else d.sample()
            acts.append(a)
            lps.append(d.log_prob(a))
        acts = torch.stack(acts, 1)
        lps = torch.stack(lps, 1)
        if slot_valid is not None:
            lps = lps * slot_valid.float()
        return acts, lps.sum(1), value

    def evaluate(self, state, mask, actions, slot_valid, graph_contexts=None):
        dists, value = self.forward(state, mask, graph_contexts=graph_contexts)
        sv = slot_valid.float()
        lp = torch.stack([dists[i].log_prob(actions[:, i]) for i in range(MAX_DECISIONS)], 1) * sv
        ent = torch.stack([dists[i].entropy() for i in range(MAX_DECISIONS)], 1) * sv
        return lp.sum(1), ent.sum(1) / sv.sum(1).clamp(min=1), value
