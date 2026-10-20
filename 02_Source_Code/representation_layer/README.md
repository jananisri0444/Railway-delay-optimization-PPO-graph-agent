# Representation Layer — GraphSAGE Encoder

## Role in the architecture

```
Graph Construction Layer
        │  nx.MultiDiGraph (DORG snapshot per tick)
        ▼
┌─────────────────────────────────────────────────┐
│            Representation Layer                 │
│                                                 │
│  HeteroConverter  ──►  PyG HeteroData           │
│       │                                         │
│  DORGEncoder       ──►  per-node embeddings     │
│  (GraphSAGE)             dict[type → Tensor]    │
│       │                                         │
│  StateBuilder      ──►  flat state vector       │
│                          np.ndarray (state_dim) │
└─────────────────────────────────────────────────┘
        │
        ▼
Decision Layer (PPO agent)
```

## Files

| File | Purpose |
|---|---|
| `hetero_converter.py` | Converts a DORG `nx.MultiDiGraph` → PyG `HeteroData` with node features and edge indices for all 10 relation types |
| `graphsage_encoder.py` | 2-layer heterogeneous GraphSAGE (`HeteroConv + SAGEConv`) that produces a 64-dim embedding per node |
| `state_builder.py` | Pools embeddings into a fixed-size 1-D state vector; also contains `RepresentationLayer` convenience wrapper |
| `demo_representation.py` | Quick sanity check — prints state vector stats for 10 ticks |

## Node feature vectors

| Type | Dim | Features |
|---|---|---|
| station | 5 | distance\_km\_norm, platforms\_norm, loop\_line\_count\_norm, is\_junction, is\_terminus |
| platform | 2 | platform\_id\_norm, is\_occupied |
| loop | 2 | loop\_id\_norm, is\_occupied |
| train | 11 | 6×status\_onehot, delay\_min\_norm, speed\_kmph\_norm, position\_km\_norm, priority\_norm, is\_conflicting |

## State vector layout

```
[0:5]              global scalars (sim_time, n_active, n_waiting, n_conflicts, total_delay)
[5:5+H]            mean-pooled embedding over all train nodes
[5+H : 5+H+W*H]   per-waiting-train embeddings  (W = MAX_WAITING = 5, zero-padded)
[rest]             conflict descriptors           (C = MAX_CONFLICTS = 8, zero-padded)
                   each descriptor: 4×type_onehot + priority_diff_norm
```

Total `state_dim` with H=64: **5 + 64 + 5×64 + 8×5 = 429**

## Quick usage

```python
from representation_layer import RepresentationLayer

rep = RepresentationLayer(hidden_dim=64, num_layers=2)
# graph  = DORGBuilder.build(railway_state)
# rs_dict = railway_state.as_dict()
state = rep.encode(graph, rs_dict)   # np.ndarray, shape (429,)
```

## Demo

```bash
cd railway_scheduler
python representation_layer/demo_representation.py
```
