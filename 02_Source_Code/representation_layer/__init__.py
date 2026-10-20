"""
Representation Layer
====================
Converts a tick's DORG (nx.MultiDiGraph from DORGBuilder) into a flat
state vector ready for the PPO agent.

Pipeline:
    DORG (nx.MultiDiGraph)
        │
        ▼
    HeteroConverter.convert()          →  torch_geometric.data.HeteroData
        │
        ▼
    DORGEncoder.forward()              →  per-node embeddings  (GraphSAGE)
        │
        ▼
    StateBuilder.build_state()         →  1-D numpy array  (state for PPO)
"""
from .hetero_converter import HeteroConverter
from .graphsage_encoder import DORGEncoder
from .state_builder import StateBuilder, RepresentationLayer
