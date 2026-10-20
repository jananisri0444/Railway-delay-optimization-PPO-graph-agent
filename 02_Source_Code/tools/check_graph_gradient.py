"""Smoke test: verify PPO gradients reach GraphSAGE parameters.

Run from 02_Source_Code:
    python tools/check_graph_gradient.py
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_layer.environment import RailwayPPOEnv
from decision_layer.ppo_agent import GraphPPOActorCritic


def main():
    env = RailwayPPOEnv(480, 720, stress_mode=True, obs_mode='graph')
    obs, _ = env.reset(seed=123)
    for _ in range(100):
        obs, done = env.advance_to_decision()
        if done:
            break
        if env._candidates:
            break
    if not env._candidates:
        raise RuntimeError('No decision state found in the first 100 ticks; try another seed/window.')
    agent = GraphPPOActorCritic(env.state_dim, use_graph=True)
    state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    mask = torch.as_tensor(env.action_mask()).unsqueeze(0)
    valid = torch.as_tensor(env.slot_valid()).unsqueeze(0)
    _, log_prob, value = agent.act(state, mask, slot_valid=valid, graph_contexts=[env.graph_context()])
    loss = -log_prob.mean() + value.square().mean()
    loss.backward()
    grad_sum = sum(float(p.grad.abs().sum()) for p in agent.graph_encoder.parameters() if p.grad is not None)
    print(f'GraphSAGE gradient absolute sum: {grad_sum:.6f}')
    if grad_sum <= 0:
        raise AssertionError('GraphSAGE received no gradient; end-to-end training is not working.')
    print('PASS: gradients flow from PPO loss into GraphSAGE.')


if __name__ == '__main__':
    main()
