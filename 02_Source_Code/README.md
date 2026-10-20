# 02_Source_Code — one folder per layer of the system
data_layer/                 1 Data: timetable, stations, platforms, loops, priorities, disruptions, weekday resolution
simulation_layer/           2 Simulation: minute-by-minute railway simulator and delay/cost rules
graph_construction_layer/   3 Graph: builds the DORG each tick
representation_layer/       4 Representation: HeteroData -> GraphSAGE -> state vector (graph) / flat_state.py (no graph)
decision_layer/             5 Decision: environment (actions, mask, reward), PPO network, training, rule baselines
tools/                      helpers used by the run scripts (evaluation, real-day replay, timeline, dashboard page)
project_paths.py            where every folder is (all scripts import it)
Run scripts live in ../04_Run_Scripts. Full table of files and jobs: ../START_HERE.md section 2.
