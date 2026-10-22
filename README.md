# Railway Dispatch with DORG + GraphSAGE + PPO — project guide

AI dispatcher for the Chennai–Bangalore corridor: a minute-by-minute simulator → a dynamic railway graph (DORG) → GraphSAGE → a PPO agent that
chooses HOLD / PROCEED / GRANT PLATFORM / LOOP & OVERTAKE. This folder also replays REAL historical delays and compares the models with what really happened.

## 1. Folder map (open them in this order when presenting)

| Folder | What is inside | Show it when you explain… |
|---|---|---|
| `01_Data/1_Simulator_Timetable` | the 4 CSV files that feed the simulator (127-train timetable, 84 stations, platforms, loop lines) | "what data the simulator uses" |
| `01_Data/2_Real_Railway_History` | collector script + train list + coverage summary for the REAL arrival/departure history | "where the real data came from" |
| `01_Data/3_Real_Delay_Data_Processed` | the real data cleaned into delays (`real_delays_clean.csv`), per-train / per-station summaries, one real-day timeline | "how real delays were prepared" |
| `02_Source_Code` | all the model code, one folder per layer (see section 2) | "how the system is built" |
| `03_Trained_Models` | the trained Graph PPO and Flat PPO files (`README_models.txt` says which is which) | "the models" |
| `04_Run_Scripts` | every script you run, grouped A–E and numbered (see section 3) | "live demo" |
| `05_Results` | all outputs: charts, tables, the REAL-vs-models dashboards and tables | "results" |
| `06_Viva_Material` | viva report, slides, neural-network diagram | revision before the viva |

## 2. Which code file belongs to which layer (`02_Source_Code`)

| Layer | Folder | Main file(s) | Job |
|---|---|---|---|
| 1 Data | `data_layer/` | `data_layer.py` | loads timetable/stations/platforms/loops/priorities, builds disruptions, and `TimetableDB.resolve()` picks the trains that run on a given weekday |
| 2 Simulation | `simulation_layer/` | `simulation_layer.py` | moves trains each minute, manages platforms/loops, detects conflicts, applies the delay costs |
| 3 Graph | `graph_construction_layer/` | `dorg_builder.py` | builds the DORG (stations, platforms, loops, trains, conflict/precedes edges) every tick |
| 4 Representation | `representation_layer/` | `hetero_converter.py`, `graphsage_encoder.py`, `state_builder.py`, `flat_state.py` | graph → numbers: HeteroData → GraphSAGE (64-d per node) → 459-number state; `flat_state.py` is the NO-graph state (105 numbers) |
| 5 Decision | `decision_layer/` | `environment.py`, `ppo_agent.py`, `train_ppo.py`, `baselines.py` | the Gym environment (decisions, action mask, reward, +50 extra features → 509), the actor-critic network, PPO training, and the rule baselines |
| Helpers | `tools/` | `policy_eval.py`, `real_delay_env.py`, `real_timeline.py`, `real_dashboard_template.html` | evaluation code, the real-day replay environment, timeline maths, dashboard page |
| Paths | `project_paths.py` | — | one place that knows every folder location |

## 3. How to run things

Install once (Python 3.10/3.11): `pip install -r requirements.txt`. Then open a terminal in the folder shown, or give the full path to the script. All outputs go to `05_Results`.

**A. Live simulator dashboard (the interactive demo)**
`python 04_Run_Scripts/A_Live_Simulator_Dashboard/run_live_dashboard.py` → open http://127.0.0.1:5000 → pick Model (Graph/Flat), Day, press Stress test, then Step or Play.

**D. REAL data → dashboard and table (the main new part)** — folder `04_Run_Scripts/D_Real_Data_Pipeline`
| Command | What it does |
|---|---|
| `python 3_dashboard_and_table.py --date 2025-03-05` | **dashboard AND table in one run** (one replay) → `05_Results/B_Real_Data_Replay/dashboard_and_table/<dates>_full/` |
| `python 3a_dashboard_only.py --date 2025-03-05` | **dashboard alone** (`real_dashboard.html`) |
| `python 3b_table_only.py --date 2025-03-05` | **table alone** (`comparison_table.html` + `.csv`) |
| add `--from 2025-03-03 --to 2025-03-09 --max-days 7` | a whole week instead of one day |
| add `--mode observed` | simulate only the trains that really ran (quiet days: the models may have nothing to decide) |

You do NOT have to run them to present: the finished dashboards and tables are already in `05_Results/B_Real_Data_Replay/dashboard_and_table/` — just double-click the `.html` files.

Other steps of the pipeline (only needed if you change the data): `1_build_real_delay_data.py` (clean the history CSV), `2_station_timeline.py` (real station-by-station timeline), `4_compare_policies_on_real_days.py` + `4b_plot_compare_charts.py` (all policies vs real), `5_evaluate_on_real_day_scenarios.py`, `6_train_on_real_delays.py` (optional fine-tuning).

**C. Compare models on synthetic stress scenarios:** `python 04_Run_Scripts/C_Evaluate_Models_Synthetic/evaluate_models.py --seeds 21 --steps 200`, then `plot_results.py`.
**B. Train a model:** `python 04_Run_Scripts/B_Train_Models/train_model.py --help` (copy `03_Trained_Models` first: training overwrites `policy_final.pt`).
**E. Quick self-test:** `python 04_Run_Scripts/E_Smoke_Test/run_smoke_test.py`.

## 4. What is in `05_Results`
* `A_Synthetic_Testing/` — the 21-scenario comparison of all policies (`evaluation.csv`, `plots/1_comparison.png`, `plots/2_per_seed.png`, training curves).
* `B_Real_Data_Replay/compare_policies/` — all policies vs real, 7 days (2025-03-03 … 03-09), only trains that really ran.
* `B_Real_Data_Replay/dashboard_and_table/` — REAL vs Graph PPO vs Flat PPO dashboards and tables (week 2025-03-03…09 and the day 2026-05-11).

## 5. summary of the results
* On real days the models face very few decisions, so Graph PPO and Flat PPO behave almost identically. Real delays contain causes the simulator does not model (signals, crew), so the comparison with reality is context, **not proof of beating real dispatchers**.
* Delay is only added by dispatching in the simulator, so it does not "recover" along the route the way real delay does.
