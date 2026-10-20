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

## 5. Honest summary of the results (say this in the viva)
* Synthetic stress test, 21 scenarios, average reward (closer to 0 is better): Graph PPO −135.3, Flat PPO −139.6, priority rule −139.6, always proceed −145.6, always hold −164.4. Graph PPO has the best average, but the gap is **not statistically significant**.
* On real days the models face very few decisions, so Graph PPO and Flat PPO behave almost identically. Real delays contain causes the simulator does not model (signals, weather, crew), so the comparison with reality is context, **not proof of beating real dispatchers**.
* Delay is only added by dispatching in the simulator, so it does not "recover" along the route the way real delay does.

## 6. IMPORTANT data problem found
`historical_reported_observations.csv` (and `historical_all_rows.csv`) had exactly **1,048,575 rows** = Excel's row limit. The file was cut off: only **79 of 125 trains** are in it (e.g. train 22652 has 10,577 of 32,690 rows; all 40xxx, 43xxx and 66xxx trains are missing — mostly low-priority passenger/MEMU trains). All real-data results here use that incomplete data.
**Fix:** run `01_Data/2_Real_Railway_History/collect_railway_history.py` again (needs internet) and never open or save the CSV in Excel, then `check_history_completeness.py`, then `1_build_real_delay_data.py`, then `3_dashboard_and_table.py`. The two huge raw CSVs are not inside this zip (about 360 MB, and truncated); `1_build_real_delay_data.py` reads the file from `01_Data/2_Real_Railway_History/`.
If you see `PermissionError` when running a script, close the CSV in Excel (or any program that has it open).

## 7. New in this version
**Train focus (live simulator dashboard).** Run `04_Run_Scripts/A_Live_Simulator_Dashboard/run_live_dashboard.py`, open http://127.0.0.1:5000, press Stress test and Play/Step. Then:
* click any train in the decision table or the active-trains table (or use the "Train focus" drop-down);
* you get **every decision that involved that train** (time, action, probabilities, why: who it blocks and what blocking vs holding costs), a **station-by-station journey** (scheduled vs simulated time, delay, "waited in loop here"), and a **loop & overtake animation** at the loop station (the waiting train goes into the loop line, the more important train passes on the main line, then it leaves).
* Good demo scenarios where the Graph PPO really chooses loop & overtake: **Monday + seed 5** and **Friday + seed 0** (Stress test, 08:00–22:00); the trains with loop events appear first in the drop-down.
* Forced decisions (only one possible action) are hidden by default; untick the box to see them.
* The loop is modelled as: a one-off 1.5-minute cost for the waiting train and the conflict cleared for 10 minutes. The animation shows what that means at the station; it is not track-level signalling.

**Why Graph PPO and Flat PPO match** — measured in `06_Viva_Material/Graph_vs_Flat_Diagnosis.md` with two scripts in `04_Run_Scripts/C_Evaluate_Models_Synthetic/`: `diagnose_graph_vs_flat.py` and `tune_rules_graph_headroom.py` (results: `05_Results/A_Synthetic_Testing/diagnosis_graph_vs_flat.csv`, `rule_sweep.json`).

**Decision follower (no train to choose).** With the live dashboard running, open http://127.0.0.1:5000/follow (or click "Decision follower" at the top of the overall dashboard). Press Play: the page chooses the train for you.
* **★ Train X = the train being decided by the PPO agent**, **⚠ Train Y = the trains that are blocked and wait for the outcome**, with the action, probabilities and the reason in plain words.
* Loop & overtake decisions play an animation automatically (the waiting train goes into the loop line, the others pass, it leaves). Times for trains that have not passed yet are marked "expected".
* A journey table (scheduled vs simulated time at each station) follows the train being decided, and a feed lists every decision (click one to replay it).
* **Speed slider**: slow = about 7 seconds per decision. "show forced decisions" shows moves where only one action was possible.
* In the overall dashboard's "Train focus" drop-down, trains being decided now are marked **★ DECIDING NOW** and blocked trains **⚠ blocked now**, and are listed first.
* **"Compare Graph vs Flat on this scenario"** button: runs both models from scratch on the same day/seed and shows the real totals and the decisions where they differ. Nothing is adjusted. Example (Friday, seed 0, Stress test, 08:00–22:00): Flat PPO added 37.7 min of delay and Graph PPO 58.2 min — here the flat model was better; on the 21-scenario average the graph model was slightly better (−135.3 vs −139.6, not significant). Present both honestly.
* Two models give identical numbers (even to the second) only when they make the same choices: the simulator is deterministic, so same scenario + same actions = same result.

## Updated Graph PPO implementation (2026-10)

See `GRAPH_PPO_UPDATED_IMPLEMENTATION.md` for the end-to-end trainable GraphSAGE update, reward correction, gradient smoke test, and fair Graph-vs-Flat retraining commands. **Retrain both policies from fresh initialization** before interpreting any performance comparison; the old checkpoints do not contain the new trainable encoder.
