# 04_Run_Scripts — every script you run (they work from any folder)
A_Live_Simulator_Dashboard/run_live_dashboard.py     live simulator dashboard at http://127.0.0.1:5000 (per-train 'Train focus' view + http://127.0.0.1:5000/follow auto-following decision page)
B_Train_Models/train_model.py                        train graph or flat PPO
C_Evaluate_Models_Synthetic/evaluate_models.py       compare all policies on synthetic stress scenarios
C_Evaluate_Models_Synthetic/plot_results.py          charts from that comparison
C_Evaluate_Models_Synthetic/diagnose_graph_vs_flat.py   why do Graph PPO and Flat PPO behave alike? (same-situation comparison)
C_Evaluate_Models_Synthetic/tune_rules_graph_headroom.py   can graph information help at all? (rule search + held-out test)
D_Real_Data_Pipeline/1_build_real_delay_data.py      history CSV -> real delay data
D_Real_Data_Pipeline/2_station_timeline.py           real station-by-station timeline (day / range / month)
D_Real_Data_Pipeline/3_dashboard_and_table.py        REAL vs Graph PPO vs Flat PPO: dashboard + table
D_Real_Data_Pipeline/3a_dashboard_only.py            same, dashboard only
D_Real_Data_Pipeline/3b_table_only.py                same, table only
D_Real_Data_Pipeline/4_compare_policies_on_real_days.py + 4b_plot_compare_charts.py   all policies vs real
D_Real_Data_Pipeline/5_evaluate_on_real_day_scenarios.py   score models on real-delay scenarios (held-out days)
D_Real_Data_Pipeline/6_train_on_real_delays.py       optional fine-tuning on real delays
E_Smoke_Test/run_smoke_test.py                       10-second self-test
Each file starts with a comment explaining its options (open it or add --help).
