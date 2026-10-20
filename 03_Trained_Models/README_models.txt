WHICH FILE IS WHICH MODEL
-------------------------
policy_final.pt                  GRAPH PPO (main model). State = 509 numbers (DORG + GraphSAGE + slot + graph features). Used by every script/dashboard as "Graph PPO".
policy_best.pt                   Graph PPO, best checkpoint found during training (alternative to policy_final.pt).
policy_flat.pt                   FLAT PPO, no graph. State = 105 plain numbers. Used as "Flat PPO".
policy_flat_same_teacher.pt      Control model: flat PPO that copied the same teacher rule as the graph model (used only in evaluate_models.py).
policy_flat_same_teacher_best.pt Best checkpoint of the control model.

All five were trained for 300 episodes in stress mode after a 60-episode warm start (copying a rule). Training CODE: 02_Source_Code/decision_layer/train_ppo.py
If you retrain, COPY this folder first: training overwrites policy_final.pt by default.
