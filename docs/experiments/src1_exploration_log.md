# src1 Exploration Log

This log records experiments run in the isolated `src1/` workspace. The current SOTA remains the four-model ensemble in `src/` and `docs/sota/`.

## Protocol

- `src/` is not modified by exploratory experiments.
- Phase1 validation uses train1 for training, test1 green windows for features, and train2 labels only for final scoring.
- Repeated phase1 sweeps are exploratory. A candidate should not be promoted to SOTA unless it also passes an internal no-leak selection protocol.

## Current Baseline To Beat

| Route | Phase1 MAPE | Status |
| --- | ---: | --- |
| `src/` four-model `validate-ensemble` hour weights | `0.116167` | Current best no-leak reported route |
| `src/` four-model `validate-ensemble --weight-scope global` | `0.118018` | Previous SOTA baseline |
| `src/` default single model | `0.120175` | Current best single-model route |

## Commands

```sh
python run_task2_exp.py list
python run_task2_exp.py validate sota_single_extra
python run_task2_exp.py sweep --preset quick
```

## Results: 2026-06-05

Runtime:

```powershell
& 'C:\Users\SULAW\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_task2_exp.py sweep --preset tuning --output outputs/experiments/src1_tuning_sweep.csv
& 'C:\Users\SULAW\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_task2_exp.py sweep --preset boosting --output outputs/experiments/src1_boosting_sweep.csv
& 'C:\Users\SULAW\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_task2_exp.py sweep --preset tree --output outputs/experiments/src1_tree_sweep.csv
```

The default system `python` in this workspace is Python 3.14, while `.codex_deps/` is built for Python 3.12. The commands above use the bundled Python 3.12 runtime so the existing local dependencies can be loaded.

### Tuning Sweep

| Candidate | Main change | Phase1 MAPE |
| --- | --- | ---: |
| `sota_single_extra` | current single-model baseline | `0.120175` |
| `extra_lv_ratio07` | looser low-volume trigger | `0.120489` |
| `extra_lv_ratio08` | much looser low-volume trigger | `0.120489` |
| `extra_lv_ratio05` | stricter low-volume trigger | `0.121067` |
| `extra_weight02` | lower MAPE sample weighting | `0.121346` |
| `extra_weight025` | slightly lower MAPE sample weighting | `0.121932` |
| `extra_weight035` | slightly higher MAPE sample weighting | `0.122345` |
| `extra_weight04` | higher MAPE sample weighting | `0.123107` |

Takeaway: the current `sample_weight_power=0.3` and `low_volume_ratio=0.6` remain best among this local sweep. Loosening the low-volume trigger is close but not better.

### Tree Sweep

| Candidate | Main change | Phase1 MAPE |
| --- | --- | ---: |
| `sota_single_extra` | current single-model baseline | `0.120175` |
| `extra_leaf6` | lower ExtraTrees leaf size | `0.121849` |
| `extra_leaf14` | higher ExtraTrees leaf size | `0.121887` |
| `extra_depth18_leaf8` | deeper ExtraTrees | `0.121982` |
| `extra_global_leaf6` | remove low-volume block switch | `0.122689` |
| `extra_weather` | include weather features | `0.123011` |
| `extra_weight04` | higher MAPE sample weighting | `0.123107` |

Takeaway: coarse ExtraTrees tuning did not beat the current single-model setup. Weather still looks harmful under this feature set.

### Boosting Sweep

| Candidate | Main change | Phase1 MAPE |
| --- | --- | ---: |
| `lgbm_global` | LightGBM global model | `0.159851` |
| `hgb_global` | sklearn histogram boosting | `0.161509` |
| `xgb_global` | XGBoost global model | `0.163907` |

Takeaway: direct global boosting models are much weaker than ExtraTrees on the current features. They may still be useful as ensemble diversity, but not as standalone replacements.

## Current Conclusion

This round did not find a candidate that improves on the existing SOTA:

- Best isolated single model remains `sota_single_extra` at `0.120175`.
- At this point in the exploration, the best overall no-leak route remained `src/` four-model global `validate-ensemble` at `0.118018`.
- The most promising future direction is not broad model replacement; it is better ensemble calibration or adding a genuinely different, legally validated signal.

## Next Exploration Ideas

1. Learn fusion weights by combo or by morning/evening block using only train1 internal folds.
2. Add a legal trajectory-derived green-window feature set in `src1`, then validate before using it in any ensemble.
3. Build an internal rolling-fold selector so phase1 labels are not used to choose hyperparameters.
4. Try robust per-combo post-calibration learned only from train1 folds, then apply once to phase1.

## Results: 2026-06-05 Ensemble Weighting

Runtime:

```powershell
& 'C:\Users\SULAW\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_task2_exp.py ensemble-validate sota4_global
& 'C:\Users\SULAW\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_task2_exp.py ensemble-sweep --preset quick --output outputs/experiments/src1_ensemble_quick_sweep.csv
& 'C:\Users\SULAW\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_task2_exp.py ensemble-sweep --preset scopes --output outputs/experiments/src1_ensemble_scopes_sweep.csv
& 'C:\Users\SULAW\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_task2_exp.py ensemble-sweep --preset members --output outputs/experiments/src1_ensemble_members_sweep.csv
& 'C:\Users\SULAW\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_task2_exp.py ensemble-sweep --preset block_members --output outputs/experiments/src1_ensemble_block_members_sweep.csv
```

The `sota4_global` experiment reproduces the current SOTA four-model validation route:

| Ensemble | Weight scope | Phase1 MAPE | Learned weights |
| --- | --- | ---: | --- |
| `sota4_global` | global | `0.118018` | `low_volume_block=0.7770`, `xgb=0.0000`, `mlp=0.1536`, `ratio_lag7=0.0695` |

### Weight Scope Sweep

| Ensemble | Weight scope | Calibration MAPE | Phase1 MAPE | Takeaway |
| --- | --- | ---: | ---: | --- |
| `sota4_block_weights` | morning/evening | `0.137887` | `0.116096` | Best result in this round |
| `sota4_combo_weights` | combo | `0.135097` | `0.117580` | Improves over global, but less than block |
| `sota4_global` | global | `0.141467` | `0.118018` | Then-current documented SOTA |
| `sota4_combo_block_weights` | combo + block | `0.126075` | `0.119297` | Calibration improves but validation worsens, likely too many small groups |

Block-level learned weights from `outputs/experiments/src1_ensemble_scopes_sweep.csv`:

| Block | `low_volume_block` | `xgb` | `mlp` | `ratio_lag7` |
| --- | ---: | ---: | ---: | ---: |
| morning | `0.8974` | `0.0000` | `0.0291` | `0.0735` |
| evening | `0.3835` | `0.2217` | `0.2705` | `0.1242` |

Interpretation: morning windows mostly trust the ExtraTrees low-volume-block model. Evening windows benefit from a much more diversified blend, including XGBoost and MLP despite their weak standalone scores.

### Fusion Member Sweep

| Ensemble | Candidate change | Phase1 MAPE |
| --- | --- | ---: |
| `sota4_global` | original four candidates | `0.118018` |
| `sota4_no_xgb` | remove XGBoost | `0.118018` |
| `sota4_plus_ridge` | add ridge combo-slot | `0.118157` |
| `sota4_no_ratio` | remove ratio-lag7 | `0.119142` |
| `sota4_plus_lgbm` | add LightGBM | `0.119292` |
| `replace_xgb_lgbm` | replace XGBoost with LightGBM | `0.119310` |
| `sota4_no_mlp` | remove MLP | `0.119471` |
| `sota4_plus_boosting` | add LightGBM + HGB | `0.119732` |
| `sota4_plus_extra_variants` | add ExtraTrees variants | `0.120155` |

Takeaway: changing members under global weights did not improve on the original four-model set. The ratio-lag7 and MLP members still matter, even though their standalone MAPE is weak.

### Block Weights With Replaced Members

| Ensemble | Candidate change | Phase1 MAPE |
| --- | --- | ---: |
| `sota4_block_weights` | original four candidates | `0.116093` |
| `sota4_no_xgb_block_weights` | remove XGBoost | `0.116998` |
| `sota4_plus_extra_variants_block_weights` | add ExtraTrees variants | `0.117066` |
| `sota4_plus_lgbm_block_weights` | add LightGBM | `0.117094` |
| `sota4_plus_boosting_block_weights` | add LightGBM + HGB | `0.119161` |
| `sota4_plus_ridge_block_weights` | add ridge combo-slot | `0.123280` |

Takeaway: the improvement comes from block-level weighting, not from adding more model families. Adding weak or correlated candidates lets the optimizer fit the calibration week better but usually hurts phase1 validation.

## Current Ensemble Conclusion

`sota4_block_weights` is the strongest exploratory result so far:

- It uses the same four candidate models as the current SOTA.
- It still learns weights only from train1's latest internal fold.
- It does not use train2 labels to fit weights.
- It improves phase1 MAPE from `0.118018` to about `0.11609` in this exploration.

This is not yet promoted to `src/` SOTA because it was selected after looking at phase1 validation results during exploration. Before making it the official route, the next step should be an internal rolling-fold check inside train1 to see whether block-level weighting is consistently selected without using phase1 labels as the selector.

## Results: 2026-06-05 Continued Ensemble Search

This round continued from the block-weight finding and only records approaches that were compared against the existing `0.118018` four-model global baseline.

Runtime:

```powershell
& 'C:\Users\SULAW\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_task2_exp.py ensemble-sweep --preset shrinkage --output outputs/experiments/src1_ensemble_shrinkage_sweep.csv
& 'C:\Users\SULAW\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_task2_exp.py ensemble-sweep --preset granular_scopes --output outputs/experiments/src1_ensemble_granular_scopes_sweep.csv
& 'C:\Users\SULAW\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_task2_exp.py ensemble-sweep --preset fine_shrinkage --output outputs/experiments/src1_ensemble_fine_shrinkage_sweep.csv
& 'C:\Users\SULAW\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_task2_exp.py ensemble-sweep --preset micro_shrinkage --output outputs/experiments/src1_ensemble_micro_shrinkage_sweep.csv
& 'C:\Users\SULAW\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_task2_exp.py ensemble-sweep --preset second_stage --output outputs/experiments/src1_ensemble_second_stage_sweep.csv
& 'C:\Users\SULAW\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_task2_exp.py ensemble-sweep --preset post_scale --output outputs/experiments/src1_ensemble_post_scale_sweep.csv
```

### Results Below the `0.118018` Baseline

| Ensemble | Idea | Phase1 MAPE | Result |
| --- | --- | ---: | --- |
| `sota4_block_shrink14` | 86% block-weight prediction + 14% global-weight prediction | `0.116001` | Best in this round |
| `sota4_block_shrink15` | 85% block-weight prediction + 15% global-weight prediction | `0.116001` | Tied within rounding |
| `sota4_block_shrink16` | 84% block-weight prediction + 16% global-weight prediction | `0.116003` | Very close |
| `sota4_block_shrink18` | 82% block-weight prediction + 18% global-weight prediction | `0.116007` | Very close |
| `sota4_block_shrink12` | 88% block-weight prediction + 12% global-weight prediction | `0.116009` | Very close |
| `sota4_block_shrink20` | 80% block-weight prediction + 20% global-weight prediction | `0.116015` | Improves over plain block |
| `sota4_block_shrink10` | 90% block-weight prediction + 10% global-weight prediction | `0.116018` | Improves over plain block |
| `sota4_block_shrink05` | 95% block-weight prediction + 5% global-weight prediction | `0.116047` | Improves over plain block |
| `sota4_block_weights` | separate morning/evening weights | `0.116096` | Stronger than global baseline |
| `sota4_hour_weights` | separate target-hour weights | `0.116166` | Better than global, worse than block |
| `sota4_slot_weights` | separate 20-minute-slot weights | `0.116959` | Better than global, likely higher variance |
| `sota4_combo_weights` | separate combo weights | `0.117580` | Better than global, worse than block |

The 10%-20% shrinkage band is stable. The best result is effectively around `0.11600`, but the improvement from `0.116096` to `0.116001` comes from a manually swept shrinkage ratio. Treat this as an exploration result, not yet as a promoted SOTA.

### Strategies That Did Not Help

| Strategy | Best Phase1 MAPE | Interpretation |
| --- | ---: | --- |
| `sota4_block_global_blend` | `0.116096` | Letting the calibration fold learn a second-stage block/global blend chose no useful global shrinkage for phase1. |
| post-scale correction | `0.117089` after shrink14 + global scale | Calibration-fold scale made validation worse. |
| target-hour weights | `0.116166` | Useful, but not as good as simple morning/evening weights. |
| slot weights | `0.116959` | Still beats global, but likely too granular. |

### Current Best Exploratory Result

| Route | Phase1 MAPE | Compared To |
| --- | ---: | --- |
| Existing official four-model global weights | `0.118018` | previous documented SOTA |
| `src1` block weights | `0.116096` | valid no-leak experiment, selected after phase1 exploration |
| `src1` block weights + 14%-15% global shrink | `0.116001` | best exploratory result, shrink ratio selected by phase1 sweep |

Next validation step: add train1-only rolling folds for ensemble scope and shrinkage selection. If block weights and a small global shrinkage remain preferred inside train1, then this route is a strong candidate to promote back into `src/`.

## Results: 2026-06-09 Rolling-Fold Selection

This round tested whether the ensemble weighting improvements can be selected using train1 only, without looking at phase1 labels as the selector.

Runtime:

```powershell
& 'C:\Users\SULAW\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_task2_exp.py rolling-ensemble-sweep --preset micro_shrinkage --output outputs/experiments/src1_rolling_micro_shrinkage.csv
& 'C:\Users\SULAW\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_task2_exp.py rolling-ensemble-sweep --preset granular_scopes --output outputs/experiments/src1_rolling_granular_scopes.csv
& 'C:\Users\SULAW\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_task2_exp.py ensemble-validate sota4_hour_weights
& 'C:\Users\SULAW\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_task2_exp.py ensemble-validate sota4_block_shrink20
```

### Rolling Fold Setup

The runner used two train1-only 7-day validation folds:

| Fold | Validation days | Role |
| --- | --- | --- |
| 1 | `2016-10-04` to `2016-10-10` | National Day / post-holiday stress fold |
| 2 | `2016-10-11` to `2016-10-17` | Latest pre-phase1 fold |

For each fold, models are trained only on dates before the validation week. Fusion weights are learned from the latest internal calibration week before that fold. The validation week's red-window labels are used only for scoring that fold.

### Rolling Scope Results

| Ensemble | Fold scores | Mean MAPE | Phase1 MAPE |
| --- | --- | ---: | ---: |
| `sota4_hour_weights` | `0.300456`, `0.144388` | `0.222422` | `0.116166` |
| `sota4_global` | `0.303073`, `0.142993` | `0.223033` | `0.118018` |
| `sota4_block_weights` | `0.303073`, `0.143124` | `0.223099` | `0.116096` |
| `sota4_slot_weights` | `0.301792`, `0.145944` | `0.223868` | `0.116959` |
| `sota4_combo_weights` | `0.437802`, `0.143740` | `0.290771` | `0.117580` |

Interpretation: if selection is based on the mean of both rolling folds, `sota4_hour_weights` is preferred. This is a more defensible no-leak selector than choosing `block_shrink14` from phase1 directly, and it still beats the `0.118018` baseline on phase1.

### Rolling Shrinkage Results

| Ensemble | Fold scores | Mean MAPE | Phase1 MAPE |
| --- | --- | ---: | ---: |
| `sota4_block_shrink20` | `0.303073`, `0.143018` | `0.223046` | `0.116015` |
| `sota4_block_shrink18` | `0.303073`, `0.143028` | `0.223051` | `0.116007` |
| `sota4_block_shrink16` | `0.303073`, `0.143039` | `0.223056` | `0.116003` |
| `sota4_block_shrink15` | `0.303073`, `0.143044` | `0.223059` | `0.116001` |
| `sota4_block_shrink14` | `0.303073`, `0.143050` | `0.223062` | `0.116001` |

The first rolling fold is insensitive to shrinkage because block and global behavior coincide there. The latest rolling fold prefers more global shrinkage, with `sota4_block_shrink20` best inside train1 and still strong on phase1.

### Practical Update

There are now two useful candidates above the old `0.118018` baseline:

| Candidate | Why it is useful | Phase1 MAPE | Promotion status |
| --- | --- | ---: | --- |
| `sota4_hour_weights` | Selected by two-fold train1 rolling mean | `0.116166` | More defensible selector |
| `sota4_block_shrink20` | Selected by latest train1 rolling fold; near phase1 shrinkage optimum | `0.116015` | Stronger score, but more recency-driven |

## Promotion: 2026-06-09

`sota4_hour_weights` has been promoted back into `src/` as the default `validate-ensemble` and `predict-ensemble` behavior via `--weight-scope hour`.

Rationale:

- It keeps the same four candidate models as the previous SOTA.
- It changes only the blend granularity from global weights to target-hour weights.
- It was selected by train1 rolling-fold mean, not by directly choosing the best phase1 sweep result.
- It improves the no-leak phase1 MAPE from `0.118018` to `0.116167`.
- The previous global-weight SOTA remains reproducible with `--weight-scope global`.

## Results: 2026-06-13 Graph / GCN Exploration

This round tested whether representing the five tollgate-direction pairs as graph nodes and using graph convolution can improve Task 2 volume prediction.

Runtime:

```powershell
$env:PYTHONPATH='D:\Dev\kdd_cup_2017_su\.codex_deps'
& 'C:\Users\SULAW\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_task2_graph_exp.py --epochs 1000 --hidden 16 32 --output outputs/experiments/src1_graph_gcn_sweep.csv --graph-feature-output outputs/experiments/src1_graph_feature_extra.csv

.\.venv\Scripts\python.exe -m pip install torch
.\.venv\Scripts\python.exe run_task2_torch_graph_exp.py --epochs 1800 --hidden 32 64 --dropout 0.0 0.1 --output outputs/experiments/src1_torch_gcn_sweep.csv
.\.venv\Scripts\python.exe run_task2_torch_graph_exp.py --epochs 2500 --modes topology corr full --hidden 64 128 --dropout 0.0 0.1 --lr 0.001 --output outputs/experiments/src1_torch_gcn_lr001_sweep.csv
```

### Setup

- Nodes are the five `(tollgate_id, direction)` combos.
- Training labels use train1 only.
- Phase1 validation uses train1 for training/statistics, test1 green windows as legal inputs, and train2 labels only for scoring.
- Tested adjacency modes:
  - `identity`: no cross-node propagation.
  - `topology`: connect nodes with the same tollgate or same direction.
  - `full`: fully connected graph.
  - `corr`: positive train1 label correlations across nodes.

Two graph-style routes were tested:

1. A lightweight two-layer GCN implemented in numpy, trained on `log1p(volume)` with train1 internal early stopping.
2. A graph-convolved feature control: concatenate each node's original features with `A @ X` and `A @ X - X`, then train the same ExtraTrees-style tabular model.
3. A PyTorch graph neural network with node embeddings, self/neighbor message passing, LayerNorm, Dropout, and train1 internal early stopping.

### Pure GCN Results

| Mode | Hidden | Internal MAPE | Phase1 MAPE | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `identity` | `16` | `0.207365` | `0.172921` | Best pure GCN, still far below baseline |
| `identity` | `32` | `0.220283` | `0.209327` | Wider hidden layer worsens |
| `corr` | `32` | `0.199865` | `0.188591` | Correlation graph helps versus full/topology but not enough |
| `topology` | `16` | `0.240726` | `0.215161` | Manual graph smoothing hurts |
| `full` | `32` | `0.381742` | `0.388951` | Over-smoothing across all nodes is harmful |

Best pure GCN phase1 MAPE: `0.172921`.

### Graph-Convolved Feature Control

| Mode | Phase1 MAPE | Interpretation |
| --- | ---: | --- |
| `identity` | `0.121563` | Best graph-feature control, but still below current single model |
| `corr` | `0.122104` | Correlation neighbor features do not improve |
| `topology` | `0.124157` | Topology smoothing hurts |
| `full` | `0.131504` | Full graph smoothing hurts most |

Best graph-feature phase1 MAPE: `0.121563`, worse than the current single-model `0.120175` and much worse than the official hour-weight ensemble `0.116167`.

### PyTorch GNN Results

PyTorch `2.12.0+cpu` was installed into `.venv`. The model uses learnable node embeddings plus two self/neighbor graph message-passing layers. It is closer to a practical GNN than the small numpy prototype.

| Mode | Hidden | Dropout | LR | Internal MAPE | Phase1 MAPE | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `full` | `64` | `0.0` | `0.003` | `0.147408` | `0.133801` | Best torch GNN |
| `topology` | `64` | `0.0` | `0.003` | `0.150532` | `0.135571` | Best hand-topology graph |
| `corr` | `128` | `0.1` | `0.001` | `0.153298` | `0.136018` | Best lower-lr correlation graph |
| `topology` | `128` | `0.1` | `0.001` | `0.141295` | `0.136678` | More capacity does not close the gap |
| `identity` | `64` | `0.1` | `0.003` | `0.147753` | `0.142225` | No-graph neural baseline |

Best PyTorch GNN phase1 MAPE: `0.133801`.

This is a real improvement over the numpy GCN prototype (`0.172921`), but still much worse than:

- current single-model ExtraTrees low-volume-block: `0.120175`
- current official hour-weight ensemble: `0.116167`

### Conclusion

This idea does not currently look promising as a replacement or direct add-on:

- The graph has only five nodes, so message passing is very coarse.
- The hand-built topology between tollgate-direction pairs is weak; same tollgate or same direction does not imply similar target-volume residuals.
- Full or topology graph convolution over-smooths node-specific signals that the current combo and green-window features already capture.
- The neural GCN is data-hungry relative to this small tabular time-window dataset. PyTorch improves the implementation quality but does not change that data limitation.
- Adding graph-convolved features to ExtraTrees also fails to beat the existing single-model baseline.

Future graph work should only continue if it uses a richer, legal graph source such as route/trajectory-derived upstream relationships with strict per-window visibility. The current five-node tollgate graph is not enough.

## Results: 2026-06-13 Trajectory Candidate And Observation Adjustment

This round continued the graph discussion by testing richer legal signals derived from trajectory records and from same-day green observation windows.

### Optimizer Fix

The first train1-only rolling check for trajectory-capped blending exposed a bug in the exploratory capped optimizer:

- With `trajectory_cap=0`, the optimizer should exactly reduce to the four-model block-weight baseline.
- Instead, the first rolling fold returned zero predictions and MAPE `1.000000`.
- Root cause: a short-history MLP candidate produced an extreme calibration value, SLSQP reported failed/infeasible results, and the exploratory capped optimizer did not reject an all-zero infeasible weight vector.
- Fix: `src1/kddcup2017_task2_exp/trajectory_ensemble_exp.py` now seeds only feasible starts, evaluates feasible starts as a fallback, and accepts optimized results only when they satisfy the weight-sum and cap constraints.

After the fix, `trajectory_cap=0` on the first rolling fold returns the expected four-model block baseline MAPE `0.303073`.

### Trajectory As A Capped Fifth Candidate

Runtime:

```powershell
.\.venv\Scripts\python.exe run_task2_traj_ensemble_exp.py --weight-scopes global block hour --trajectory-caps 0.00 0.05 0.10 0.15 0.20 0.30 0.40 0.50 --output outputs/experiments/src1_trajectory_ensemble_caps_fixed.csv
.\.venv\Scripts\python.exe run_task2_traj_rolling_exp.py --caps 0.00 0.05 0.10 0.15 0.20 0.30 --output outputs/experiments/src1_trajectory_rolling_caps_fixed.csv
```

Best phase1 trajectory-capped result:

| Route | Phase1 MAPE | Notes |
| --- | ---: | --- |
| block weights + trajectory cap `0.15`, no route means | `0.115924` | Best direct phase1 trajectory result |
| block weights + trajectory cap `0.10`, no route means | `0.115934` | Very close |
| hour weights + trajectory cap `0.10`, no route means | `0.116021` | Slightly better than current official hour route |
| current official hour four-model route | `0.116167` | Current `src/` SOTA |

Rolling-fold check:

| Candidate | Fold scores | Mean MAPE | Interpretation |
| --- | --- | ---: | --- |
| block, cap `0.00` | `0.303073`, `0.143124` | `0.223099` | Four-model block baseline |
| block, cap `0.20`, no route means | `0.303073`, `0.141428` | `0.222251` | Trajectory helps latest fold only |
| block, cap `0.30`, route means | `0.303073`, `0.140642` | `0.221858` | Best rolling trajectory-only variant |

Interpretation: trajectory records contain legal incremental signal, but the gain is fold-dependent. A small cap improves phase1, while rolling folds prefer more cap mainly because the latest fold improves and the holiday-stress fold is insensitive after the optimizer fix.

### Observation-Window Posterior Adjustment

Idea: after producing an ensemble prediction, apply a small multiplicative correction using only the same-day green observation window strength:

```text
adjusted_prediction = base_prediction * exp(beta * log((current_obs_sum + smoothing) / (historical_expected_obs_sum + smoothing)))
```

The expected observation sum is computed from legal prior training days by `(combo, block)` or `(combo, block, day_of_week)`. The beta values are learned only on the train1 calibration fold. Phase1 labels are used only for scoring exploratory outputs.

Runtime:

```powershell
.\.venv\Scripts\python.exe run_task2_obs_adjust_exp.py --beta-max 0.10 --output outputs/experiments/src1_observation_adjust_beta010.csv
.\.venv\Scripts\python.exe run_task2_obs_adjust_rolling_exp.py --beta-max 0.10 --output outputs/experiments/src1_observation_adjust_rolling_beta010.csv --summary-output outputs/experiments/src1_observation_adjust_rolling_beta010_summary.csv
.\.venv\Scripts\python.exe run_task2_obs_adjust_exp.py --beta-max 0.05 --output outputs/experiments/src1_observation_adjust_beta005.csv
.\.venv\Scripts\python.exe run_task2_obs_adjust_exp.py --beta-max 0.15 --output outputs/experiments/src1_observation_adjust_beta015.csv
.\.venv\Scripts\python.exe run_task2_obs_adjust_exp.py --beta-max 0.20 --output outputs/experiments/src1_observation_adjust_beta020.csv
.\.venv\Scripts\python.exe run_task2_obs_adjust_rolling_exp.py --beta-max 0.05 --output outputs/experiments/src1_observation_adjust_rolling_beta005.csv --summary-output outputs/experiments/src1_observation_adjust_rolling_beta005_summary.csv
.\.venv\Scripts\python.exe run_task2_obs_adjust_rolling_exp.py --beta-max 0.15 --output outputs/experiments/src1_observation_adjust_rolling_beta015.csv --summary-output outputs/experiments/src1_observation_adjust_rolling_beta015_summary.csv
.\.venv\Scripts\python.exe run_task2_obs_adjust_rolling_exp.py --beta-max 0.20 --output outputs/experiments/src1_observation_adjust_rolling_beta020.csv --summary-output outputs/experiments/src1_observation_adjust_rolling_beta020_summary.csv
```

Best direct phase1 results:

| Route | Phase1 MAPE | Selection caveat |
| --- | ---: | --- |
| `traj_hour_cap010` + `(combo, block)` expected obs + hour beta, `beta_max=0.10` | `0.114456` | Strongest phase1 result, but chosen from phase1 sweep |
| `hour4` + `(combo, block)` expected obs + hour beta, `beta_max=0.10` | `0.114607` | Does not require trajectory fifth candidate |
| `traj_block_cap020` + `(combo, block, dow)` expected obs + block beta, `beta_max=0.10` | `0.115834` | Also selected near the top by rolling |

Rolling-fold results by beta cap:

| Beta cap | Best rolling route | Fold scores | Mean MAPE | Same route phase1 MAPE |
| ---: | --- | --- | ---: | ---: |
| `0.05` | `traj_block_cap020`, `(combo, block, dow)`, block beta | `0.295705`, `0.140619` | `0.218162` | `0.115834`-range |
| `0.10` | `traj_block_cap020`, `(combo, block, dow)`, block beta | `0.290683`, `0.140061` | `0.215372` | `0.115834` |
| `0.15` | `traj_block_cap020`, `(combo, block, dow)`, block beta | `0.287673`, `0.139862` | `0.213768` | `0.115868` |
| `0.20` | `traj_block_cap020`, `(combo, block, dow)`, block beta | `0.286962`, `0.139852` | `0.213407` | `0.115834` |

The unrestricted `beta_max=0.80` variant can find phase1 MAPE `0.115536`, but rolling selection rejects the adjusted variants because the first fold overfits. Capping beta is therefore essential.

### Current Interpretation

Two new promising variants now exist:

| Candidate | Why it matters | Phase1 MAPE | Status |
| --- | --- | ---: | --- |
| Direct phase1 best observation-adjusted route | Shows the same-day green-window strength signal can reduce error materially | `0.114456` | Exploratory only; selected by phase1 sweep |
| Rolling-supported observation-adjusted route | Selected by train1-only rolling among the tested observation-adjustment variants | about `0.11583` | More defensible candidate to consider for promotion |

This direction is more convincing than the five-node GCN route because the signal is directly tied to the problem statement: the current-day green windows are the only legal observations available before predicting the red windows. The adjustment is also easy to explain: if the green window is stronger than historical expectation, gently lift the target-window prediction; if weaker, gently lower it. The cap on beta prevents this signal from overriding the base ensemble.

## Route Organization Status

The current route classification has been summarized in `docs/route_exploration_candidates.md`.

Current decision:

- Do not promote any `src1` exploration route to `src/` yet.
- Keep the official `src/` SOTA as the four-model hour-weight ensemble with phase1 MAPE about `0.116167`.
- Treat the phase1-best observation-adjusted result `0.114456` as an exploratory upper bound because the exact configuration was selected from the phase1 sweep.
- Treat the rolling-supported observation-adjusted route around `0.11583` as the best candidate for the next formalization pass.
- Keep trajectory-capped blending as a useful supporting candidate, not the main promotion target.
- Keep the five-node graph/GCN route documented as a tried but currently low-priority direction.

## Results: 2026-06-13 Additional Neural-Network Exploration

This round broadened neural-network experiments beyond the previous five-node GCN route. All work stayed in `src1/`; nothing was promoted to `src/`.

Runtime:

```powershell
.\.venv\Scripts\python.exe run_task2_torch_nn_exp.py --methods tabular sequence calibrator --hidden 32 --dropout 0.1 --epochs 500 --correction-scales 0.05 0.10 --calibrator-bases hour4 traj_hour_cap010 --output outputs/experiments/src1_torch_nn_smoke.csv
.\.venv\Scripts\python.exe run_task2_torch_nn_exp.py --methods gate --hidden 16 --dropout 0.1 --epochs 300 --correction-scales 0.05 0.10 --calibrator-bases hour4 traj_hour_cap010 --output outputs/experiments/src1_torch_nn_gate_small.csv
.\.venv\Scripts\python.exe run_task2_torch_nn_exp.py --methods gate --hidden 16 --dropout 0.1 --epochs 300 --correction-scales 0.02 0.05 0.10 0.20 --calibrator-bases traj_block_cap020 --output outputs/experiments/src1_torch_nn_gate_traj_block.csv
.\.venv\Scripts\python.exe run_task2_torch_nn_exp.py --methods gate --hidden 16 32 --dropout 0.0 0.1 --epochs 400 --correction-scales 0.20 0.30 --calibrator-bases traj_block_cap020 --output outputs/experiments/src1_torch_nn_gate_traj_block_refine.csv
.\.venv\Scripts\python.exe run_task2_torch_nn_exp.py --methods gate --hidden 16 --dropout 0.0 --epochs 400 --correction-scales 0.35 0.40 0.50 --calibrator-bases traj_block_cap020 --output outputs/experiments/src1_torch_nn_gate_traj_block_scale_high.csv
.\.venv\Scripts\python.exe run_task2_torch_nn_exp.py --methods gate --hidden 16 --dropout 0.0 --epochs 400 --correction-scales 0.38 0.42 0.45 --calibrator-bases traj_block_cap020 --output outputs/experiments/src1_torch_nn_gate_traj_block_scale_peak.csv
.\.venv\Scripts\python.exe run_task2_torch_nn_exp.py --methods tabular sequence --hidden 64 --dropout 0.0 0.1 --epochs 800 --lr 0.001 --output outputs/experiments/src1_torch_nn_direct_lr001.csv
```

### Tested Neural Ideas

| Method | Architecture | Input | Purpose |
| --- | --- | --- | --- |
| direct tabular | MLP / residual MLP | existing FeatureBuilder vector | Check whether a pure neural tabular regressor can replace ExtraTrees |
| direct sequence | GRU / Conv1D | 6 green-window observations + 14-day target-slot history + categorical embeddings | Check whether neural sequence encoders capture temporal shape better |
| neural residual calibrator | MLP bounded multiplicative correction | base ensemble predictions + observation strength + categorical context | Learn a nonlinear post-calibration over existing ensemble output |
| neural prior gate | MLP constrained candidate reweighting | base candidate predictions + prior ensemble weights + observation context | Reallocate candidate weights without discarding the known-good ensemble prior |

### Direct Neural Predictors

| Route | Best phase1 MAPE | Interpretation |
| --- | ---: | --- |
| tabular ResNet, hidden `32`, dropout `0.1`, lr `0.003` | `0.145551` | Better than sequence nets, but far below ExtraTrees |
| tabular ResNet, hidden `64`, dropout `0.0`, lr `0.001` | `0.151451` | Lower lr did not help |
| sequence Conv1D, hidden `64`, dropout `0.1`, lr `0.001` | `0.164807` | Still weak |
| sequence GRU | `0.282066` or worse in these sweeps | Underpredicts and does not match tabular baseline |

Direct neural prediction is not competitive on this small tabular dataset. It can fit the latest internal fold moderately, but phase1 generalization is poor compared with tree models.

### Neural Residual Calibrator

| Base | Scale | Phase1 MAPE | Interpretation |
| --- | ---: | ---: | --- |
| `traj_hour_cap010` | `0.05` | `0.120547` | Best residual calibrator smoke result |
| `hour4` | `0.05` | `0.121707` | Worse than base hour ensemble |
| larger scale `0.10` | - | `0.129496`-`0.131666` | Over-corrects |

The residual calibrator is too unconstrained relative to the small calibration fold. It improves calibration MAPE but hurts phase1 validation.

### Neural Prior Gate

The most promising neural route is not direct prediction. It is a small MLP that starts from existing ensemble weights and learns bounded residual logits:

```text
weights = softmax(log(prior_weights) + gate_scale * tanh(MLP(context)))
prediction = sum(weights_i * candidate_i)
```

This keeps the model anchored to the known-good four/five-model ensemble while allowing context-dependent reweighting.

Best results so far:

| Base prior | Hidden | Dropout | Gate scale | Seed | Phase1 MAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `traj_block_cap020` | `16` | `0.0` | `0.40` | `13` | `0.114758` |
| `traj_block_cap020` | `16` | `0.0` | `0.30` | `13` | `0.114983` |
| `traj_block_cap020` | `16` | `0.0` | `0.38` | `13` | `0.114845` |
| `traj_block_cap020` | `16` | `0.0` | `0.42` | `13` | `0.114805` |
| `traj_hour_cap010` | `16` | `0.1` | `0.10` | `13` | `0.115677` |

Seed check for the best-looking `traj_block_cap020`, hidden `16`, dropout `0.0`, gate scale `0.40`:

| Seed | Phase1 MAPE |
| ---: | ---: |
| `7` | `0.115089` |
| `13` | `0.114758` |
| `21` | `0.116207` |
| `42` | `0.115718` |

Interpretation:

- Neural prior gating can beat the official `0.116167` SOTA and the trajectory-capped `0.115924` result in direct phase1 scoring.
- It is still less stable than the explicit observation-window adjustment route.
- The result is seed-sensitive and selected by phase1 exploration, so it must remain exploratory until a train1-only rolling selection protocol is added.
- The most defensible next neural step is a rolling version of the prior gate, with candidate matrices cached so the sweep is not prohibitively slow.
