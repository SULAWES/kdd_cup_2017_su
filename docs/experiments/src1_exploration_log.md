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
