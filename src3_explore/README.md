# src3_explore: Task 2 structure diagnostics

`src3_explore/` is an isolated exploration workspace for understanding predictable structure, noise, anomalies, and model failure modes in KDD Cup 2017 Task 2. It does not replace or import changes into the official `src/`, `src1/`, or `src2/` routes.

## Scope

This workspace is not for short-term leaderboard tuning. Its default outputs are CSV files, small SVG charts, and experiment cards under `outputs/src3_explore/`.

Run one experiment:

```powershell
.\.venv\Scripts\python.exe -m src3_explore residual_atlas
```

List experiments:

```powershell
.\.venv\Scripts\python.exe -m src3_explore list
```

Run all prototypes:

```powershell
.\.venv\Scripts\python.exe -m src3_explore all
```

Some experiments reuse the official four-model candidate matrix and may take as long as the ensemble validation path.

## Visibility policy

All modules should enter data through `src3_explore.common.visibility`:

- train1 rolling diagnostics train only on earlier train1 days and expose only held-out same-day green windows.
- phase1 diagnostics train on train1 and expose test1 green windows.
- train2 labels may be joined only after predictions are fixed, for final phase1 observation metrics.
- phase2 rows are visible but unlabeled; test2 target red windows are never loaded as labels.

Phase1 numbers emitted here are diagnostic observations, not SOTA claims. Repeated phase1 sweeps must not be used to select parameters for the formal route.

## Modules

| Area | Module | Purpose |
| --- | --- | --- |
| diagnostics | `residual_atlas.py` | Cache candidate predictions and summarize residuals by date, combo, hour, slot, green strength, ETC share, trajectory signal, and model disagreement. |
| diagnostics | `model_disagreement.py` | Identify which candidate model is closest when candidate predictions disagree. |
| diagnostics | `green_red_transfer_analysis.py` | Fit constrained 6x6 green-to-red transfer, cluster green shapes, and export ratio surfaces. |
| diagnostics | `adversarial_validation.py` | Test distribution shift between train1, phase1-visible, phase2-visible, and time splits. |
| representations | `curve_dictionary.py` | Use PCA/NMF/dictionary day curves to reconstruct red slots from green slots. |
| representations | `day_embedding_clustering.py` | Build day embeddings and label regimes such as weekday/weekend, holiday/post-holiday, low-volume, ETC anomaly, and tollgate allocation anomaly. |
| mechanisms | `route_arrival_kernel.py` | Study route/trajectory lead-lag kernels instead of continuing five-node GNN tuning. |
| mechanisms | `tollgate12_allocation.py` | Analyze `z12 = y1 + y2` and `r2 = y2 / (y1 + y2)` for allocation anomalies. |
| mechanisms | `etc_component_model.py` | Treat ETC, vehicle model, and vehicle type as generated sub-flow components and compare component reconciliation. |
| probabilistic | `quantile_baselines.py` | Fit p10/p50/p90 quantile baselines and score coverage. |
| probabilistic | `conformal_intervals.py` | Use train1 calibration residuals for conformal intervals and compare with ensemble spread. |

## Initial interpretation from existing docs

Likely true signal:

- Same-day green observation strength is a real, problem-aligned signal. Existing `src1` observation posterior adjustment is the strongest candidate, but it still needs train1-only selection before promotion.
- The official four-model ensemble has useful error diversity, especially by target hour.
- Route/trajectory data has incremental signal, but previous fifth-candidate blending was fold-dependent and should be treated as supporting evidence.
- Low-volume regime behavior, especially around `1_0`, is a real failure mode for global models.

Likely noise or low-priority directions:

- Direct LSTM and Transformer sequence models in `src2` are runnable but far behind the tree ensemble.
- Most direct tabular/sequence neural predictors are weak on this small tabular dataset.
- The five-node tollgate GNN is a useful contrast result, not a route to keep tuning in place.
- Weather and broad feature re-addition have not shown stable gain in the formal route.

Worth continuing:

- Train1-only protocol for green observation posterior adjustment.
- Mechanism checks that connect residuals to green strength, route lead-lag counts, ETC/component shifts, and tollgate 1/2 allocation.
- Uncertainty diagnostics where model disagreement or conformal intervals identify failure cases before seeing labels.

Archive unless new evidence appears:

- Further five-node GNN hyperparameter tuning without richer route/trajectory graph structure.
- Phase1-selected caps, beta values, or neural gate scales without rolling support.
- Direct neural sequence prediction as a replacement for the current tree/ensemble route.

## Output contract

Every experiment writes an experiment card with:

- hypothesis
- data visibility
- prototype
- metrics
- result
- insight
- next step

Metrics should include more than pooled MAPE where possible: signed error, grouped error, coverage, regime-specific behavior, and failure-case rows.
