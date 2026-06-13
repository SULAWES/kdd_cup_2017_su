# Task 2 src2 Exploration Workspace

`src2/` is a second isolated exploration area for speculative neural-network ideas. It does not replace the official implementation in `src/`.

Current focus:

- direct LSTM sequence regressor
- direct Transformer sequence regressor
- legal phase1-style validation boundary

The sequence input uses:

- same-day green observation windows for the target block
- previous same-slot daily target history
- combo, target hour, weekday, minute, block, and simple derived context

Internal early stopping holds out the latest train1 week. For that holdout week, the known aggregate contains earlier train1 labels plus only the holdout days' green observation windows. Phase1 train2 labels are used only for final scoring.

Quick CPU smoke command:

```powershell
.\.venv\Scripts\python.exe run_task2_src2_nn_exp.py --device cpu --methods lstm transformer --hidden 16 --epochs 30 --patience 30 --output outputs/experiments/src2_sequence_nn_smoke.csv
```

Longer exploratory command:

```powershell
.\.venv\Scripts\python.exe run_task2_src2_nn_exp.py --device auto --methods lstm transformer --hidden 32 64 --dropout 0.0 0.1 --epochs 300 --output outputs/experiments/src2_sequence_nn_sweep.csv
```

Treat all phase1 scores from this folder as exploratory evidence only.

## Initial CPU Results

Environment:

- `.venv` Python 3.14
- `torch==2.12.0+cpu`
- CUDA not used; the attempted GPU wheel download was stopped and pip cache was cleared.

Commands run:

```powershell
.\.venv\Scripts\python.exe run_task2_src2_nn_exp.py --device cpu --methods lstm transformer --hidden 16 --epochs 30 --patience 30 --output outputs/experiments/src2_sequence_nn_smoke.csv
.\.venv\Scripts\python.exe run_task2_src2_nn_exp.py --device cpu --methods lstm transformer --hidden 32 --dropout 0.0 --epochs 120 --patience 80 --output outputs/experiments/src2_sequence_nn_cpu_try.csv
.\.venv\Scripts\python.exe run_task2_src2_nn_exp.py --device cpu --methods lstm transformer --hidden 64 --dropout 0.1 --lr 0.001 --epochs 180 --patience 100 --output outputs/experiments/src2_sequence_nn_cpu_try_lr001.csv
```

Observed exploratory phase1 MAPE:

| Method | Hidden | Dropout | LR | Epoch selected | Phase1 MAPE | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Transformer | 64 | 0.1 | 0.001 | 40 | 0.191686 | Best initial src2 result |
| LSTM | 32 | 0.0 | 0.003 | 80 | 0.193614 | Similar to larger LSTM |
| LSTM | 64 | 0.1 | 0.001 | 160 | 0.194850 | Lower LR did not improve LSTM |
| Transformer | 32 | 0.0 | 0.003 | 120 | 0.292356 | Overpredicts on phase1 |
| Transformer | 16 | 0.1 | 0.003 | 20 | 0.388355 | Smoke only |
| LSTM | 16 | 0.1 | 0.003 | 20 | 0.395921 | Smoke only |

Interpretation: these direct sequence neural models run end-to-end but are not competitive with the official tree/ensemble routes. Keep them as exploratory baselines only.
