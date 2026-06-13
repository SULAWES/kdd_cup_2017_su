# src2 Exploration Log

This file records the second isolated exploration area requested for speculative Task 2 neural models. Nothing here is promoted to the official `src/` implementation.

## 2026-06-13 LSTM And Transformer Sequence Baselines

### Data Boundary

- Phase1-style validation.
- Model fitting uses train1 labels only.
- Internal early stopping holds out the latest train1 week.
- For that internal holdout week, features expose earlier train1 labels plus only same-day green observation windows from the holdout days.
- Final phase1 scoring uses test1 green windows as inputs and train2 labels only for final MAPE.
- No phase1 result here is a formal SOTA claim.

### Environment

```text
torch_version=2.12.0+cpu
torch_cuda=None
torch_cuda_available=False
device=cpu
```

An attempted CUDA wheel install was stopped because the download was large. Pip cache was purged afterward, removing about 1782.5 MB of cached files. The CPU torch install remains active.

### Commands

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_src2_sequence_nn_exp
.\.venv\Scripts\python.exe -m py_compile run_task2_src2_nn_exp.py src2\kddcup2017_task2_exp2\__init__.py src2\kddcup2017_task2_exp2\sequence_nn_exp.py tests\test_src2_sequence_nn_exp.py
.\.venv\Scripts\python.exe run_task2_src2_nn_exp.py --device cpu --methods lstm transformer --hidden 16 --epochs 30 --patience 30 --output outputs/experiments/src2_sequence_nn_smoke.csv
.\.venv\Scripts\python.exe run_task2_src2_nn_exp.py --device cpu --methods lstm transformer --hidden 32 --dropout 0.0 --epochs 120 --patience 80 --output outputs/experiments/src2_sequence_nn_cpu_try.csv
.\.venv\Scripts\python.exe run_task2_src2_nn_exp.py --device cpu --methods lstm transformer --hidden 64 --dropout 0.1 --lr 0.001 --epochs 180 --patience 100 --output outputs/experiments/src2_sequence_nn_cpu_try_lr001.csv
```

### Results

| Method | Hidden | Dropout | LR | Best epoch | Internal MAPE | Phase1 MAPE | Pred mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM | 16 | 0.1 | 0.003 | 20 | 0.321201 | 0.395921 | 50.937 |
| Transformer | 16 | 0.1 | 0.003 | 20 | 0.381330 | 0.388355 | 53.106 |
| LSTM | 32 | 0.0 | 0.003 | 80 | 0.220117 | 0.193614 | 73.349 |
| Transformer | 32 | 0.0 | 0.003 | 120 | 0.201311 | 0.292356 | 93.571 |
| LSTM | 64 | 0.1 | 0.001 | 160 | 0.212111 | 0.194850 | 74.054 |
| Transformer | 64 | 0.1 | 0.001 | 40 | 0.240544 | 0.191686 | 74.734 |

### Interpretation

The LSTM and Transformer runners work as direct sequence baselines, but their phase1 MAPE remains much worse than the official four-model ensemble (`0.116167`) and the best single ExtraTrees route (`0.120175`). The lower learning-rate Transformer run is the best initial `src2` result, but the gap is too large to justify promotion. This folder is useful for future architecture experiments or neural pretraining ideas, not for replacing the current formal route.
