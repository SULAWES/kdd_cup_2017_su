# Task 2 Exploration Workspace

`src1/` is an isolated exploration area. The current best-known solution remains in `src/` and `docs/sota/`.

Rules for this directory:

- Do not change `src/` while testing speculative ideas.
- Use `run_task2_exp.py` for experiments.
- Treat phase1 scores from repeated sweeps as exploration evidence, not as a new SOTA claim.
- Promote an idea back to `src/` only after it passes a no-leak validation protocol and is documented.

Quick commands:

```sh
python run_task2_exp.py list
python run_task2_exp.py validate sota_single_extra
python run_task2_exp.py sweep --preset quick
python run_task2_exp.py ensemble-list --preset quick
python run_task2_exp.py ensemble-validate sota4_global
python run_task2_exp.py ensemble-sweep --preset scopes
```

In the current Codex desktop environment, the default `python` is 3.14 while `.codex_deps/` contains Python 3.12 wheels. Use the bundled Python 3.12 runtime or install matching dependencies for your local interpreter.

The experiment runner reuses the stable data and feature code from `src/kddcup2017_task2`, but candidate model families and hyperparameters live in `src1/kddcup2017_task2_exp`.
