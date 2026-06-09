#!/usr/bin/env python
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
for extra_path in (ROOT / "src", ROOT / "src1"):
    if extra_path.exists():
        sys.path.insert(0, str(extra_path))

codex_deps = ROOT / ".codex_deps"
if codex_deps.exists() and sys.version_info[:2] == (3, 12):
    sys.path.insert(0, str(codex_deps))

try:
    from kddcup2017_task2_exp.experiments import main
except ImportError as exc:
    raise SystemExit(
        "run_task2_exp.py requires numpy, scikit-learn, scipy, and optional xgboost/lightgbm. "
        "This checkout has .codex_deps built for Python 3.12, so use a Python 3.12 runtime "
        "or install matching dependencies for the current interpreter."
    ) from exc


if __name__ == "__main__":
    main()
