#!/usr/bin/env python
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
for extra_path in (ROOT / "src", ROOT / "src1"):
    if extra_path.exists():
        sys.path.insert(0, str(extra_path))

try:
    from kddcup2017_task2_exp.graph_gcn import main
except ImportError as exc:
    raise SystemExit(
        "run_task2_graph_exp.py requires numpy, scikit-learn, and scipy. "
        "Create .venv and install dependencies with: "
        ".venv\\Scripts\\python -m pip install -r requirements.txt"
    ) from exc


if __name__ == "__main__":
    main()
