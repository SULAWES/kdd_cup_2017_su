#!/usr/bin/env python
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
for extra_path in (ROOT / "src", ROOT / "src1"):
    if extra_path.exists():
        sys.path.insert(0, str(extra_path))

from kddcup2017_task2_exp.observation_adjust_rolling_exp import main


if __name__ == "__main__":
    main()
