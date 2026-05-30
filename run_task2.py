#!/usr/bin/env python
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kddcup2017_task2.pipeline import main


if __name__ == "__main__":
    main()
