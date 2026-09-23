"""SOUND INSIGHT: ローカルCPU音声検査。"""
import os
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"
os.environ.setdefault("NUMBA_CACHE_DIR", str(_root / ".cache" / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(_root / ".cache" / "matplotlib"))
