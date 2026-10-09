"""Shared, frozen protocol for causal next-M1 prediction."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/angular_mbo_causal_20261008'
CACHE = OUT / 'cache'
START, VALID, TEST, END = '2025-10-01', '2026-06-01', '2026-07-01', '2026-07-31'
CONTEXT, EVENTS, MAX_AGE = 512, 2048, 1800
LAGS = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55])
COLS = [0, 2, 3]
RAW = np.dtype([('ts_recv', '<u8'), ('price', '<i8'), ('size', '<u4'),
                ('action', 'S1'), ('side', 'S1'), ('flags', 'u1'), ('ts_in_delta', '<i4')])

def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    temp.replace(path)
