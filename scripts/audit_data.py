"""Reproduce the read-only audit of the local XAUUSD M1 parquet file.

Run from the repository root: python scripts/audit_data.py
The report contains observations and embedded metadata claims, not a source attestation.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/M1/xauusd_m1.parquet"
DESTINATION = ROOT / "reports/data-audit.json"
MINUTE_NS = 60_000_000_000


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def examples(mask: np.ndarray, frame: pd.DataFrame, limit: int = 5) -> list[dict]:
    positions = np.flatnonzero(mask)[:limit]
    return [
        {"row_index_zero_based": int(i), "timestamp": frame["timestamp"].iloc[i].isoformat()}
        for i in positions
    ]


def main() -> None:
    parquet = pq.ParquetFile(SOURCE)
    schema = parquet.schema_arrow
    metadata_claims = {
        key.decode("utf-8", "replace"): value.decode("utf-8", "replace")
        for key, value in (schema.metadata or {}).items()
        if key != b"ARROW:schema"
    }
    frame = parquet.read().to_pandas()
    if list(frame.columns) != ["timestamp", "open", "high", "low", "close", "volume"]:
        raise ValueError(f"Unexpected columns: {list(frame.columns)}")
    n = len(frame)
    ts = frame["timestamp"]
    valid_ts = ts.notna().to_numpy()
    # pandas 3 can store datetime arrays at microsecond resolution even when the
    # Arrow field is timestamp[ms]; explicitly convert before minute arithmetic.
    ts_ns = ts.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    diffs = np.diff(ts_ns)
    valid_pairs = valid_ts[1:] & valid_ts[:-1]
    positive_gap = valid_pairs & (diffs > MINUTE_NS)
    gap_positions = np.flatnonzero(positive_gap)
    largest_gap_positions = sorted(gap_positions, key=lambda i: int(diffs[i]), reverse=True)[:10]
    duplicate_ts = ts.duplicated(keep=False).to_numpy() & valid_ts
    prices = frame[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
    o, h, l, c = prices.T
    finite_prices = np.isfinite(prices).all(axis=1)
    positive_prices = (prices > 0).all(axis=1)
    bad_envelope = finite_prices & ((h < l) | (o < l) | (o > h) | (c < l) | (c > h))
    volume = frame["volume"].to_numpy(dtype=np.float64)
    weekday = ts.dt.weekday
    years = ts.dt.year
    versions = {}
    for package in ["torch", "numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "psutil"]:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    interval_counts = Counter(int(v // MINUTE_NS) for v in diffs[valid_pairs & (diffs > 0) & (diffs % MINUTE_NS == 0)])
    compute = {"logical_cpu_count": os.cpu_count()}
    try:
        import psutil

        compute["physical_memory_bytes"] = psutil.virtual_memory().total
    except ImportError:
        compute["physical_memory_bytes"] = None
    try:
        import torch

        compute["torch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            compute["cuda_device_0"] = torch.cuda.get_device_name(0)
            compute["cuda_device_0_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except ImportError:
        compute["torch_cuda_available"] = None
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "bytes": SOURCE.stat().st_size,
            "sha256": digest(SOURCE),
            "parquet_rows": parquet.metadata.num_rows,
            "parquet_row_groups": parquet.metadata.num_row_groups,
            "parquet_created_by": parquet.metadata.created_by,
            "schema": str(schema.remove_metadata()),
            "embedded_metadata_claims": metadata_claims,
        },
        "software": {"python": sys.version.split()[0], "platform": platform.platform(), **versions},
        "compute_resources": compute,
        "observations": {
            "rows_read": n,
            "null_counts": {column: int(frame[column].isna().sum()) for column in frame.columns},
            "first_row_timestamp": ts.iloc[0].isoformat() if valid_ts[0] else None,
            "last_row_timestamp": ts.iloc[-1].isoformat() if valid_ts[-1] else None,
            "minimum_timestamp": ts.min().isoformat(),
            "maximum_timestamp": ts.max().isoformat(),
            "rows_by_utc_year": {str(int(k)): int(v) for k, v in years.value_counts().sort_index().items()},
            "rows_by_utc_weekday_monday_zero": {str(int(k)): int(v) for k, v in weekday.value_counts().sort_index().items()},
            "weekend_utc_rows": int((weekday >= 5).sum()),
            "duplicate_timestamp_rows": int(duplicate_ts.sum()),
            "duplicate_timestamp_groups": int(ts[duplicate_ts].nunique()),
            "duplicate_examples": examples(duplicate_ts, frame),
            "descending_adjacent_pairs": int((valid_pairs & (diffs < 0)).sum()),
            "equal_adjacent_pairs": int((valid_pairs & (diffs == 0)).sum()),
            "non_minute_aligned_timestamps": int((valid_ts & (ts_ns % MINUTE_NS != 0)).sum()),
            "positive_non_integer_minute_intervals": int((valid_pairs & (diffs > 0) & (diffs % MINUTE_NS != 0)).sum()),
            "adjacent_interval_minutes_top": [{"minutes": k, "count": v} for k, v in interval_counts.most_common(12)],
            "gaps_over_one_minute": int(positive_gap.sum()),
            "missing_minute_slots_between_adjacent_rows": int(np.sum(diffs[positive_gap] // MINUTE_NS - 1)),
            "largest_gaps": [
                {
                    "after_row_index_zero_based": int(i),
                    "from": ts.iloc[i].isoformat(),
                    "to": ts.iloc[i + 1].isoformat(),
                    "elapsed_minutes": int(diffs[i] // MINUTE_NS),
                }
                for i in largest_gap_positions
            ],
            "nonfinite_price_rows": int((~finite_prices).sum()),
            "nonpositive_price_rows": int((finite_prices & ~positive_prices).sum()),
            "ohlc_envelope_violations": int(bad_envelope.sum()),
            "ohlc_envelope_examples": examples(bad_envelope, frame),
            "zero_volume_rows": int((volume == 0).sum()),
            "negative_volume_rows": int((volume < 0).sum()),
            "volume_min": int(np.nanmin(volume)),
            "volume_max": int(np.nanmax(volume)),
            "price_min": float(np.nanmin(prices)),
            "price_max": float(np.nanmax(prices)),
        },
        "interpretation_limits": [
            "Parquet metadata declares the source, instrument, timestamp semantics and volume type; this audit cannot independently authenticate those claims.",
            "UTC weekend gaps may be normal market closures. Missing minute slots are counts between stored rows, not independently confirmed missing provider data.",
            "Prior model tuning or data exposure is unknown. Any later chronological test interval is only nominally held out until that history is established.",
        ],
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {DESTINATION}; {n:,} rows; SHA256 {report['input']['sha256']}")


if __name__ == "__main__":
    main()
