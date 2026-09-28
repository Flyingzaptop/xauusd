# XAUUSD M1 data audit — 2026-09-28

This is a read-only audit of `data/M1/xauusd_m1.parquet`. Reproduce the machine-readable result with `python scripts/audit_data.py` from the repository root. The script writes `reports/data-audit.json`; it does not modify the parquet file. The SHA-256 of the exact 46,110,894 input bytes is `a9dd281e0100047d56732540168af6e971ef23d858043e3e64991849a2fa6539`.

## Identity and coverage

The parquet file has 3,505,124 rows in four row groups. Its fields are `timestamp: timestamp[ms, tz=UTC]`, `open/high/low/close: double`, and `volume: int64`. The first and last rows are 2016-09-04 22:00 UTC and 2026-09-04 13:08 UTC; these are also the observed minimum and maximum timestamps. The 2026 portion ends on September 4, even though this audit was run on September 28.

Embedded parquet metadata declares `source=cTrader Open API`, `symbol=XAUUSD`, `symbol_id=41`, `timeframe=M1`, `timestamp_semantics=UTC candle open time`, `volume_type=tick volume`, `price_digits=2`, and `environment=demo`. These are file metadata claims. This audit cannot independently verify the provider, that timestamps are candle *opens*, or that volume is tick count rather than traded quantity. In particular, do not interpret `volume` as ounces or lots.

## Structural checks

All six columns have zero nulls. Timestamps are strictly increasing, unique, and aligned to exact UTC minutes: zero duplicate timestamps, zero descending or equal adjacent pairs, and zero non-minute-aligned values. All prices are finite and positive. Every row satisfies `low <= open, close <= high` and `low <= high`; there are zero OHLC envelope violations. Volume is an integer from 1 to 961 with zero null, zero, or negative rows. These checks establish internal consistency, not economic accuracy.

There are 3,498,577 adjacent one-minute intervals and 6,546 longer intervals. The latter span 1,753,225 minute slots in total. That slot count includes scheduled market closures and holidays, so it is **not** a count of confirmed missing trading candles. The largest observed interval is 5,263 minutes, from 2022-04-14 20:59 UTC to 2022-04-18 12:42 UTC. Other leading intervals cross year-end and Easter weekends. There are 50,297 rows timestamped Sunday UTC and none on Saturday UTC; this is consistent with a market reopening on Sunday UTC, though the audit does not validate the provider's session calendar. See the JSON report for yearly counts, interval frequencies and exact largest-gap endpoints.

## Reproducibility and research boundary

The audit ran with Python 3.14.3, NumPy 2.4.3, pandas 3.0.1, PyArrow 23.0.1, SciPy 1.17.1, scikit-learn 1.8.0, PyTorch 2.10.0+cu128 and psutil 7.2.2. The machine reports 16 logical CPUs, 42,569,789,440 bytes of RAM, and an NVIDIA GeForce RTX 3060 Laptop GPU with 6,441,926,656 bytes of memory. `requirements.txt` pins the observed package versions; the CUDA-tagged PyTorch build may require the corresponding PyTorch wheel index on a fresh system. The audit script itself chiefly uses NumPy, pandas and PyArrow; compute information is supplemental.

The bounded experiment in `configs/experiment.json` uses train candles from 2024-01-01 to 2025-07-01, validation from 2025-07-01 to 2026-01-01, and test from 2026-01-01 to 2026-09-01, with each upper bound exclusive. Its target and full input window must remain inside one split, and fitted transforms use train data only. Do not inspect test labels or outcomes while selecting rules. Earlier tuning or exposure to this file is unknown, so the test portion is only **nominally held out**, not guaranteed independent. A final independence claim requires confirming previous data and strategy exposure or acquiring genuinely untouched later data.
