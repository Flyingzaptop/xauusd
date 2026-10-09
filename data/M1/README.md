# XAUUSD minute candles

The full local `xauusd_m1.parquet` is included in Git at the user's explicit request. MBO and L2 source datasets are excluded.

- Rows: 3,505,124.
- Time coverage: 2016-09-04 22:00 UTC through 2026-09-04 13:08 UTC.
- Schema: `timestamp` (UTC milliseconds), `open`, `high`, `low`, `close` (float64), `volume` (int64).
- SHA256: `a9dd281e0100047d56732540168af6e971ef23d858043e3e64991849a2fa6539`.
- Original venue/vendor definition is not established by the local file. These XAUUSD candles must not be described as identical to GC futures MBO data.

The original full-data audit is in `reports/angular_distillation_20261008/data_audit.json`. Historical experiments use different train/validation/test splits; consult each experiment's protocol. The newest MBO experiment trains only on the October 2025–May 2026 overlap, with June validation and July test. The repository still preserves the entire M1 source for reproducing earlier experiments.
