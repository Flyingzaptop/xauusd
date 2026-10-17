# First joint-market LightGBM study

Scope: predict XAUUSD future changes from all222user-providedM1instruments, compared with gold-only context. No Transformer, classifier, or RL in this stage. All source files are read-only. See PROTOCOL.md for fixed targets, cuts and model-selection budget.

Source data: Drive folder1BHeZ-romQ4oIZ970lxZCEYDKpyZ8QFMd. The full222-file manifest is in reports/intermarket_lightgbm_20261015/drive_manifest.json. Each Parquet includes UTC minute-open semantics and source metadata; availability is timestamp+60seconds. Original files are public/shared by the user and downloaded directly into the runtime. Content hashes are recorded in audit.json.

- features.py:39causalrelative features per instrument, explicit quote age/missingness, calendar rather than compressed time. Current and lagged return/volatility/position/volume/candle geometry.
- prepare.py:downloads allfiles, builds a float32 feature-major memmap at5minute gold anchors, resumable per instrument, three bounded feature workers. Uses all underlyingM1bars.
- check_features.py:synthetic future-mutation, prefix and explicit-gap checks.
- inventory_remote.py:checks every source file's schema/metadata and rowcount.
- train.py:gold-only47inputs versus all-market8666inputs; twoLightGBMconfigurations, horizon-specific boosting counts, frozen final predictions. Seed177. EightCPUthreads initially. Dataset reuse across horizons bounds memory.
- evaluate.py:forecast metrics, monthly stability, seven-day block confidence intervals, development-selected trading policy with delayed entry and hypothetical roundtrip costs. Validation trades extending beyond the evaluation boundary are censored from policy selection; no final-period price enters validation P&L.
- patterns.py:illustrative cross-instrument tree paths selected from trained model structure and training counts, not final-test labels. Rules are not independently validated strategies and gain shares are not causal importance.
- report.py:Russian report plus uniformly spaced forecast examples, no cherry-picking.

Reproduction: provision a Colab high-RAM CPU runtime (the study used8vCPU and~50GiBRAM), at least60GBfree disk. Install pandas, numpy, pyarrow, lightgbm, requests and psutil. Copy features.py/prepare.py/train.py and drive_manifest.json under /content/intermarket; run prepare.py, inventory_remote.py, then train.py. Retrieve results/ plus prepared.json/audit.json/inventory.json. Put those under the report's remote directory for local evaluate.py and report.py. Run patterns.py on the runtime before deleting its feature matrix. Paths are intentionally explicit to make the recorded run reproducible.

The~24GB feature matrix and5.42GB sourceParquet are not duplicated into the compact result archive; regenerate from included code/manifest and verify source hashes. Runtime/checkpoint/software metadata and predictions are saved. Original large spectral-year files are unrelated and unchanged.

Limitations: surviving present-day instrument universe, demo indicative prices without realbid/ask or execution timestamps, onefeaturedesign and oneseed. Target normalization requires enough past60calendar-minute gold observations, so early minutes after a long gap may be excluded. Pure interactions without marginal split gains can be missed by greedy trees. Failure here does not establish absence of predictive information.

Completed 2026-10-15. See REPORT_RU.md and source_verification.json: archived runtime training-script hash does not match the provided local train.py; exact runtime source was not retained. Model outputs and snapshot checksums are verified. No Transformer/RL stage was launched.
