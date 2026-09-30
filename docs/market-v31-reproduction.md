# Reproduce frozen v3.1 M1 research

Use a fresh checkout of commit `8c45bbf0cafc8397384dde87bd8c84aceedb7d62` and run commands from the repository root. It follows the preserved v3 result commit `45a4aac`. Keep the committed source, [config](../configs/market-v31.json), [protocol](market-v31-protocol.md), and [prefit budget](market-v31-budget.md) unchanged through fitting and scoring. Python 3.14.3 and the versions pinned in `requirements.txt` were used. The CPU fits each use one PyTorch thread; the pinned `torch==2.10.0+cu128` installation may require the matching PyTorch wheel index.

The Git-ignored file `data/M1/xauusd_m1.parquet` must have SHA-256 `a9dd281e0100047d56732540168af6e971ef23d858043e3e64991849a2fa6539`. The v3.1 config SHA-256 is `cab89fd80671f73e70e7dc660c71a6de4837391d6b132151b5ad0f1efb5169b6`. Preparation also requires the original v3 cache in `artifacts/market-v3`; if it is absent, regenerate it with `python scripts/prepare_market_v3.py --config configs/market-v3.json` before preparing v3.1. The v3.1 preparer checks the original v3 config, raw-data hash, and frozen train/evaluation cache hashes before writing any new split.

```powershell
python -m pytest -q
(Get-FileHash data/M1/xauusd_m1.parquet -Algorithm SHA256).Hash
python scripts/prepare_market_v31.py --config configs/market-v31.json
```

The new local caches are `artifacts/market-v31/{train,selection,calibration,evaluation}.npz` and `audit.json`; [data-preflight.json](../reports/v31/data-preflight.json) records population counts and provenance. Train and evaluation NPZ files are copied byte for byte from v3. The four expected cache SHA-256 values are train `58650707d3360c010f70cf3a6cdcc688ecba975253875237a58dbce2dbb6417c`, selection `20df9781f6c210dbcd272c06753214b9acf5a9175884775065949a66f74b7320`, calibration `5da2ffff520aca00158ea7142cfec1a905a5aae336b6048fb6ab686a2f2c0c7b`, and exposed evaluation `fa491f0e14a16976a64c610030166e33c7fed3e21ad35d0fe46e32466e90ea43`. Cache artifacts and checkpoints remain local and Git-ignored.

Start the fixed matrix of five models (`a`, `b`, `c`, `no_transfer`, `no_open`) and three seeds (`7`, `19`, `43`) with the launcher:

```powershell
python scripts/launch_market_v31.py --workers 9
```

The launcher writes a separate log under `artifacts/market-v31/logs/` and a separate `reports/v31/fit-<model>-<seed>.json` per fit. For a single fresh fit, use `python scripts/fit_market_v31.py --model c --seed 7`. On interruption, use `python scripts/launch_market_v31.py --workers 9 --resume`: jobs with `last.pt` restore model, optimizer, RNG, early-stop controller and curves at an epoch boundary; jobs that had not started begin fresh. The launcher reports any failures rather than declaring the matrix complete. A manual interrupted fit can use `python scripts/fit_market_v31.py --model c --seed 7 --resume`. Avoid starting a second copy of a fit while its first process is still running.

Each fit sees all 491,102 valid train examples and 294 optimizer updates per epoch. At least four and at most twenty epochs are permitted; five post-minimum nonimprovements make epoch 9 the earliest stop. Patience uses the fixed `0.00001` improvement threshold, while checkpoint selection uses the exact minimum selection loss. `short_best` selects the best of epochs 1–4; `full_best` selects the best over the completed run. Both are retained even when they point to the same weights. A `last.pt` file supports recovery and is never a scored checkpoint. The maximum epoch cap does not prove a plateau; use actual stopping curves to judge training adequacy.

After **all fifteen fit reports are complete and checkpoint hashes match**, run calibration on October–December 2025 only:

```powershell
python scripts/score_market_v31.py --phase calibrate
```

This fixes one positive-slope affine logit calibrator for each of the thirty selected checkpoint identities, then writes `reports/v31/calibrators.json` and `calibration-freeze.json`. The fixed slope is strictly positive, so rank order and raw-logit ROC-AUC are preserved; log loss, Brier score and calibration can change. The scoring code refuses to load calibration labels before the fifteen-fit barrier.

Only after the calibrator freeze is complete, run the exposed January–August 2026 scoring phase:

```powershell
python scripts/score_market_v31.py --phase evaluate
```

The evaluator checks source/data/cache provenance, all fit identities and selected checkpoint hashes, all thirty calibrator identities, the calibration-freeze hash, and calibration-logit hashes before loading evaluation labels. It reports both raw and calibrated results for `short_best` and `full_best`, each seed and month, plus the train-frequency constant, in `reports/v31/evaluation.json` and `metrics.csv`. Paired intervals resample whole UTC calendar weeks after averaging paired minute losses across seeds. January–August 2026 was examined in earlier research, so these results are exploratory and cannot serve as independent confirmation or a trading-profit claim.
