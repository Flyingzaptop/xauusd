# Reproduce the frozen M1 v3 research run

Run these commands from the repository root in a fresh checkout of commit `2c4da15e33bcd2c8527fae68230f325bc953499f`. Keep the checked-out source and `configs/market-v3.json` unchanged through fitting and evaluation. The frozen [protocol](market-v3-protocol.md), [data preflight](../reports/v3/data-preflight.json), [numeric preflight](../reports/v3/speed-stability.json), and [budget](../reports/v3/frozen-budget.json) describe the committed run.

The local, Git-ignored input must exist at `data/M1/xauusd_m1.parquet`. Its exact 46,110,894 bytes have SHA-256 `a9dd281e0100047d56732540168af6e971ef23d858043e3e64991849a2fa6539`. The frozen config SHA-256 is `1d81163949fe2b9634910d62e525efcfbce8cc71d1ac2f07feca493af5784f9b`. Python 3.14.3 and the pinned packages in `requirements.txt` were used; the `torch==2.10.0+cu128` wheel may require its matching PyTorch wheel index. The models run on CPU with one PyTorch thread per fit process.

```powershell
python -m pytest -q
(Get-FileHash data/M1/xauusd_m1.parquet -Algorithm SHA256).Hash
python scripts/prepare_market_v3.py --config configs/market-v3.json
```

Preparation writes `artifacts/market-v3/train.npz`, `validation.npz`, `evaluation.npz`, and `audit.json`, plus `reports/v3/data-preflight.json`. Use the committed `reports/v3/frozen-budget.json` for the fit; its timing preflight was measured before the source freeze and need not be rerun. The runner verifies the cache audit's config hash before fitting. The frozen budget records train/validation/evaluation cache hashes of `58650707d3360c010f70cf3a6cdcc688ecba975253875237a58dbce2dbb6417c`, `26ec7ff2f72eeba04cbb4742feb808f885bba5ef0895f956973b2271d1234c64`, and `fa491f0e14a16976a64c610030166e33c7fed3e21ad35d0fe46e32466e90ea43`. Cache bytes may depend on the pinned runtime, so compare hashes before claiming an exact byte-level reproduction. Data and model artifacts under `artifacts/` are local and excluded from Git.

The train split contains 522,437 observed candles across 462 contiguous segments, of which 491,102 have supervised labels. Validation has 177,888 candles and 167,691 scored labels; the exposed evaluation split has 234,102 candles and 221,343 scored labels. The first 64 observations of each segment warm state without loss, and the final 10 before any gap or split lack a target. All rows advance stream state. The longest train segment is 1,380 observations (23 hours); its median segment length is 1,319. Thresholds and both feature/metadata normalization parameters come from train only. The train median contiguous log-close move is `0.00011590241030035031`, yielding coarse-to-fine thresholds `[0.0009272192824028025, 0.00046360964120140125, 0.00023180482060070062]`.

Launch the fifteen independent fits with up to five one-thread workers:

```powershell
python scripts/launch_market_v3.py --workers 5
```

Each worker writes a separate `artifacts/market-v3/logs/<model>-<seed>.log`, selected checkpoint `<model>-<seed>-best.pt`, and `reports/v3/fit-<model>-<seed>.json`. The fixed matrix is models `a`, `b`, `c`, `no_transfer`, `no_open` crossed with seeds `7`, `19`, `43`. For an isolated single-fit reproduction, the corresponding command is:

```powershell
python scripts/run_market_v3.py --phase fit --seed 7 --model c
```

Each fit processes four complete epochs: 294 label-bearing train chunks and 491,102 scored examples per epoch, or 1,176 optimizer updates and 1,964,408 label visits per fit. Model state carries across 128-observation blocks; only the gradient graph is detached. Checkpoint selection uses validation log loss. Concurrent workers share the machine, so elapsed fit time is affected by CPU contention.

Only after every fit report and selected checkpoint exists, run:

```powershell
python scripts/run_market_v3.py --phase evaluate
```

The evaluation command checks matching source/config/data/cache provenance, the complete set of fifteen model/seed reports, and selected-checkpoint hashes **before loading evaluation labels**. It writes `reports/v3/evaluation.json`, `reports/v3/metrics.csv`, and a local `artifacts/market-v3/evaluation-predictions.npz`. January–August 2026 was used in earlier research, so this interval is explicitly **exposed exploratory evaluation**, not independent confirmation. The reported log loss, calibration and paired weekly intervals are predictive diagnostics; they do not establish trading profitability.

The final result commit adds postprocessing scripts. With completed outputs and
matplotlib 3.10.8 available, run `python scripts/plot_market_v3.py`,
`python scripts/report_market_v3.py` and `python scripts/verify_market_v3.py`.
The verifier recomputes metrics and intervals from saved predictions without
fitting or changing model settings. File-byte hashes can reflect checkout line
endings; exact byte-level claims require matching the recorded hashes as well
as the Git source identity. No market fitting is part of postprocessing.
