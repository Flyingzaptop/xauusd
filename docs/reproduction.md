# Reproduce Crossing Memory v1

Repository: https://github.com/Flyingzaptop/xauusd ; branch: `crossing-memory-v1`.
Frozen executable source: `a8d7f4c88e92dd3171c4173d01562d52e5d5a682`.

Use Python 3.14.3 and the versions in `requirements.txt`. The recorded Torch build
is `2.10.0+cu128`; obtaining that exact CUDA wheel may require the PyTorch CUDA
12.8 package index. The experiment itself runs on CPU with four threads; the
synthetic benchmark runs with one thread. No network services are used in fitting.

The original dataset is not distributed. The owner must supply
`data/M1/xauusd_m1.parquet` (46,110,894 bytes), SHA-256
`a9dd281e0100047d56732540168af6e971ef23d858043e3e64991849a2fa6539`.
Its embedded metadata attributes it to cTrader Open API demo, XAUUSD M1.
No public download or exact export procedure was provided; independently
obtaining identical bytes is therefore a reproducibility limitation. Do not
substitute a freshly exported file without treating it as a different data version.

Run from the repository root with the frozen commit checked out:

```powershell
python -m pip install -r requirements.txt
python scripts/audit_data.py
python -m pytest -q
python scripts/synthetic_checks.py
python scripts/run_experiment.py --phase fit
python scripts/run_experiment.py --phase evaluate
```

The final published branch adds reports and documentation after the frozen code
commit. To exactly reproduce the recorded source identity, first check out the
frozen SHA in a separate checkout with the dataset available at the stated path.
Avoid source changes between fit and evaluate. Evaluation verifies equal commit,
configuration SHA-256 and dataset SHA-256. Git commit identity alone does not
detect uncommitted edits; keep the source tree clean. Audit regeneration includes
machine-specific resource information.

Local artifacts are under
`C:/Users/r3d_flzp/Desktop/xauusd/artifacts/crossing-memory-v1/`:
15 selected `.pt` state dictionaries, `fit.json`, and `test_predictions.npz`.
They are excluded from Git and can be regenerated with the commands above.
`reports/artifact-manifest.json` records their sizes and SHA-256 checksums.
Published machine-readable market results include per-seed validation, test and
monthly metrics plus calibration bins and paired weekly-block intervals. The
synthetic report includes generator configuration and hashes of generated data.

On the final report commit, `python scripts/summarize_results.py` regenerates
the acceptance Markdown and artifact manifest from completed outputs;
`python scripts/routing_diagnostics.py` regenerates descriptive train-only
routing counts. These post-run analysis scripts were added after test evaluation
and were not used for model selection.

Changing the architecture after these reported test outcomes makes any reuse
of this test exploratory. A future confirmation run needs an independently
untouched period and a new preregistered protocol.
