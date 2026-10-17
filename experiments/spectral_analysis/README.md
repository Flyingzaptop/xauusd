# Spectral research, 2021

Outcome report: `/workspace/xauusd/reports/spectral_analysis_20261012/REPORT_RU.md`.
The entire causal year dataset was read: 354340 rows × 2356 spectral components.
No images and no GPU use. Prepared HDF5 files remain unchanged.

Run sequentially with `/workspace/colab-tools/bin/python` from this directory:

1. `extract.py`: full-bin discovery/validation/test moments and derived features.
2. `analyze.py`: ridge diagnostics, activity, validation-selected rules/trades and causal shock detector.
3. `nonlinear.py`: histogram boosting; chronological validation early stopping.
4. `robustness.py`: non-overlap metrics, clustered uncertainty, same-entry controls and independent source checks.
5. `phase_controls.py`: magnitude-preserving random-sign sanity checks.
6. `descriptive_followup.py`: explicitly descriptive mechanisms and frozen strategies without future-gap filtering.
7. `candidate_fade.py`: explicitly POST-HOC exploratory fade candidate, no future-gap filter.
8. `write_report.py`: Russian report using saved results.

`events_only.py` is a convenience for recomputing event statistics after the technical cooldown-indexing correction. Final `analyze.py` already contains that correction.

Dependencies are listed in `requirements.txt`; exact installed versions in report `versions.json`.
Requires prepared year at `reports/causal_spectral_year_2021_20261012/` and original OHLC source `reports/pcgrad_mse_month_20261009/dataset.npz`. Source raw OHLC order is H,O,C,L.

Important distinctions:
- Phase/energy correlation is not independent evidence from thousands of overlapping samples.
- Normalized absolute-return targets divide by *past* volatility; their negative energy association must not be described as a smaller absolute price movement.
- Intraday descriptive metrics condition on no future gap. They are not live entry filters. Gap-inclusive trading results remove this ex-post selection.
- Shock cooldown is updated immediately after detecting the shock, independently of any future gap/label availability.
- Preliminary fade-after-extension candidate was noticed after inspecting results. All2021 is discovery for this candidate; another year is necessary for independent confirmation.
- Per-ounce round-trip cost scenarios are illustrative, not broker execution data.
- The saved source inputs/outputs support research reproducibility, not a claim of live profitability.

The share archive contains code, report, tables, labels, predictions and diagnostic models. It excludes the two large derived `.npy` matrices and the already-existing22.9GB HDF5 dataset. Those remain in the workspace and can be regenerated from the source data.
