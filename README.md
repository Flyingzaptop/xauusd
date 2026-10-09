# XAUUSD angular sequence research

Progress snapshot: 2026-10-09. Source code, experiment protocols, metrics, plots, compact results are preserved in this repository. The **complete M1 parquet is included**; raw MBO/L2 datasets and MBO event caches are excluded.

- [Full progress and limitations](docs/progress-20261009.md)
- [M1 dataset](data/M1/README.md)
- [Latest causal M1 + asynchronous MBO protocol](experiments/angular_mbo/README.md)
- [Latest pipeline state](reports/angular_mbo_causal_20261008/pipeline.json) and [verification](reports/angular_mbo_causal_20261008/verification.json)
- [Controlled reconstruction report](reports/angular_reconstruction_v2_20261008/results.html) and [metrics](reports/angular_reconstruction_v2_20261008/metrics.csv)
- [Bidirectional infilling](experiments/angular_infill/README.md) and [five reconstruction examples](reports/angular_infill_512_20261008/examples_checkpoint_12/index.html)
- [Full chronological angular training](experiments/angular_walkforward/README.md)
- [Initial angular TCN](experiments/angular_tcn/README.md)
- [Distillation protocol](experiments/angular_distillation/protocol.md)
- [Publication inventory](reports/github_publication_20261009.json)

The MBO experiment is stopped by the user in this snapshot; do not resume automatically. Smoke metrics and partial training curves do not establish held-out accuracy. Completed reconstruction experiments still produced overly smooth paths and did not establish an advantage over the strong interpolation baseline. Historical evaluation periods have been researched; these results are not a trading-profit claim.

## Reproduce

Use Python with the dependencies in `requirements.txt`. CUDA Torch may require the matching CUDA wheel index. Each experiment's README/protocol contains its commands and data split.

```powershell
python -m pytest experiments/angular_mbo/test_causality.py -q
python -u experiments/angular_mbo/pipeline.py
```

The second command requires local Databento MBO archives under `data/MBO/`, which are intentionally absent from Git. M1-only experiments can use the committed `data/M1/xauusd_m1.parquet`. Checkpoint weights and `.npz` array dumps remain local and are excluded from Git; inference scripts using them require independently trained/local artifacts.

[Legacy workspace README](docs/legacy-workspace-readme.md) is retained as historical text; it contains references to earlier branches/reports that were not present in this local research checkout.
