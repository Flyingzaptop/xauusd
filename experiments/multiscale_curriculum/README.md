# Multiscale TCN forecast over 60–120 minutes

Read [protocol.md](protocol.md) for the complete predeclared experiment. The model sees only completed historical bars at three scales. It predicts eight future 15-minute summaries (mean Close, minimum Low, maximum High), each with marginal q10/q50/q90 forecasts. This does not reproduce individual M1 candles.

`prepare.py` creates the regular calendar grids, past-only dollar scales, and chronological splits from the original M1 NPZ. `check_data.py` independently verifies targets and completed aggregates against raw OHLC. `model.py` implements three six-block TCN towers with dropout 25%, plus a matched static or rank-four hypernetwork head. `objective.py` defines four curriculum tasks. `train.py` runs both architectures with curriculum and direct training, two seeds each, with plateau stopping and full chronological epochs.

Prepare and audit data on CPU (Python + NumPy):

```bash
python experiments/multiscale_curriculum/prepare.py --out reports/multiscale_curriculum_20261010/dataset.npz
python experiments/multiscale_curriculum/check_data.py --data reports/multiscale_curriculum_20261010/dataset.npz
```

Run on CUDA with PyTorch and NumPy, using an empty output folder:

```bash
python experiments/multiscale_curriculum/reproduce.py --out reports/multiscale_reproduction/remote
```

The original run used PyTorch 2.11.0+cu130 and RTX PRO 6000 Blackwell. Kernel/hardware differences can affect checkpoint selection. Best weights for each stage and final model are saved. Optimizer states are used for stage transfer in-process; archived stage checkpoints retain model weights, not optimizer state.

Audit downloaded results and create CPU plots (NumPy + Matplotlib):

```bash
python experiments/multiscale_curriculum/report.py
```

For another output folder, place the prepared dataset.npz and dataset.json beside `remote`, then use `report.py --out path/to/folder`. The wrapper `reproduce.py` writes worker_result.json after a successful exit. No test labels are used to select weights or the model family. The final report contains descriptive day-bootstrap intervals, operating limitations, and evenly spaced test forecasts rather than manually selected successes.
