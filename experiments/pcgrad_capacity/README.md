# Capacity experiment recovered on 2026-10-09

This directory contains the width-scaled TCN and the sequential parameter-halving
experiment requested after the one-month MSE/PCGrad pilot. See `protocol.md` for
the frozen train/validation selection rule. This is two-sided reconstruction of
eight hidden M1 candles, not a future-price forecasting experiment.

The recovered dependencies are `../loss_energy_full/train.py`,
`../pcgrad_mse/train.py`, and `../pcgrad_mse/projection.py`. `reproduce.py` stages
their original import names automatically.

From the repository root, on a machine with PyTorch CUDA and NumPy:

```bash
python experiments/pcgrad_capacity/reproduce.py \
  --data reports/pcgrad_mse_month_20261009/dataset.npz \
  --out /path/to/fresh-output
```

The saved reference aggregates default to
`reports/pcgrad_mse_month_20261009/remote/run.json`. Reproduction uses a fresh
output directory and fresh initialization. It does not resume partial trials.
Last checkpoints include optimizer state and projection RNG state for recovery.

Training writes a checkpoint and JSON curve after every complete epoch.
The complete results from this run belong under
`reports/pcgrad_capacity_20261009/remote/`. Generate the CPU report there with:

```bash
python experiments/pcgrad_capacity/report.py
```

The report additionally requires matplotlib. The actual run records the torch
version and GPU in `remote/run.json`. No OAuth credentials belong in this
directory or in experiment archives.

The user requested the remaining smaller models after the original stopping
rule passed. Run that fixed extension with `reproduce.py --smaller`, otherwise
using the same arguments. It exhausts widths13/8/5/3/2. Results are saved under
`reports/pcgrad_capacity_smaller_20261009/remote/`; `merge.py` combines the two
completed result sets while retaining `run_initial.json` and `run_smaller.json`.
The finished study has40 fits at10 smaller widths, with12 epochs per fit.
