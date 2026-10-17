# Precision-first classification

`protocol.md` specifies the frozen objective, validation partitions, operating
rules and test procedure. Target and data remain the confirmed eight-minute,
strict +/-$2 classification problem with only past512 candles.

On a CUDA machine with PyTorch and NumPy, from the repository root:

```bash
python experiments/direction_precision/reproduce.py --out /path/to/fresh-output
```

CPU report, requiring NumPy and matplotlib:

```bash
python experiments/direction_precision/report.py
```

The report expects artifacts under `reports/direction_precision_20261010/remote`.
Each treatment averages the two seeds' scores using checkpoints selected on
the first validation partition. Policies fitted on the second partition are
in `run.json`. `policy.py` applies exactly those policies to model scores.
Raw scores from the asymmetric loss are not guaranteed calibrated probabilities.

Checkpoints, optimizer states, curves and per-example validation/test scores
are retained. No minimum test signal quota is imposed: zero-signal precision
is undefined and cannot be reported as success.
