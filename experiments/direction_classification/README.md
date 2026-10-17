# Past-only direction classification

See `protocol.md` for the exact target, month split and loss. Runtime dependencies:
PyTorch with CUDA and NumPy. `report.py` additionally requires matplotlib.

Run from the repository root on a CUDA machine:

```bash
python experiments/direction_classification/reproduce.py --out /path/to/fresh-output
```

The default dataset is the exact recovered month-pilot NPZ. It contains the raw
ten-year series and the month-specific index arrays; only those indices train
or evaluate the model. The old reconstruction scales are unused.

The completed run resides in `reports/direction_classification_20261010/remote`.
To regenerate plots, independent metric checks and the report:

```bash
python experiments/direction_classification/report.py
```

SHORT/NONE/LONG logits and labels are preserved for each test example. Both
best-by-validation and last checkpoints are saved; the latter include Adam
state and the PCGrad projection RNG state. All four fits start from scratch.
