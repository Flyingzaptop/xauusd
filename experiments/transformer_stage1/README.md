# Transformer, stage one only: read and aggregate visible history

The experiment implements the user's first curriculum stage and stops. It does not forecast future candles or run any other stage. See [protocol.md](protocol.md) for the fixed settings.

`representation.py` implements the repository's angular H/O/C/L log-displacement family. Raw displacement vectors add; normalized atan angles do not. Path reconstruction preserves coarse High/Low extrema, first Open and last Close. `prepare.py` fits scales on January TRAIN, makes angular multiscale inputs, and creates historical 15-minute targets. `check_data.py` independently checks original OHLC, vector reversibility, completed inputs, and an analytic oracle that obtains target vectors from input vectors alone without price anchors.

`model.py`: four-layer Transformer encoder, width96, four heads, dropout25%, three timeframe embeddings, eight learned historical summary queries. It receives only angular vectors and observation fractions. Compare static and rank-four hypernetwork heads at nearly identical capacity. `loss.py` uses Huber on cumulative decoded displacement plus0.1 angular MSE. `train.py` performs four fits with two seeds, selects epochs and architecture on validation, then evaluates stage one on test and stops for user review. No teacher forcing; no sorting of predicted OHLC hides invalid geometry.

CPU data preparation and independent checks:

```bash
python experiments/transformer_stage1/prepare.py --out reports/transformer_stage1_20261012/dataset.npz
python experiments/transformer_stage1/check_data.py --data reports/transformer_stage1_20261012/dataset.npz
```

On a CUDA host with PyTorch and NumPy:

```bash
python experiments/transformer_stage1/reproduce.py --out reports/stage1_reproduction/remote
```

Use an empty output directory. Each epoch traverses every eligible January training origin chronologically once. Best and last checkpoints are saved; last includes optimizer and RNG state. The original run used PyTorch2.11.0+cu130 and NVIDIA RTX PRO6000 Blackwell. Small numerical/kernel differences may change selected epochs.

CPU audit and plots (NumPy + Matplotlib):

```bash
python experiments/transformer_stage1/report.py
```

For a reproduced output directory, put dataset.npz and dataset.json beside `remote`, then use `report.py --out reports/stage1_reproduction`. Stage-one images show already-visible past, not future predictions. The exact analytic aggregation is a reference proving that the answer is present in the input, not a competing learned model or forecasting baseline.

No credentials are included. Stage2 requires the user's review of stage1 and a separate instruction.

Interactive offline review of every test sample:

```bash
python experiments/transformer_stage1/review.py
```

Open reports/transformer_stage1_20261012/review.html. It is a standalone file without remote scripts or API calls.
