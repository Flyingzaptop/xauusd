# Transformer — stage two only: future activity

This experiment uses the existing best stage-one static/hyper checkpoints, without extending stage one or training from scratch. It predicts future 60/120-minute High-Low range and total absolute minute Close movement, with positive ordered q10/q50/q90 estimates. Neither output predicts direction or net price change. Read [protocol.md](protocol.md) for the fixed settings.

Inputs are the unchanged angular H/O/C/L features and availability flags from stage one. No absolute price level or future token enters forward. Targets are relative to current Close, normalized by fixed TRAIN medians. Current Close is used outside the model only to display dollar magnitudes.

CPU preparation and actual future-price mutation checks:

```bash
python experiments/transformer_stage2/prepare.py --out reports/transformer_stage2_20261012/dataset.npz
python experiments/transformer_stage2/check_data.py --data reports/transformer_stage2_20261012/dataset.npz
```

On CUDA with PyTorch and NumPy, using an empty output directory:

```bash
python experiments/transformer_stage2/reproduce.py --out reports/stage2_reproduction/remote
```

All legacy weights load exactly. Warmup freezes them for three complete epochs; then they fine-tune at one tenth of the new heads' learning rate. The old reconstruction readout remains frozen and is measured before and after the new training, without adding stage-one loss. PyTorch2.11.0+cu130 / RTX PRO6000 Blackwell was used; hardware/kernel differences may change selected epochs.

CPU audit and figures (NumPy + Matplotlib):

```bash
python experiments/transformer_stage2/report.py
```

For reproduction, put the prepared dataset.npz and dataset.json alongside `remote` and pass that parent to report.py --out. Original stage-one datasets/predictions and checkpoints are included in the archive for transfer and retention audits. `reproduce.py` writes worker_result.json only after trainer success.

Both simple baselines use TRAIN fitting only. The past-activity baseline derives its conditioning proxy solely from observed Close vectors, not raw price levels or future labels. The report includes reused-period limitations, overlapping-window dependence and descriptive whole-day bootstrap comparisons. There is no stage-three code or automatic curriculum transition. No credentials are included.

Interactive standalone review of all554 forecasts:

```bash
python experiments/transformer_stage2/review.py
```

Open reports/transformer_stage2_20261012/review.html. No remote scripts or API calls.
