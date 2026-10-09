# Dynamic angular infilling with512+512 context

Initialize a bidirectional TCN from the completed chronological forecaster. Inputs
are only angular features plus availability/gap flags. Train H/C/L angles on hidden
positions; recursively reconstruct prices from the last visible left candle.
Open inside the gap comes from the previous predicted Close, never a hidden true bar.

Context:512 feature tokens left and512 right. Fibonacci lags require55 additional
raw history bars before the left context. The large uninterrupted context reduces
eligible source data; exact counts are saved in data_audit.json.

Curriculum: random gap lengths1–5, then1–10,1–20,1–40. All eligible hidden targets
are covered exactly once per epoch in chronological order; there is no training
subsample. Boundaries may produce a shorter last span. The maximum increases only
after at least3 stage epochs and two consecutive successful validation checks.
Gates require20% reductions in H/C/L and Close reconstructed-price MSE versus the
frozen causal rollout at the stage maximum, plus no worse angular MSE at every
evaluated length. If unsuccessful, stay at the same difficulty.

Train on the pre-last-year partition. First3 months of the reserved year are
validation, remaining9 months final test. Neither context crosses partitions.
This is research-exposed data, not globally untouched confirmation.

Settings: initialLR1e-4, dropout25%, AdamW weight_decay0.01, gradient_clip1,
30 full epochs, batch16 spans. Learning rate can decrease; it never increases.
Validation drives curriculum only; final epoch30 is used for final testing.

```powershell
python -m pytest experiments/angular_infill/test_model.py -q
python experiments/angular_infill/run.py train
python experiments/angular_infill/run.py train --resume
# Apply any saved checkpoint (default: latest completed epoch) to a hidden span:
python experiments/angular_infill/reconstruct.py --input data/M1/xauusd_m1.parquet --gap-start-row 10000 --length 5
```

Output: `reports/angular_infill_512_20261008/`.
Training finishes and evaluates automatically. No manual monitor is needed.
`resume.json` points to the latest atomic checkpoint. `progress.json` gives current
stage and coverage; `validation_epoch_*.json` gives reconstruction quality and gate
decisions; `training_curve.csv` gives all epochs. Final artifacts: `results.html`,
`metrics.csv`, `comparisons.csv`, `candle_metrics.csv`, `test_predictions.npz`,
`verification.json`, and `checkpoints/epoch_30.pt`.

Baselines: frozen old model rolled forward without future, ordinary linear bridge,
and a bridge using known right Open as a proxy for the final hidden Close. Removing
right input from the infiller is an OOD ablation, not a separately trained forecaster.
The decoder never uses actual hidden previous prices. This experiment measures
reconstruction with future context, not tradable forecasting accuracy.

`reconstruct.py` erases the requested hidden rows before feature generation and
requires567 preceding raw bars and512 subsequent bars, all consecutive by minute.
Hidden rows may contain placeholder OHLC or missing OHLC, but timestamps must exist.
It rejects lengths beyond the checkpoint's trained curriculum maximum. Returned
prices are not silently sorted or projected; `valid_ohlc` flags invalid geometry.
