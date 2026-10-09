# Four-angle next-candle TCN

The network receives ONLY normalized historical angles: 128 timestamps x 9 lags x
4 channels. Its output is exactly four next-candle angles in H,O,C,L order,
relative to the last observed candle. Last raw OHLC is used only by the decoder.
No teacher or distillation is involved.

Architecture: causal dilated residual TCN, 32 channels, 6 blocks x 2 convolutions,
kernel3, dilations1,2,4,8,16,32; 39,332 parameters. Channel normalization is per
timestamp and does not look ahead. Training criterion is angular MSE.

From the workspace root:

```powershell
python -m pytest experiments/angular_tcn/test_model.py -q
python experiments/angular_tcn/run.py train
python experiments/angular_tcn/verify.py
python experiments/angular_tcn/predict.py --input data/M1/xauusd_m1.parquet --output reports/angular_tcn_20261008/forecast.json
```

Input to `predict.py` may be CSV or Parquet with columns `timestamp,high,open,close,low`.
The last 183 rows must be consecutive, sorted, valid, closed M1 bars. Time is UTC.
Only that recent history is encoded for inference; the network never needs a future
bar. The saved three-seed ensemble averages angles before decoding prices.

Outputs and saved weights: `reports/angular_tcn_20261008/`.
`models/{17,29,43}/best.pt` includes weights, scale, channel order and architecture
metadata. `results.html` contains per-channel angle/price errors, candle-shape errors,
baseline comparisons and visualizations. `metrics.csv`, `candle_metrics.csv` and
`comparisons.csv` hold full scores. `predictions/` holds exact evaluation predictions.

The model does not force independent predicted OHLC points into valid candle order.
`valid_ohlc` is returned and violations are included in evaluation. Seed disagreement
is not a calibrated uncertainty interval. A supplied historical file produces a
historical forecast, not a live quote. Research evaluation2026 is already exposed.

See `protocol.md` for the fixed experiment and `independent_verification.json` in
the report folder for separate artifact and inference checks.
