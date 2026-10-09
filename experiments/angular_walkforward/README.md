# All-origin chronological angular TCN

Thirty full chronological passes over every eligible training target. No sample,
shuffle or early stop. Training: September2016–September2025. Last calendar year
of source reserved for final evaluation; do not use it to select epochs or LR.

Model input: only128 x36 angular values; raw support183 closed M1 bars. Output:
three next-step angles H,C,L. Open is supplied as previous Close outside the net.
Width64, dropout0.25, AdamW initial LR1e-4, weight_decay0.01. LR may decrease on
training-loss plateau. Final epoch30 is used for evaluation and inference.

Dense causal convolution predicts ALL points in each chronological chunk. Its
receptive field is exactly128, and independent-window equivalence is unit-tested.
Chunks never cross gaps. Gradients update after up to2048 targets, not each candle.

```powershell
python -m pytest experiments/angular_walkforward/test_model.py -q
python experiments/angular_walkforward/run.py train
# Resume from the latest fully completed epoch if interrupted:
python experiments/angular_walkforward/run.py train --resume
python experiments/angular_walkforward/verify.py
python experiments/angular_walkforward/predict.py --input data/M1/xauusd_m1.parquet
```

All outputs: `reports/angular_walkforward_20261008/`.
`progress.json` updates during epochs; `training_curve.csv` records exact target
counts, H/C/L losses, updates and learning rates. `train_years_epoch_*.csv` records
counts/losses by target year. `checkpoints/epoch_30.pt` is the final model;
`resume.json` identifies the latest atomic training checkpoint.

The model has no persistent state between chunks beyond its trained weights; its
per-prediction context remains128 feature tokens. Repeated training epochs are
not independent historical walk-forward trading returns. This dataset has been
researched before, so the last year is reserved for this run but not globally new.

Inference accepts CSV/Parquet with timestamp,high,open,close,low and at least183
consecutive closed minute bars. It returns raw angles/prices and valid_ohlc, without
silently projecting invalid candles. Forecasts from historical input are not live
market quotes. Baseline comparisons and all metrics are in results.html and CSVs.
