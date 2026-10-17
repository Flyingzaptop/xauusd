# Every-candle causal daily inspection

```bash
pip install -r experiments/local_spectral/requirements.txt
python experiments/local_spectral/causal_day.py \
  --source reports/global_fourier_month_20261012/predictions.npz \
  --out reports/causal_day_reproduction --day 2021-06-16
python experiments/local_spectral/causal_review.py \
  --out reports/causal_day_reproduction
```

Every real observed candle, no time-grid insertion. STFT windows end at the
selected closed candle. Wavelets are an explicitly one-sided Morlet bank with
zero-DC/L2-normalized kernels, distinct from standard centered CWT. Color
references are fitted strictly before the day. All coefficient columns are
saved at step 1, with no thinning. The viewer hides candles after the selected
point and supports one-candle stepping.

Prefix recomputation, future-price mutation and independent direct sums audit
causality. Synthetic fixed-period input audits the nominal period axis.
Minute timestamps are treated as candle start; available_timestamp_minutes is
one minute later, after that candle closes. No predictions or trading strategy
are fitted. See the generated REPORT.md and checks.json.
