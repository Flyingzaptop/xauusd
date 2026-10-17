# Local spectral inspection

Consecutive observed XAUUSD candles, June 2021 and July 1–7. No filling of
market closures. This is descriptive analysis, not forecast evaluation.

```bash
pip install -r experiments/local_spectral/requirements.txt
python experiments/local_spectral/run.py \
  --source reports/global_fourier_month_20261012/predictions.npz \
  --out reports/local_spectral_reproduction
python experiments/local_spectral/review.py \
  --out reports/local_spectral_reproduction
```

STFT uses trailing windows. Standard centered complex Morlet CWT uses future
neighbors; the figures explicitly label this, and a second display delays
coefficients until full finite-kernel availability. Both are audited by future
mutation, and delayed CWT is checked against prefix recomputation. A known
64-candle sinusoid validates period axes. Phase statistics use non-overlapping
windows to avoid interpreting overlap as persistence.

49 shuffled-return and 49 independent-sign surrogates provide descriptive
controls; no trading significance or predictive advantage is inferred from
their intervals. Price, coefficient arrays, source hashes and plotting code are
archived. CWT is calculated at every candle and saved every 32 candles for
inspection. GPU is not needed.
