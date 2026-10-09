# Reconstruction objective comparison

This implements the diagnostic-first approach after flat reconstruction was observed.
It preserves all prior runs and does not restart the stopped curriculum training.

- Diagnostic: eight fixed TRAIN windows, no dropout/weight decay, original angle MSE
  versus geometry loss. Memorization results are not held-out scores.
- A: angle head and original angle MSE (continued-training control).
- B: same head with differentiable decoded-price/shape/multiscale loss.
- C: pretrained encoders plus a zero-initialized residual head over the visible
  boundary log-price bridge. With zero residual it is exactly interpolation.
- Four full chronological fine-tuning passes per model, every eligible hidden
  training target, dynamic gaps1..40, context512+512, LR1e-4, dropout25%, WD0.01.
- Select checkpoints only on validation performance relative to linear interpolation.
  Final test compares original, A/B/C and both visible-boundary bridges.
- M3/M5/M10 are complete UTC-aligned bars wholly inside each hidden span. They are
  auxiliary training targets and aggregated reconstruction metrics, not independent
  coarse-timeframe forecasting models.

```powershell
python -m pytest experiments/angular_reconstruction_v2/test_core.py -q
python experiments/angular_reconstruction_v2/pipeline.py
# After the comparison finishes:
python experiments/angular_reconstruction_v2/reconstruct.py --input data/M1/xauusd_m1.parquet --gap-start-row 10000 --length 20 --variant best
```

The pipeline skips the diagnostic if its results already exist, runs training,
independent verification and generates an HTML report automatically.
Artifacts: `reports/angular_reconstruction_v2_20261008/`.
`diagnostic/results.json`, `models/{A,B,C}/curve.csv`, `selection.json`,
`metrics.csv`, `shape_metrics.csv`, `multiscale_metrics.csv`, `comparisons.csv`,
`verification.json`, `results.html`, and the fixed examples under `plots/`.
Failures are recorded in `pipeline_failure.json`.

Checkpoint files store only this run's models and do not overwrite the previous
epoch12 or forecaster. The final report uses the exact same five earlier examples.
No adversarial/variance reward or random oscillation is added. A stochastic
generative model is outside this controlled comparison; it would need separate
distributional evaluation and must not be judged by best-of-N agreement with truth.
