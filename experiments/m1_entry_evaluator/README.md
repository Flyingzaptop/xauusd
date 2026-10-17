# Exact retrospective XAUUSD M1 entry evaluator

This stage implements the user-approved evaluator of consequences before any neural critic or actor. No scalar quality score, arbitrary reward weights, RL learning, or predictive claim. Two sets of measurements per entry, LONG and SHORT. A future-informed neural critic would approximate quantities the historical simulator can compute exactly; that is a separate stage.

Inputs: the full user-supplied XAUUSD_41.parquet, 3,505,124 M1 OHLC bars. Timestamp is UTC candle OPEN. At timestamp t, entry reference is Open[t]. Past observation spans [t-120m,t), future observation spans [t,t+120m), ending with the Close of the bar opened at t+119m. Thus each complete point uses two calendar hours before and after the entry, with no added future horizon. No absolute price is fed to a neural network: no network is used in this stage.

Each actual entry bar is emitted. Future outcomes use the continuous observed prefix up to 120 minutes, a missing candle, the source end, or UTC midnight, whichever comes first. This is a retrospective coverage convention, NOT an executable decision to avoid future gaps. Partial windows are explicitly flagged and excluded from comparisons requiring 120 minutes. No interpolation, forward fill, weekend compression, or assumptions about missing prices. UTC is an explicit provisional trading-day convention; broker/account day and real session-close execution require a later specification.

Past scale s = sqrt(120 * mean(delta_close^2)) over the 119 close-to-close changes within the prior 120 bars. It uses no future prices and is invalid when the past is incomplete or flat. This is an RMS-movement unit, not a fitted forecast or monetary risk budget. Raw-dollar outcomes are retained. Existing angular-vector model inputs are not replaced; this exact calculator does not construct model inputs.

Fields per side:
- Mean, 10th/50th/90th percentile marked-to-close PNL over all observed future minutes. These describe a realised path, not predictive probabilities or expected optimal execution.
- Maximum favourable/adverse excursion from entry, using future High/Low; adverse is nonnegative. Minute-high/low ordering is unknown.
- Fraction of positive closes and fraction of candles wholly in profit according to OHLC. Neither is an estimate of the exact continuous-time proportion in profit.
- Average negative close PNL and maximum drawdown on the close-sampled PNL curve including entry value zero.
- Close PNL at 5,15,30,60,120 elapsed minutes (NaN if unavailable).
- First minute bin touching profit/loss thresholds 0.25/0.5/1/2 times the past scale. Independent hit times do not resolve target/stop ordering in the same bar; NaN means not observed within coverage or missing scale.

All measures are gross quote-price consequences. No bid/ask is supplied. A fixed roundtrip cost c shifts mean/quantile/endpoint PNL by -c; profitability fractions and hit times need recomputation. Excursions do not ensure fillability at extremal prices. Capital, leverage, lot size, position limits, spread, slippage and overnight funding are unspecified; daily percent return cannot be inferred from this dataset.

Daily opportunity analysis is retrospective and descriptive, using only full 120-minute intraday windows with valid past scale. The 3x3 risk/profit grid is a sensitivity table, NOT a trained score or prescribed thresholds. A qualifying entry can be LONG or SHORT. Counts of greedily selected non-overlapping 120-minute windows stop neighbouring minute entries being treated as independent opportunities. These are not simulated trades or realised daily profits.

Pareto representatives: minimise max adverse/s, maximise mean PNL/s and profitable-close fraction, separately per side across all eligible historical points. No objective weights. Exact duplicate objective vectors retain one representative. The comparator is this explicitly chosen set of three objectives, not a universal definition of trading quality. The global frontier can be sparse and concentrated in unusual days. This is not evidence that other days lack tradable opportunities. All-years frontier/grid are exploratory oracle descriptions, not an independent final strategy test. Later actor training must split and purge future context chronologically.

Run with Python + numpy/pandas/pyarrow:
  python checks.py
  python run.py
  python analyze.py
  python report.py
  python evaluate_at.py 2025-06-16T10:00:00Z --points /path/to/points

run.py paths point to the existing workspace layout; edit BASE/SRC/OUT for another machine. analysis_inputs.parquet is a compact local intermediate and need not be distributed: it can be reconstructed from selected columns of the annual partitions. Code, versions, input/output SHA256, checks and column schema accompany the results. No cloud GPU is needed.
