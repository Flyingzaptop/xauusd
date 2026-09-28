# Frozen protocol: Crossing Memory v1

This engineering experiment is specified before fitting or inspecting test predictions. The source data may have been used in earlier work outside this repository; exposure is unknown. The test is chronologically held out in this run, not a guaranteed previously untouched confirmation sample.

## Samples and target

Only M1 OHLC is used; volume is omitted because its economic meaning is undocumented. Embedded metadata declares UTC candle-open timestamps; forecasts occur after the associated candle completes, one minute after its timestamp. Provider semantics are not independently verified.

Each independent example contains exactly 64 consecutive minute candles ending at t. Predict whether close(t+10) > close(t), and the corresponding log return. Zero returns map to class 0 (not-up). Require the entire input through target to be consecutive one-minute timestamps. Never bridge sessions, gaps or weekends. Train starts 2024-01-01; validation 2025-07-01; test 2026-01-01; end exclusive 2026-09-01. A target must precede the next split boundary. All input timestamps must also be inside the same split, conservatively foregoing cross-boundary warmup.

Candidate end indices have fixed stride 10 source rows. Apply deterministic evenly spaced subsampling within each split up to 4096/2048/4096 examples, without using values or labels for selection. This is a bounded feasibility experiment, not an exhaustive use of available data. Monthly results and weekly paired block bootstrap account for temporal dependence; individual-minute bootstrap is prohibited.

## Features and state

OHLC features are log close change, log(close/open), log(high/close), log(low/close). The first change in every window is zero. Standardization uses training windows only, with clipping to [-10,10]. Return targets are divided by train return standard deviation for the auxiliary squared-error loss; predictions and MAE are converted back to log-return units. Fixed nested thresholds are train median absolute consecutive one-minute log return times [2,4,8]. Thresholds never adapt during an open event. Routing starts fresh at the first close of every window for all models. No model receives older history or hidden states across windows/batches/runs.

A: small GRU on four candle features. B: same GRU plus six causal statistics per scale. C: same GRU inputs as B plus learned Crossing Memory readout. Ablations remove interscale transfer or open-event readout. Thus C versus B isolates learned operator contribution while retaining the same routed representation. Parameter counts are explicitly reported; optimization updates, history, data and seed budgets are equal, parameter counts need not be identical. Constant control predicts train up frequency and train mean log return.

## Training and decision rules

Exactly one fixed configuration, three seeds 17/29/43, six epochs each, Adam 0.001, batch 256, gradient clipping 1.0. Optimize binary cross entropy plus 0.1 normalized-return mean squared error. Select the epoch with lowest validation log loss; no test-based tuning. CPU with four PyTorch threads provides reproducible small-run execution. Deterministic algorithms are enabled. No architecture selection or additional search after viewing test.

Primary metric: test log loss. Secondary: ROC-AUC, Brier, ten-bin equal-width ECE with calibration tables, return MAE. Report validation and monthly test metrics, elapsed train time, parameters, sample counts and seeds. Paired bootstrap resamples whole UTC calendar weeks of test observations, aggregating mean per-observation loss across seeds before resampling; 95% percentile intervals describe time-block variability, not independent replications or trading performance. Report C-B and C-A; negative differences favor C. No claim of profitability is supported.

Code/config/data SHA provenance is stored with results. Mechanism and synthetic checks precede market training; corrections required by failed premarket checks will be documented before freezing the executable code. Test is evaluated only after all market fits complete and best-validation checkpoints have been saved locally.


