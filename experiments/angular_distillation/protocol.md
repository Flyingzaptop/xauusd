# Frozen protocol: angular OHLC reconstruction and privileged distillation

Protocol written before model fitting and held-out scoring, 2026-10-08.

## Question and acceptance criterion

Does a teacher which reconstructs an internal 20-minute gap from left/right
context improve a student forecasting that gap from past information only?
Primary comparison: distilled C versus direct A, with identical student inputs,
architecture, seed and supervised-update budget. Also compare against A_long
(twice the direct-training updates) and B (teacher weights, then causal fine-tuning).
Success requires lower held-out close-return MSE, stability across seeds/months,
and a paired day-block bootstrap 95% CI excluding zero. One bounded experiment
cannot prove a universal positive or negative result, nor trading profitability.

## Data and partitions

- Local `data/M1/xauusd_m1.parquet`, observed range 2016-09-04--2026-09-04 UTC.
- Train: before 2025-01-01; validation: calendar 2025.
- Research evaluation: January--August 2026. README identifies this interval
  as exposed to prior research, so it is NOT an untouched confirmation set.
- September 1--4, 2026: separately reported short tail outside the exposure
  documented in README; unknown other exposure, too short for confirmation.
- Full source audit, timestamp ordering, duplicate/nonfinite/OHLC checks.
- Use only uninterrupted one-minute windows. No filling weekends or missing bars.
- Every complete support window [-74,+40] belongs wholly to its split. This
  includes the teacher's right context and the longest lag's history.
- Uniform fixed sample up to 100,000 eligible train origins; validation up to
  12,000 origins. Research/tail evaluation every fifth eligible minute.
  This is a bounded training-window sample, not exhaustive optimization.

## Inputs, outputs and masks

- Left tokens: t-19..t; hidden target: t+1..t+20; teacher right: t+21..t+40.
- At each token: 4 anchored log OHLC values and 4 x 9 multi-lag values using
  lags [1,2,3,5,8,13,21,34,55], plus feature-level availability masks.
- Lag support extends actual historical access to t-74 for BOTH representations.
- Compare scaled log-displacements versus (2/pi)*atan(scaled displacement).
  Same lag grid and parameter count: this isolates the angular transform, not
  Fibonacci optimality. Add linear ridge and zero-return baselines.
- Robust scale S0 and exponent H fit only uninterrupted training close returns;
  fixed gain g=1. The same S(k)=S0*k^H is used by log and angular variants.
- No feature using any hidden raw candle may be visible. The student also hides
  all right-context values and features. Availability mask is explicit.
- Output: all 20 future OHLC log prices relative to C_t, divided by S0*sqrt(h).
  Loss computed in these fixed units, never in learnable angular target units.
  Prices are not forced to satisfy OHLC ordering; report violation rate.
- Teacher T learns the same targets with right context. Student C starts from
  the same initialization as A and mixes true targets and frozen teacher outputs:
  (1-lambda)*MSE(student,Y)+lambda*MSE(student,T), lambda in {0.25,0.75}.
- Teacher soft labels on training windows are in-sample, as in conventional
  distillation; no cross-fitted teacher. Validation/test targets never train weights.

## Fixed architecture and budget

- Flattened 60-token feature tensor -> MLP 128 -> 128 -> 80; GELU and LayerNorm,
  dropout 0.1 in hidden layers. Identical parameter count for all neural methods.
- AdamW, learning rate 0.0003, weight decay 0.01, batch 512, gradient clip 1.
- Seeds 17,29,43. Twelve complete passes over the fixed train sample for A,T,B,C.
- A_long continues A's optimizer trajectory for another twelve passes. Save best
  validation checkpoint, not the best test checkpoint. B starts from best T.
- Selection score: mean normalized close MSE at horizons 1,5,10,20 on validation.
  Distillation lambda selected using mean validation score across all three seeds,
  separately per representation. No test-based selection or search expansion.
- Random-walk negative control and AR(1) positive control: generated OHLC,
  independent innovations, angular representation, seed17, 12,000 train windows,
  six epochs (A_long twelve), same masking/model/training pipeline.

## Metrics and verification

- Close MSE and MAE in basis points, skill versus zero return, directional accuracy,
  prediction/return correlation at 1,5,10,20 minutes; all-OHLC reconstruction error.
- MSE on normalized outputs is a secondary scale-balanced summary.
- Direction metrics exclude exactly-zero realized return. No probability-calibration
  claims because the decoder estimates a conditional mean, not probabilities.
- Teacher compared with linear interpolation between known endpoints on same data.
- Pairwise C-A and C-A_long differences: seed-average per-window squared error,
  bootstrap resampling UTC days (2,000 replicates), plus monthly and seed tables.
- Fixed checkpoint inference, no refitting after inspecting research/tail outcomes.
- Mutation tests prove hidden values cannot change teacher inputs and any future
  values cannot change student inputs. A deliberately unsafe feature demonstrates
  the original masking-after-feature-computation leak.
- Record source/code/protocol hashes, versions, configuration, training curves,
  checkpoints, predictions, full metrics and a self-contained result report.

## Execution flow

Audit -> freeze support splits -> fit train-only scale -> mutation tests ->
synthetic controls -> all market training/validation -> freeze lambda selections ->
one held-out scoring phase -> block uncertainty -> report and artifact verification.
