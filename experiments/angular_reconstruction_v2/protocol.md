# Reconstruction diagnosis and objective comparison, 2026-10-08

User authorized implementing the best next approach after flat infilling results.
Keep the stopped experiment unchanged. Do not restart its long run.

## Phase1: capacity/objective diagnostic

Use eight fixed TRAIN windows, gaps1,5,10,20 (two each), chosen by fixed quantiles.
Start from saved infiller epoch12. Compare angular MSE and geometry loss separately
on the exact same windows. Dropout0, weight_decay0 ONLY for this memorization test.
AdamW LR1e-4, gradient clip1, at most1200 updates per variant, same initialization.
Report loss curves, per-candle outputs, angular error, decoded-price error and
total variation. Memorization is NOT held-out evidence. The diagnostic budget
does not prove impossibility if near-exact fit is not reached.

## Phase2: controlled full-data fine-tuning

Three variants, all from the original epoch12 rather than diagnostic weights:
A: original3-angle head, original angular MSE.
B: same angle head, geometry-aware objective through differentiable price decoding.
C: same encoders with a zero-initialized unbounded3-output residual head. Predict
log-price corrections to a visible-boundary linear log-price bridge; derive angles
from the resulting trajectory. Zero residual exactly reproduces the baseline.
C receives no new raw-price input: visible boundary prices are used by its decoder.

All have512 left/right context and the same hidden-endpoint masking. Each epoch
partitions ALL eligible training targets into chronological dynamic gaps1..40,
without sampling or shuffle. Same per-epoch span partition across A/B/C.
Four full epochs per variant; this is a bounded fine-tuning comparison, not a
claim of exhaustive optimization. Preserve dropout0.25, AdamW LR1e-4, WD0.01,
gradient clip1. No LR search. Save epoch0 and every fine-tuning epoch.

Use original train before2025-09-04 13:08 UTC; validation next3 months; final
evaluation remaining9 months. These data are research-exposed. Validation anchors
are the same fixed512 as the prior run. Evaluate lengths1,5,10,20,40.
Select each variant's checkpoint by the mean across lengths of its decoded H/C/L
MSE divided by the ordinary linear bridge MSE on the same validation cases.
Test never selects checkpoints, loss weights, examples or architectures.

## Geometry objective

Prices are in log coordinates relative to the last visible Close, normalized by
fixed train-only S0. No target-derived per-window scale. Reconstruct Open from
previous PREDICTED Close; true hidden previous prices never enter the decoder.

L = L_angle +0.25 L_path +0.10 L_shape +0.10 L_multiscale +0.05 L_invalid.
L_angle = per-hidden-position squared normalized-angle error over H,C,L.
L_path = Huber(delta1) decoded H/C/L log-price error divided by S0.
L_shape = Huber(delta1) errors in body, range, upper/lower wick, divided by S0.
L_multiscale = Huber(delta1) aggregated H/C/L errors at M3,M5,M10, divided by S0;
only complete UTC-clock-aligned bars wholly contained inside the hidden span count.
L_invalid = ReLU penalties for High below max(Open,Close) or Low above min(Open,Close),
divided by S0. No output sorting, injected oscillation or variance reward.
Term weights are fixed before evaluation; record each term separately.

## Evaluation and decision

Compare saved epoch12, A/B/C, ordinary linear bridge and right-Open bridge on ALL
eligible final-test hidden targets with one fixed dynamic partition. Report angular
and decoded-price MSE/MAE, aligned body/range errors, total-variation ratio, direction,
OHLC validity, monthly/length summaries and complete M3/M5/M10 candle metrics.
Day-block bootstrap comparisons versus bridges, not only versus a causal model.
Use the exact same five previously requested picture examples, without reselection.
Sharper appearance alone does not establish better reconstruction.

Do not add probabilistic generation merely to create jagged charts. First establish
capacity and whether a price/shape objective learns real held-out residuals. A
generative follow-up would require distribution scoring and separate sample-versus-
truth evaluation; no best-of-N selection using the hidden answer.

## Verification

Differentiable decoder roundtrip, no hidden anchors, geometry-loss padding masks,
UTC aggregation, bridge identity at zero residual, finite gradients and all-target
coverage. Save code/source/config hashes, origins, checkpoints, predictions, scores,
diagnostic curves and HTML comparison. No trading-profit claim.
