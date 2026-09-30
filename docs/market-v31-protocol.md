# V3.1 preregistration: adequacy of training and separate calibration

This is a new explicitly labelled research iteration after viewing v3. V3 and
its negative results remain unchanged. No new architecture, feature, scale,
instrument or data source is introduced. The final January–August 2026 period
is exposed exploratory evaluation, never fresh independent confirmation.

## Fixed data and state policy

Keep the exact v3 M1 file, train period 2024-01-01 through 2025-06-30, ten-minute
direction target (zero = not-up), thresholds [8,4,2] times the original train
median absolute log return, original train normalization/clipping, model widths,
discrete routing inputs, and TreeCrossingMemory implementation. Reuse original
train and evaluation arrays after hash/array equality checks.

The former validation period is split before any fitting:

| Role | Inclusive start | Exclusive end | Permitted use |
|---|---|---|---|
| Train | 2024-01-01 | 2025-07-01 | Adam updates and unchanged preprocessing |
| Selection | 2025-07-01 | 2025-10-01 | Checkpoint choice and stopping only |
| Calibration | 2025-10-01 | 2026-01-01 | Positive affine calibration only |
| Exposed evaluation | 2026-01-01 | 2026-09-01 | Final research reporting only |

Reset router and neural state at each split boundary and every actual minute
gap. Rebuild October routing from zero; do not carry September state or labels
across the selection/calibration boundary. Purge targets crossing each boundary.
Common warmup is 64 observations per contiguous segment. All rows advance state,
and numerical state carries across TBPTT blocks of128 observations; detach the
gradient graph only. Sixteen lanes, deterministic whole-segment packing, epoch
resets and padding remain v3-identical. No calibration/evaluation state enters
selection, and neither calibration nor evaluation labels train the neural model.

## Fifteen from-scratch fits and stopping

Models A/B/C/no-transfer/no-open and seeds7/19/43 are unchanged. All fits restart
from the recorded seed, not v3 weights. Use the same Adam defaults, learning rate
0.001, gradient norm clip1.0 and one PyTorch thread. Every full train epoch has
491102 supervised examples and294 optimizer updates. No scheduler or learning
rate search. Checkpoint selection uses stable binary log loss from raw logits.

Run at least four and at most20 full epochs. During epochs1–4 maintain the exact
minimum selection loss; patience is not charged. After epoch4 initialize the
significant-improvement reference to that minimum. For epochs5 onward, a loss
at least0.00001 below the reference updates it and resets the nonimprovement
counter; otherwise increment that counter. Stop when it reaches five, or at20
epochs. Therefore the earliest stop is epoch9. Independently retain the exact
lowest-loss checkpoint over all seen epochs, including improvements smaller
than the stopping threshold. Exact ties favor the earlier epoch.

Per fit retain two selected checkpoints, both including model and optimizer:
`short_best` is the exact best among epochs1–4; `full_best` is the exact best
over the completed run. A separate overwritten `last.pt` is recovery state,
not an evaluation candidate. It retains optimizer, RNG, counter and curve state
for an exact epoch-boundary restart. It does not add a third selected variant.
Checkpoint writes and progress reports are atomic. All epoch/batch curves and
selection losses are kept, including late deterioration. Four-epoch v3.1
checkpoints use the **new selection period** and must not be confused with v3's
old six-month selection.

Cap outcomes are not automatically called convergence. Report stopping reason,
last meaningful improvement, exact best epoch and the last-five-epoch loss
change for every run. If meaningful improvement still occurs near epoch20,
state explicitly that no plateau was demonstrated. Do not extend the budget.

## One calibration, fixed in advance

For each of the30 selected checkpoint identities, replay the calibration split
with fresh state and common warmup. Fit `sigmoid(a*z+b)` on its raw logit z,
with `a >= 1e-6 > 0`, minimizing unregularized mean binary log loss. Optimize
directly in the convex variables (a,b), from (1,0), with L-BFGS-B, maxiter1000,
ftol1e-12 and gtol1e-8. There is no upper bound on a or b, no temperature/model
search and no evaluation-informed retry. Publish solver status, objective and
gradient diagnostics and flag lower-bound slopes as near-constant calibration.
Numerical nonconvergence must be reported rather than hidden.
If the solver reports failure but its projected gradient infinity norm is at
most1e-6, retain it with the distinct `projected_gradient_converged` status;
otherwise stop and report the numerical failure. This fixed acceptance check
does not change the objective, fit another calibrator or inspect evaluation.

Same-epoch short/full selections represent the same neural checkpoint and may
reuse identical inference/calibration, but both identities remain in tables.
Calibrators are fit only after all fifteen neural runs finish. Freeze all fitted
calibrator coefficients and their hashes before opening evaluation scores.

Positive a preserves ranking. Report ROC-AUC using the original logits for raw
and calibrated variants, avoiding artificial probability-saturation ties.
Check monotonicity numerically and assert equal reported AUC. Calibration can
change log loss, Brier and reliability, not ranking ability. A constant trained
only on train up frequency remains a raw control, not another selected model.

## Comparisons and inference

The primary comparison is **full_best calibrated C minus full_best calibrated
A** in log loss. Report raw/full, raw/short and calibrated/short C-A alongside
it, not whichever variant looks best on evaluation. C-B is secondary and cannot
establish success by itself. Compare C with both ablations under each matched
checkpoint/calibration choice. Compare full versus short and calibrated versus
raw within every model to separate extended training from calibration effects.

After all training AND all calibrators are complete, score all60 selected
variant identities (5models ×3seeds ×2checkpoints ×2calibration states), plus
the constant. Publish raw and calibrated log loss, AUC, Brier, ten-bin ECE and
all calibration-bin counts; every seed and every month. Do not select variants
or alter anything based on exposed evaluation.

Use paired bootstrap of entire UTC calendar weeks,2000 replicates, fixed RNG
seed20260930. Average paired minute-loss differences over the three fixed seeds
before resampling weeks; retain per-seed differences. No independent-minute
bootstrap. Intervals condition on these seeds and the exposed period and do not
measure full initialization/regime uncertainty; secondary contrasts are not
multiplicity-adjusted discoveries.

If full calibrated C continues to lose to A, finish this iteration with a
negative conclusion. Even if C wins, require the ablations to assess whether
transfer/open memory explains it, and avoid a fresh-confirmation or trading-profit
claim. No automatic wider search follows either outcome.

## Budget and preservation

The stopping rule implies9–20 epochs per run:2646–5880 updates and
4419918–9822040 supervised visits per fit. Across15 fits the fixed bounds are
39690–88200 updates and66298770–147330600 supervised visits. Actual values,
not these maxima, will be reported. Up to nine independent one-thread fit
processes may overlap; this changes scheduling only. Prior v3 speed suggests
several aggregate CPU-hours, so this is not a short smoke test. Checkpoint and
calibration/evaluation inference costs are additional and separately timed.

Preserve every v3 source/result file. New caches, optimizer states, logits and
weights live under ignored `artifacts/market-v31/`; publish checksums and exact
commands. Protocol/source is pushed before any neural fitting; final results
are a separate commit. A later fitted change requires a new labelled iteration.
