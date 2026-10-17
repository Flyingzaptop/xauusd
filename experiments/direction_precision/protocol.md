# Precision-first directional classification

Same confirmed target: terminal Close after8 minutes minus current Close;
strictly above+$2 LONG, below-$2 SHORT, boundaries NONE. Past512 inputs, same
causal normalization and605315-parameter TCN. January2024 train, same full
chronological windows. No reconstruction or right context.

Training objectives, fixed before fitting:
- ce: ordinary, unweighted cross-entropy; matched control.
- fp9: CE +9 times the sum over SHORT/LONG of -log(1-p[class]) whenever that
  class is false. Thus a false directed prediction receives an additional
  penalty; natural class frequencies replace the old inverse-frequency loss.
The cost-shaped outputs are scores, not asserted calibrated probabilities.

Both objectives use Adam and Adam+grouped-PCGrad with seeds17/29, identical
initial weights within seed across all four treatments. Last-layer bias starts
at log(train class priors) for every treatment. Twelve full epochs, same LR
schedule2e-4/1e-4/5e-5/2e-5 for three epochs each; batch128, groups32, clip5.
Eight fits, expected about5 GPU minutes;25-minute watchdog.

Validation is chronologically divided: timestamps before2024-02-05 select the
checkpoint (265 windows), the remaining428 windows fit final thresholds. Only
7 SHORT and7 LONG occur in the threshold partition. This tiny pilot cannot
establish reliable high precision from isolated successes.

A signal's candidate direction is the higher of SHORT/LONG softmax scores.
The two routes are disjoint; a class-specific threshold can turn a direction
into NONE. Ties route to SHORT. At each epoch, independently choose thresholds
on the checkpoint partition maximizing the one-sided95% Wilson lower-score
per direction among cutoffs with >=5 predictions on >=2 dates. Average the
two lower scores for checkpoint selection; unavailable direction gets score0.
Include all equal-score ties. Earlier epoch wins identical selection scores.

After all checkpoints are fixed, average the two seeds' scores per treatment.
Fit final thresholds on the second validation partition with the same rule.
Pick the treatment with highest mean lower-score there. Freeze all choices
before test. Evaluate every treatment once on test; report the preselected
one, not the test winner. The Wilson score is a selection heuristic, not a
coverage guarantee after adaptive selection or under temporal dependence.

Report precision, TP/FP, signal count, coverage and recall, independently for
SHORT/LONG and pooled directed signals. Zero-signal precision is undefined,
never100%. The minimum signal rule applies only to validation; test may still
produce zero signals. No mandatory trade quota or lowered test threshold.

Test is the prior research split,690 windows and30 directional events across
six dates. Compare the natural CE control with fp9 to separate threshold
selection from loss changes; do not claim that loss engineering guarantees
high precision. Save weights, epoch curves, fitted policies, validation/test
scores and labels. Download all results and stop the Colab GPU.
