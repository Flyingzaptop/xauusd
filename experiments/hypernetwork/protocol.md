# Hypernetwork comparison — fixed protocol

Task: SHORT / NONE / LONG from Close(t+8)-Close(t), strict thresholds below -$2 and above +$2. Only 512 past M1 bars. Integer-cent targets and past-only feature normalization. No reconstruction.

Data: original January 2024 train windows (27,716); February 1–7 validation (693); February 8–14 test (690). Early validation, before February 5 (265 windows), selects checkpoints. Remaining 428 validation windows select operating thresholds and model family. The test has only 30 directional targets over six dates and has already been used in earlier research.

Compare a static TCN with 635,730 parameters and a conditional-weight TCN with 635,539 parameters. Same 128-channel, five-block trunk; static head width 335 instead of 256. Hypernetwork: context 256 -> 16 -> 1,536 values, generating A (128x4) and B (4x256). One layer becomes W(context)=W0+A(context)B(context)/2. All weights are trained. Initial trunks match exactly within each paired seed; heads differ.

Two seeds (17,29), 12 full chronological passes each; ordinary unweighted CE and Adam, learning rates [2e-4]*3+[1e-4]*3+[5e-5]*3+[2e-5]*3, batch 128, microbatch 32, gradient clipping 5. CE+Adam was the validation-selected method in the preceding precision study. No PCGrad or additional objective changes in this comparison.

Precision is prioritized in checkpoint and threshold selection. The frozen inherited policy ranks candidate per-direction thresholds by the one-sided Wilson lower score, requiring at least five signals on at least two validation dates. It is a selection heuristic, not a confidence guarantee after search. No signals means undefined precision. Two seed predictions are averaged before calibration. Family selection uses calibration only, excludes diagnostic ablation, and is saved before test evaluation.

Diagnostic ablation: load each selected hypernetwork checkpoint, replace only its conditioning input by mean TRAIN context, then recompute validation probabilities and fit independent validation thresholds. The classifier still receives the real input. This disables variation in generated weights; it does not retrain a static counterpart. It is excluded from family selection.

Gradient diagnostic: 24 fixed training examples, eight per class spanning January. Measure pairwise gradient cosine conflicts and cancellation at initialization and selected checkpoint. This small balanced probe is descriptive; fewer conflicts alone do not establish better learning or generalization.

Before training, verify capacity counts, identical paired trunks, nonzero gradients through the generator, nonconstant generated weights, explicit matrix multiplication equivalence, and changed logits when conditioning changes. Save checkpoints, probabilities, curves, targets, hashes, and diagnostics. Do not tune on test outcomes.
