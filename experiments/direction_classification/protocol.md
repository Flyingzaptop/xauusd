# Direction classification: initial month pilot

Target assumption stated before fitting: terminal Close after eight future M1
candles minus the current Close. SHORT below -$2, LONG above +$2, NONE includes
both boundaries. Prices recover exactly to cents from the source log prices;
integer-cent comparisons avoid floating-point threshold mistakes.

Use the exact January 2024 training windows and the Feb1-7 validation / Feb8-14
test windows from the preceding month pilot. Train27716, validation693, test690;
the eight future candles span exactly eight minutes. Same split support rules.
The full raw file contains ten years, but only these month-pilot indices train
the model. Test was previously used for reconstruction research.

Input consists only of512 candles ending at the decision Close. No right-side
context, interpolation, or reconstruction target enters the classifier. Scale
is the standard deviation of511 past close log changes, floored by the training
windows' fifth percentile. The old two-sided scales are never used.

TCN width128, five residual blocks with dilations1/2/4/8/16; eight-candle patches
retain all OHLC/time coordinates. Last state and mean state are concatenated;
the head is256->256->128->3. This restores the original trunk capacity, while
the head and single-sided pooling adapt to classification. Parameter count is
recorded at runtime. Fresh initialization; no reconstruction weights are used.

Class-weighted cross-entropy with w[c]=N/(3*N[c]), from train only. Take the
arithmetic mean of per-example weighted losses, so grouped and whole-batch
gradients match. Raw logits target equal class priors; report raw argmax and
separately argmax(logits-log(w)), corresponding to the training class priors.
Neither is presented as a calibrated probability guarantee.

Adam vs Adam with grouped PCGrad, two seeds17/29, identical pair initialization.
Batch128, groups32, clip5, chronological complete epochs. Twelve epochs with
LR2e-4/1e-4/5e-5/2e-5 for three epochs each. Best checkpoint by validation
class-averaged negative log likelihood; no test selection or threshold tuning.
Save epoch-end full train and validation metrics, confusion matrices and
per-class precision/recall/F1. Evaluate all test predictions only after fitting.

Baselines: always NONE, train priors and uniform probabilities. Majority
accuracy is insufficient: emphasize balanced accuracy, macro F1 and LONG/SHORT
precision and recall. Test contains only30 directional examples, so conclusions
about directional generalization are necessarily preliminary.

Record source/data hashes, threshold boundary checks, future-price perturbation
check, matched initial weights, full epoch coverage and PCGrad reference checks.
Download checkpoints and predictions; stop the Colab GPU after completion.
