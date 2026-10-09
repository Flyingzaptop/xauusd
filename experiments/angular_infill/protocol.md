# Angular missing-candle reconstruction, 2026-10-08

The previous chronological forecaster had already completed30 epochs when the user
requested stopping it. Its final checkpoint is preserved and initializes this model.
User requests reconstruction of missing middle candles using observed left AND right.

## Task and data

Input:512 angular tokens left, a dynamic gap,512 tokens right.
Each token has36 angular channels using Fibonacci lags1,2,3,5,8,13,21,34,55.
Add36 availability flags and one gap flag; no raw/log prices are neural inputs.
Output: three k1 angles H,C,L at each hidden position.
The earliest raw support extends55 bars before the first left token.

Mask a feature whenever either of its raw price endpoints is hidden. A chord whose
two endpoints are visible is allowed even when its interval crosses the gap.
The known Open of the first right candle may approximate the last hidden Close;
this is explicitly permitted future context, not a forecast result.

Use source and train-only angular scale from the completed chronological run.
Last source year (2025-09-04 13:08 UTC onward) remains a weight-training holdout.
Its first three calendar months are validation for curriculum decisions; the
remaining nine months are final test. Split segments at gaps AND all partition
boundaries, so neither context crosses partitions. Existing research exposure
is acknowledged; these periods are not untouched confirmation data.

Within each partition, valid hidden targets run from segment_start+567 through
segment_end_exclusive-513. Partition this entire interior into consecutive spans
with lengths sampled uniformly from1..current_max, clipped only at segment ends.
No random sample of targets: each eligible hidden candle is a loss target exactly
once per epoch. Change span lengths each epoch, but process them chronologically
without shuffle. This covers all eligible hidden candles, not every overlapping
placement of a missing span. Extra context can materially reduce eligible data
when sessions contain gaps; report exact counts rather than silently filling gaps.

## Curriculum (user-requested)

Stages: lengths1..5, then1..10, then1..20, then1..40. Never advance solely because
an epoch count elapsed. Validation uses512 fixed anchors with enough support for
all lengths1..40. Evaluate lengths1,2,3,4,5,10,20,40 up to the current maximum,
balanced by length; reuse frozen causal rollouts for the same anchors. Advancement requires:
- at least3 full training epochs in the current stage;
- two consecutive validation checks with >=20% lower reconstructed H/C/L price
  MSE AND >=20% lower Close-path MSE than the frozen causal forecaster at the
  current maximum gap length;
- angular H/C/L MSE no worse than the causal comparator at all evaluated lengths.
These are explicit engineering gates, not a guarantee of exact reconstruction.
If gates fail, remain at the current difficulty within the30-epoch total budget.
Do not claim mastery of unvisited stages. Validation alone controls progression;
the final test never controls it. Reset plateau statistics on stage advancement,
but never raise LR. Final epoch30 is used; no test-based checkpoint selection.

## Model and loss

Two TCN encoders, one forward and one backward, width64, dropout0.25.
Each has ten causal residual blocks, dilations1..512, kernel2 plus kernel1,
per-token LayerNorm. Each branch receptive field1024 covers its visible side.
Initialize both branches' first7 blocks from the finished causal epoch30 model;
initialize the three additional blocks as residual identities. Expand input from36 to73 with
zero weights on new mask channels. Decoder combines previous-position forward
state and current-position backward state; initialize its left weights from the
old head and its right weights to zero. Fine-tune all weights.

Pure mean squared error of normalized H/C/L angles over hidden positions only.
No loss on Open, observed candles or padding; no price/path/boundary loss.
AdamW initialLR1e-4, weight_decay0.01, gradient clip1; dropout25%; seed17.
Thirty complete chronological passes, batch16 spans; no early stopping.
Reduce LR by0.5 on training-loss plateau (patience3, relative threshold0.001,
minimum1e-6). Holdout never selects epoch/LR. Final epoch30 is primary.
Atomic epoch checkpoints preserve optimizer/scheduler/RNG for resume.

## Decoding and evaluation

Decode the entire hidden H/C/L trajectory by integrating predicted k1 angles from
ONLY the last observed left OHLC. Never use actual hidden previous candles as
anchors. First hidden Open=last observed Close; subsequent Open=previous PREDICTED
Close. This is recursive reconstruction, not teacher-forced price evaluation.

Evaluate every final-test span using the final trained maximum and deterministic
dynamic lengths. Also report metrics by length and gap position.
Compare on identical gaps: the frozen old causal model rolled forward20 steps
without future, linear log-price interpolation to first visible right OHLC, and
a bridge using right Open as last hidden Close. Same model with right features
removed is an explicitly OOD input ablation, not a separately trained forecaster.
Report angular and reconstructed price errors by H/C/L, gap position, month,
OHLC validity, body/range errors, and paired day-bootstrap intervals.
Interpolation is an easier task than forecasting; no trading-profit claim.

## Verification and autonomy

Tests cover hidden-endpoint masking, admissible visible chords, input invariance
under hidden-value changes, partial spans, all-target coverage, decoder independence
from hidden anchors, model initialization, finite loss and gradients.
The run automatically trains, evaluates baselines, saves predictions/checkpoints,
verifies saved metrics and writes an HTML report. Respect the user's request not
to monitor the long run: launch it and leave it running to completion.
