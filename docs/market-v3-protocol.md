# M1 streaming research comparison, v3

This protocol is prepared before training, then frozen and pushed with the measured preflight speed and exact sample/update counts. V1/v2 code and reports remain preserved. January–August 2026 was already evaluated in v1: all v3 evaluation on that period is **exploratory research on exposed data**, not new independent confirmation. There is one configuration and three seeds; no settings are chosen from evaluation scores.

## Data, targets and splits

Only the existing M1 XAUUSD parquet is used. No additional instrument, MBO or depth. Train: 2024-01-01 inclusive to 2025-07-01 exclusive; validation: 2025-07-01 to 2026-01-01; exposed evaluation: 2026-01-01 to 2026-09-01. Use all available rows in these periods, with no window subsampling. Hashes and exact counts are frozen in the data preflight report.

Metadata declares UTC candle-open timestamps. Forecast after the candle closes, one minute after its timestamp. Label is close(t+10)>close(t), zero change is not-up. A target is valid only if all intervening timestamps advance by exactly 60 seconds, remain in the same split, and the example has completed the common warmup. Every irregular interval (including ordinary overnight/weekend closure or a missing minute) starts a new stream segment; no holiday calendar is guessed. Invalid OHLC or duplicate/nonmonotone timestamps cause an error rather than silent repair.

## Shared history and state

Each split starts from zero state. Each contiguous segment starts fresh router and neural states. First 64 observations of a segment are processed as warmup and excluded from loss/metrics; prediction targets begin at the following observation. Last 10 observations before a gap/split are also excluded from supervised labels. All observations, including warmup/tails, advance state. No 64-minute window reset occurs.

Whole segments are assigned round-robin to 16 independent lanes, retaining chronological order within each lane. All models use the same assignment, inputs, valid masks, reset flags and label times. Padding is invalid and explicitly reset. Models never transfer state between lanes or unrelated segments. Evaluation predictions are mapped back to original timestamps before metrics.

Truncated backpropagation uses blocks of 128 observations per lane. At a block boundary the hidden state is detached from the gradient graph, **not zeroed**. The router's causal state is precomputed over entire uninterrupted segments, not independently per block. Numeric recurrent history can therefore be much longer than 128 observations, although gradient credit assignment is limited to the current block. States carried during training may have been computed using pre-update weights; this standard TBPTT policy is identical across models. No state from training enters validation/evaluation; each is replayed with the selected weights and its own common warmup.

## Preprocessing and operator

Four OHLC features: one-step log-close change, log(close/open), log(high/close), log(low/close). The first change at each real reset is zero. No volume feature. Standardization uses train mean/std only; standardized features clip to [-10,10].

V2 thresholds in descending order are [8,4,2] times the train median absolute one-minute log-close change. Thresholds are fixed throughout all streams. V2 temporal ownership, symmetric comparisons, single-observation jumps and deepest-first closure follow the accepted operator unchanged.

The six metadata channels at each of four depths include direction, log1p age, signed progress, maximum excursion, log1p retreats and latest signed move. Before train-only standardization, signed progress and latest move receive signed-log1p transforms; maximum excursion receives log1p. Already-log age/retreat channels are not logged twice. Standardized metadata clips to [-8,8]. Absent slots remain governed by explicit routing masks. Preflight reports ranges, saturation, bounded neural states and finite gradients on long streams. This normalization is a v3 model-input choice and does not alter event routing decisions.

## Models and budget

A: stateful GRU on four M1 features. B: stateful GRU on those features plus all v2 router inputs: 24 continuous metadata channels and 16 discrete channels (birth, closure, active masks and one-hot observer depth). C: exactly the same B inputs plus a learned TreeCrossingMemory readout, followed by a stateful GRU. Thus C does not get privileged routing flags that B lacks. C ablations disable learned child-content transfer or learned open-state readout, preserving the common metadata/flags and the rest of the model. A constant predicts train valid-label up frequency. All outputs are binary logits; no auxiliary regression loss in this stage.

C uses memory width 8 and GRU width 16. A/B widths are chosen by parameter-count proximity before fitting, not by labels. Exact counts are reported; ablation counts are nominal even if a disabled path gets no gradient. Every model gets all train labels each epoch, the same four full epochs, batch lanes, TBPTT length, Adam 0.001 and norm clipping 1.0. Three seeds 7/19/43, one configuration, no hyperparameter sweep. Checkpoint is the epoch with lowest **validation** log loss. Fixed epoch budget has no test-driven early stopping. Report batch/epoch train curves and validation curves, not a smoke-test result presented as full training.

Preflight timing uses throwaway models and independent random labels solely for throughput measurement; no such weights become research checkpoints. Exact full-pass steps and total expected updates are derived from the deterministic lane packer and published before fitting. Long-stream tests precede that freeze. Fit processes may run concurrently using one CPU thread each; elapsed-time comparisons must acknowledge shared-resource contention.

## Evaluation and inference

Primary metric log loss; secondary ROC-AUC, Brier and ten-bin equal-width calibration (ECE plus counts/predicted/observed frequencies). Publish each seed, overall results and monthly periods. Primary contrast is C-B, which isolates learned memory beyond the shared representation; B-A assesses representation and C-A overall effect. Also report C minus each ablation and each model versus constant.

Paired uncertainty resamples entire UTC calendar weeks, keeping all dependent minute observations within a week together. For the main aggregate contrast, average per-observation loss differences across seeds before the 2000 whole-week resamples; report 95% percentile intervals and per-seed differences. Do not bootstrap individual overlapping targets. These intervals are conditional on this exposed period and chosen training runs, not untouched confirmation or full uncertainty about future markets.

All 15 fits finish before the exposed evaluation is scored. Checkpoint selection uses validation only. After evaluation scores are opened, training source/config changes or further fitted variants require a separately labelled research iteration. Negative results, complete curves and unfavorable seeds remain published. No profitability claim follows from predictive metrics.
