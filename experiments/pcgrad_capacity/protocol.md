# Sequential capacity halving, restored after environment reset

One-month January 2024 train (27716 windows), Feb 1–7 validation (693), Feb 8–14
test (690). Exact recovered dataset SHA74534bc17ccda0a0b849997ca5bee5d1f4bb35b5a31f6075c24517f751636410.
Eight hidden minutes, 512 actual candles on either side. Two-sided reconstruction,
not causal forecasting. Same full chronological epochs and disjoint split support.

Width128 reference has 617504 parameters. Scale encoder and decoder widths
jointly, preserving five dilation blocks and the input receptive field. Choose
integer widths nearest half previous parameter count P(w)=36w²+216w+32.
Widths 90,63,44,30,20,13,8,5,3,2. Stop before width1, which makes single-channel
LayerNorm erase its input variation. Verify exact width128 state/output parity.

Two seeds17/29, two arms MSE+Adam and MSE+Adam+PCGrad at every width. Same
initialization within pair, batch128, four contiguous PCGrad groups32, clip5,
no weight decay, deterministic zero latent. Twelve epochs; LR2e-4,1e-4,5e-5,2e-5
for three epochs each. Original group gradients used in random-order PCGrad.
Record both online train loss and full train loss evaluated at fixed epoch-end
weights, matching validation evaluation. Best checkpoint by validation only.

Adaptive criterion fixed before fitting: for EACH arm, mean of seedwise best
validation in epochs4–12 must be >=0.2% lower than mean best in epochs1–3;
fixed train loss must fall between those selected epochs for BOTH seeds; late
validation must be no more than0.2% worse than the original width128 mean best.
Stop when BOTH arms pass, otherwise halve again until width2. Test is evaluated
only AFTER all width decisions are frozen. Select best width per arm by mean
best validation across seeds. Report all sizes and both best and last test.

Watchdog45min. Save checkpoints, optimizer, curves and independent diagnostics;
download results and stop GPU. A small pilot with six test dates, not proof of
absence of information or a complete optimizer/architecture search.

## User-requested extension after the first segment

The original stopping rule passed at width20 (18752 parameters). The user
explicitly requested further reduction. `train_smaller.py` therefore evaluates
ALL remaining widths13/8/5/3/2, without stopping on that criterion. Seeds, data,
initialization pairing, LR, twelve full epochs, and evaluation are identical.
The remaining five widths were fixed before this segment's test evaluation.
The first segment's test had already been evaluated. No test result is used
to select a size or checkpoint. This remains a reused research test split.
`merge.py` preserves both raw run reports and their source hashes before making
a combined summary; both scripts must be retained for reproducibility.
