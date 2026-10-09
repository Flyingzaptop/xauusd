# Full chronological angular training, 2026-10-08

User: every possible training point in chronological order, repeated full epochs;
exclude the last year; initial learning rate1e-4, reduce if needed, dropout20–30%.

## Fixed configuration

Use every valid M1 origin, no random subsampling, no shuffling. Targets H,C,L next
angles at lag1. Open is reconstructed as the last observed Close outside the net.
Input only36 angular features (H,O,C,L x lags1,2,3,5,8,13,21,34,55).
128 sequential feature tokens -> raw support183 bars. Exclude gaps/invalid OHLC
and insufficient warmup; enumerate and audit all admissible origins.

Last year boundary = final source timestamp minus one calendar year. All targets
at/after this boundary are held out. Training targets precede it. Past training
history may provide context at the start of holdout; no future context ever used.
The source period has been examined in previous experiments. This is a reserved
holdout for this run, NOT a globally untouched dataset.

Refit fixed S0,H on only training-time, gap-free close returns; g1. Same formula:
a(t,k,X)=2/pi*atan(log(X_t/X_(t-k))/(S0*k^H)). Save scale and source fingerprint.

TCN width64, seven residual blocks. Each block: causal kernel2 convolution at
dilation1,2,4,8,16,32,64, per-token LayerNorm, GELU, dropout0.25, kernel1 convolution,
per-token LayerNorm, GELU, dropout0.25, residual. Receptive field exactly128.
Head3 bounded tanh outputs H,C,L. No normalization over time or future access.

Dense sequence implementation emits an output for EVERY target token in a chunk,
instead of redoing127 overlapping feature computations for each next origin.
Chronological chunks contain up to2048 targets plus127 past feature tokens.
No chunk crosses a data gap. Predict the whole chunk before its optimizer update.
Unit tests compare dense inference with independent128-token windows.
This is chronological minibatch training, not one optimizer update per candle.

Thirty COMPLETE epochs, one initialization seed17. Each epoch returns to the start
of the training timeline and covers each eligible origin exactly once. No early
stopping and no truncated epochs. A small throughput benchmark uses only the first
training chunks and an independently discarded model; reset RNG before official fit.

AdamW initial LR1e-4, weight_decay0.01, dropout0.25, gradient clip1. Equal MSE over
H,C,L angles; no Open term or price/shape penalty. ReduceLROnPlateau on sample-weighted
chronological training loss: factor0.5, patience3, relative threshold0.001, minLR1e-6.
Holdout is NOT used for LR, epoch count, checkpoint choice or architecture selection.
Use final epoch30 checkpoint for primary results; save all epoch checkpoints and
atomic resume checkpoints including optimizer, scheduler and RNG state.

Epoch logs show exact origin count, updates, LR and H/C/L losses; per-year train
loss summaries preserve chronological diagnostics. These losses after repeated
epochs are training diagnostics, NOT out-of-sample walk-forward returns: later
passes reuse previously seen historical labels.

Evaluate frozen final weights once on ALL eligible held-out targets, no stride.
Baselines: zero angles, train mean angles, flat candle at priorClose, and train
median OHLC template; reconstruct baseline Open consistently from priorClose.
Metrics: per-channel angular MSE/MAE/R2, decoded price errors, valid OHLC, body/range
errors, direction. Monthly metrics and paired day bootstrap2000 on HCL/Close loss.
No P&L claim. Artifacts include full origin lists, coverage audit, training curves,
checkpoint hashes, predictions, independent metric verification and HTML report.
