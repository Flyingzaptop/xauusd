# Angular next-candle TCN, fixed protocol 2026-10-08

User specification: sequential model; ONLY angular history as input; exactly FOUR
next-candle angles relative to the last visible OHLC candle as output. No teacher,
distillation, extra raw/log-price features, or future observations in network input.

## Representation

Channels ordered H,O,C,L, lags 1,2,3,5,8,13,21,34,55, lag-major layout.
a(t,k,X) = (2/pi) atan(log(X_t/X_(t-k))/(S0*k^H)), fixed gain1.
Reuse verified train-only S0=9.489368821292693e-5,H=0.5134906772739439 from
the earlier audit (estimated before 2025), checking the source SHA256 matches.
Input [batch,36,128]. Raw support t-182..t. Target [batch,4] a(t+1,1,X).
Decoder outside network: Xhat_(t+1)=X_t exp(S0*tan(pi*ahat/2)).
Tanh outputs bounded to (-1+1e-6,1-1e-6) solely for finite numeric inversion;
report near-boundary frequency. No OHLC sorting/projection in primary metrics.

## Sequential architecture and training

Causal TCN, 32 hidden channels, six residual blocks with dilations 1,2,4,8,16,32.
Each block: two left-padded kernel3 convolutions, per-token channel LayerNorm,
GELU, dropout0.1; residual connection. Receptive field253 covers all128 input tokens.
Head: final token ->4->bounded tanh. No flattening of the time axis, bidirectionality,
temporal normalization, or global pooling over unseen tokens.
Train pure mean squared error of the FOUR normalized angles (equal weight).
AdamW lr0.001, weight_decay0.01, batch256, gradient clipping1; seeds17,29,43.
Up to24 epochs, early stopping patience5 after at least8 epochs. ReduceLROnPlateau
factor0.5 patience2; min_lr1e-5. Select minimum validation all-channel angular MSE.
Keep all seed checkpoints; arithmetic ensemble in angle space is a predefined
additional model for evaluation and inference, not selected using research outcomes.

## Data and splits

Local minute parquet from 2016-09-04 through 2026-09-04. Unique sorted timestamps,
positive finite valid OHLC. Every support window t-182..t+1 must be uninterrupted
at one-minute spacing and wholly inside its split.
Train before2025: fixed uniform sample100000 origins; validation2025:12000 origins;
evaluation Jan-Aug2026: every fifth eligible origin; September tail separately.
2026 evaluation including September is now exposed by prior experiments, NOT
untouched confirmation. No tuning using any of these evaluation outcomes.

## Baselines and interpretation

1. Zero angles -> repeat the entire previous candle (H,O,C,L unchanged).
2. Train mean next angles -> optimal constant under training angular MSE.
3. Repeat latest k1 angles -> constant channel-wise log-price velocity.
4. Flat next candle at last Close -> zero Close return and near-known next Open.
5. Train median OHLC offsets relative to previous Close -> static candle template.
Baselines3-5 use only available past/source-train values; baseline4/5 obtain raw
last prices externally as the decoder does. They are mechanical reference forecasts,
not additional features of the TCN.

Report per-channel angle MAE in degrees, normalized angle MSE/R2, angle-sign accuracy,
decoded log-price errors in bps, absolute price MAE, body/range/wick errors,
body-direction accuracy, Close-return direction accuracy, invalid-OHLC fraction,
and per-month/seed stability. Report Open separately: it may nearly equal prior Close.
Do not interpret high pooled accuracy as proof of Close prediction.
Compare with strong baselines, not just zero angles. Day-block bootstrap2000
replicates for ensemble vs each baseline, angle MSE and price/Close errors.
Intervals exploratory, no multiplicity correction or P&L claim.

## Verification and delivery

Tests: target/decoder roundtrip, future-mutation invariance, multiplicative price
invariance, channel ordering, split support/continuity, TCN prefix causality,
finite bounded head and gradients. Save source/code/protocol hashes, exact origins,
curves, weights, per-example predictions and metrics. Recompute metrics separately.
Plots of true vs predicted angles, candles for deterministic test quantile origins,
worst errors clearly labelled, and training curves. Inference command exports next
candle from last available historical bar; never describe it as a live market quote.
