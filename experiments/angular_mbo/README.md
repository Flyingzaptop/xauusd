# Causal M1 + asynchronous MBO experiment

Target: the **next** XAUUSD M1 candle. All inputs stop before its nominal minute opening. This is not bidirectional infilling. Existing nine-year data and previous checkpoints are not used for fitting and are not deleted. All three networks start from scratch.

## Fixed protocol

- Source overlap: 2025-10-01 through 2026-07-30 UTC. Train October–May; validation June; final test July. July was researched in earlier experiments: this is a chronological held-out comparison for this run, not a never-seen historical period.
- Slow input: 512 completed M1 tokens, each 4 OHLC angular changes × Fibonacci lags 1,2,3,5,8,13,21,34,55. Raw support 567 prior minutes. Discontinuous contexts are excluded. S0 and H fitted before June only.
- Fast input: up to 2048 individual raw MBO records within the last 30 minutes. Variable event timing is encoded through time to cutoff, inter-event time and exchange/receipt delta. No target-minute events. This finite buffer does **not** claim to consume every event in the archive.
- Event data: price relative to last known event price, size, action A/C/M/T/F/R, side, LAST flag and timing. Order IDs are not used as numerical features. No full book reconstruction; snapshots and bad receipt timestamps are excluded. This is a raw-event encoder, not a full limit-order-book encoder.
- GC.FUT is a parent containing many contracts/spreads. Only outright GC contracts are eligible. Each UTC day selects the contract with greatest T-trade size on the preceding observed day. F fills do not double-count selection volume. First day is warmup; event buffer resets daily, preventing contract mixing.
- M1 must have continuous prior 567-minute support. At least 128 recent raw events and a latest event no older than 60 seconds are required. Every variant uses exactly the same eligible target minutes.
- Main model: causal slow TCN (64 channels, 512-token receptive field), fast causal event TCN (32 channels, local receptive field 256 events), learned attention across all 2048 event tokens, fusion head emitting 3 normalized angles H/C/L. Decode relative to last observed H/C/L; Open approximated by last observed Close. No true target anchors.
- Controls: same slow model without GC; slow model + five completed GC trade OHLC bars; slow model + those GC bars + raw MBO. GC control includes the observable GC/XAU basis. There is no guarantee order flow improves the result.
- Initial learning rate 1e-4, AdamW weight decay .01, dropout .25, gradient clipping 1. Full chronological passes over every eligible training target, no random target sampling. Maximum 30 epochs; early stop after 5 epochs without validation price-MSE improvement; halve LR after 3 non-improving validation checks. Batch 16. No state is carried across samples.
- Loss = angle MSE + .25 × Huber of decoded HCL log-price error in S0 units + .05 × invalid-candle penalty. Same objective for all variants. The Open=previous Close approximation can be violated by real source gaps.
- Model selection uses June HCL log-price MSE; July evaluated after all variants finish. Report price MSE in bps squared, angle MSE, price MAE USD, Close direction and invalid-candle fraction. No PnL claim.

## Timing limits

The local M1 timestamps are treated as candle **start** labels; decision time is the next minute opening. Capture-server `ts_recv` is used for MBO availability. Broker delivery latency and the exact M1 vendor timestamp convention are not independently documented in these files. This is a historical causal experiment under those explicit conventions, not a live execution backtest. GC futures and XAUUSD are different price sources/instruments.

## Commands and artifacts

```powershell
python -m pytest experiments/angular_mbo/test_causality.py -q
python -u experiments/angular_mbo/pipeline.py
```

Outputs: `reports/angular_mbo_causal_20261008/`. `pipeline.json` reports stage or failure; `manifest.json` contains actual eligibility counts, contract decisions and source archive metadata. `cache/` preserves individual event values in bounded windows. `verification.json` audits every event cutoff. `progress.json` and stage logs report training. `models/*/best.pt`, `curve.json`, `results.json`, `results.html` and final test arrays are written automatically. No periodic UI monitor is installed.

SDK definitions: https://databento.com/docs/schemas-and-data-formats/mbo ; https://databento.com/docs/standards-and-conventions/symbology ; https://databento.com/docs/standards-and-conventions/mbo-snapshot .
