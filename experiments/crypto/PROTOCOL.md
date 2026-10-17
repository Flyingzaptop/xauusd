# BTC/ETH (Binance USD-M perpetual futures) — protocol (declared before running, 17.10.2026)
Data: data.binance.vision monthly 1m klines (incl. taker-buy volume) and funding rates, 2020-01..2026-09, CHECKSUM-verified.
Costs per round trip, reported as scenarios: 4 bp (maker/maker), 7 bp (maker+taker), 10 bp (taker/taker). No slippage model.
Walk-forward by calendar year 2021..2026 (training on all earlier data). Candidate = net mean > 0 at the stated cost, t ≥ 3, positive in ≥ 5 of 6 years.
C1 VWAP: daily VWAP (reset 00:00 UTC); after |close−VWAP| ≥ k·σ (σ = rolling 120-min std of the deviation), k ∈ {2,3,4}:
   trend (target d further, stop at VWAP) and reversion (target VWAP, stop d beyond); conservative same-bar rule, timeout 240 min.
C2 Order flow: features = returns 1/5/15/60/240/1440 min, realized vol 60/240, taker imbalance (buy−sell)/volume over 1/5/15/60 min,
   time of day; ridge → forward return over 15/60 min, entry Open[t+2]; top-5% |pred| non-overlapping trades.
C3 Funding: at each funding time, z = funding / trailing-90-day std; |z|>2 → trade the next 8 h / 24 h, direction from past years.
C4 Shock bars: |1-min return| ≥ 5σ (trailing 1-day σ) AND volume ≥ 5× trailing median → next 5/15/60 min, direction from past years.
