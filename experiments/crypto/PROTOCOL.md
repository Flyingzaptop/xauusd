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

# Part D — positioning data (declared before running, 17.10.2026)
Data: daily `metrics` (5-min): sum_open_interest, count/sum toptrader long-short ratios, count_long_short_ratio, sum_taker_long_short_vol_ratio,
2020-09..2026-09. A 5-min metrics row is used only from create_time + 5 min (conservative availability); entry at the next 1m Open.
D1 OI/price quadrants over the last 15 / 60 min: |Δprice| and |ΔOI| each beyond their trailing-30-day 90th percentile →
   4 classes (price↑OI↑, price↑OI↓, price↓OI↑, price↓OI↓; price↓OI↓ ≈ long liquidation, price↑OI↓ ≈ short squeeze) → next 1 h / 4 h.
D2 Long/short ratios (accounts, top traders positions) and taker long/short volume ratio: z vs trailing 30 days, |z|>2 → next 1 h / 4 h.
Direction per class from past years (walk-forward 2021..2026), non-overlapping trades, same cost scenarios and candidate rule.

# Part E — raw aggTrades (declared before running, 17.10.2026)
Data: raw monthly aggTrades (Binance USD-M), not aggregated. Discovery month: 2026-09. Validation: earlier months as they download
(2026-08, 2026-07, …), never used for discovery. Rules found in September are frozen and re-run unchanged on validation months.
Taker order = consecutive aggTrades with the same transact_time and the same aggressor side (sweep across levels).
Latency: 200 ms (entry = first trade price at/after t+200ms); returns measured on trade prices; costs 4/7/10 bp per round trip.
E1 Large taker orders: notional ≥ $1M and ≥ $5M (BTC), ≥ $0.5M and ≥ $2M (ETH) → returns at 10 s, 1 m, 5 m, 30 m (with vs. against).
E2 Absorption: 1-min windows with |signed taker notional| in the top 1% of the trailing day but |price change| in the bottom 30% →
   returns at 5 m / 30 m / 60 m against vs. with the aggressor.
E3 Sweeps: taker orders crossing ≥ 3 price levels with notional ≥ p99 → returns 10 s … 30 m.
E4 Bursts: number of taker orders per second ≥ p99.9 of the trailing day with side imbalance ≥ 80% → returns 1 m / 5 m / 30 m.
Reporting: mean signed bp, t clustered by minute, sign by week; a rule is a candidate only if it is positive after costs in September
AND in every validation month with the same sign.
Part E amendment (after first run, 17.10.2026): E2 strict definition gave ~2 events/week; added E2b = top-5% |flow| minutes where price did not move with the flow. Reported as an amendment.
