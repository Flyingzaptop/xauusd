# Daily momentum in crypto perps — protocol, fixed before looking at results (18.10.2026)
Data: Binance USD-M daily closes from 1m (00:00 UTC), BTC/ETH 2020-01..2026-09, 30 alts (Part G universe) 2023-01..2026-09.
Funding paid/received on positions included for BTC/ETH (funding files available); alts: funding ignored (reported caveat).
M1 time-series momentum (fixed rules, no fitting): (a) long/short sign of trailing 28-day return; (b) long when close > 50-day SMA,
short below; (c) long-only version of (b) (flat below). Rebalance daily at 00:00 UTC using yesterday's close, trade at next open
(=same price here, daily close), costs 10 bp per position change. Per-asset vol targeting: position = 1 / (20-day realized vol
annualized / 50%), capped at 2.
M2 cross-sectional momentum on the 30 alts + BTC + ETH: weekly (Mon 00:00), long top 5 by trailing 28-day return (skip last day),
short bottom 5, equal weight, costs 10 bp per traded leg.
Periods: context 2020-2023 (BTC/ETH) / 2023-24 (alts); test 2024-01..2026-09 (BTC/ETH) / 2025-01..2026-09 (alts).
Report annualized return, Sharpe, max drawdown, by year, vs buy-and-hold. Success: test Sharpe > 0.8 and better than buy-and-hold
(TSMOM) / > 0 with t > 2 (XS).

## Amendment M3 (declared after seeing M1/M2, before running)
Robustness of the SMA trend rule, no selection: SMA lookbacks {20, 100, 200} besides 50, same vol targeting and costs; alts WITH
funding (Binance monthly fundingRate files for the 30 alts); combined portfolio = equal risk thirds BTC / ETH / alt basket
(each sleeve already vol-targeted), annual vol target 20% at portfolio level (scaled by trailing 60-day realized portfolio vol).
Report the same statistics; the claim "trend following works" requires all lookbacks positive in the test period.

## Amendment M4 (declared before running): untouched history 2017-08..2019-12 on Binance SPOT daily klines (perps did not
exist), BTCUSDT and ETHUSDT, same SMA rules {20,50,100,200}, same vol targeting, 10 bp costs, no funding (shorting assumed at
spot price). Pass = SMA50 Sharpe > 0.5 on both and the BTC+ETH 50/50 combination beats buy-and-hold Sharpe.

## Amendment M5 (declared before running): survivorship-free alt trend
Universe each month = top 20 USDT perps by trailing-30-day quote volume at the month start, excluding BTC/ETH, from ALL symbols
ever listed (delisted included), using Binance daily klines. Same SMA rules (20/50/100 ensemble and SMA50 alone), per-asset vol
target, equal-risk basket, 10 bp costs, funding ignored (caveat). A coin that stops trading is closed at its last price.
Periods 2021-2023 / 2024-2026-09. Pass = test Sharpe > 0.5 for the SMA50 rule.

## Amendment M6 (declared before running): controls for the candidate spec
(a) same construction with signal = +1 always (vol-targeted long-only) and with signal = sign of a circularly shifted SMA signal
(shift by 180..2000 days, 50 shifts) -> distribution of Sharpe under "same exposure profile, no timing";
(b) stationary block bootstrap (30-day blocks, 5000 resamples) of the candidate's daily returns -> 90% CI of Sharpe.
