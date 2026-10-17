# New USD-M perpetual listings on Binance — protocol, fixed before looking at results (17.10.2026)
Hypothesis: newly listed perps drift down over the following weeks (hype fades, token unlocks, shorting finally possible).
Universe: every USDT-margined perp in data.binance.vision (delisted ones included -> no survivorship bias); listing day L =
first daily kline. Listings 2021-01..2026-08.
Trade: SHORT at the open of day L+1 (skip the partial first day), exit at the open of day L+1+h, h in {7, 14, 30}.
P&L (bp of notional) = -(log return) + sum of funding rates over the holding (shorts receive positive funding, pay negative)
- costs 15 / 30 bp round trip (new listings have wide spreads). Also BTC-hedged: minus BTC return over the same window.
Inference: clustered by listing month (mean of monthly means, t over months). Report mean, median, share positive, worst
trade (squeeze risk), by year. Context 2021-2023; test 2024-01..2026-08.
Success: test mean net of 30 bp > 0 with month-clustered t > 2 and median > 0, raw and hedged.

## Amendment H2 (declared after seeing H, before running): the log-return P&L in H overstated shorts; correct P&L is the
simple short return -(P1/P0-1). H showed positive medians but huge squeeze tails. H2 adds a protective buy-stop:
stop at +s above entry, s in {20%, 30%, 50%, 100%}, checked on daily highs; fill = max(stop, that day's open) * (1+2% slippage).
Costs 30 bp, funding as in H. Same periods; also an equal-weight portfolio (each new listing gets 1 unit, P&L summed per
calendar month) with monthly Sharpe and max drawdown. Success: 2024-26 mean net > 0, month-clustered t > 2 for some s
that was ALSO positive in 2021-23 (choose s on 2021-23 only).
