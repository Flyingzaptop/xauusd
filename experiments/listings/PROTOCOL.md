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
