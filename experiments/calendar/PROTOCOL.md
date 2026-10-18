# Crypto calendar effects — protocol, fixed before looking at results (18.10.2026)
Data: Binance USD-M 1m BTCUSDT, ETHUSDT 2020-01..2026-09.
## K1 "CME gap fill"
CME BTC futures stop Fri 21:00 UTC (20:00 in US summer time — use 21:00 UTC Nov-Mar, 20:00 Apr-Oct) and reopen Sun 23:00 / 22:00 UTC.
Gap = log(P_reopen / P_friday_close) on Binance. If |gap| > g (g in {0.5%, 1%, 2%}), trade TOWARD the Friday price at
the reopen: take-profit at the Friday close price, exit at Fri close of the same week otherwise; no stop (variant with a
stop at 2x gap). Costs 10 bp round trip. Context 2020-2023, test 2024-01..2026-09.
## K2 hour-of-week seasonality
168 hour-of-week buckets; each calendar year Y, sign of each bucket = sign of its mean return over all years < Y
(expanding, from 2020); trade the 24 buckets with the largest |t| over years < Y (1 h holds, non-overlapping by design).
Costs 10 bp per trade. Report per year 2022-2026 and pooled 2024-2026.
Success for either: test mean net > 0 with t > 2, positive in most test years.
