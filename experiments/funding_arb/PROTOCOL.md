# Cross-venue funding carry: Binance USD-M vs Hyperliquid — protocol, fixed before looking at results (17.10.2026)
Idea: same coin, two perps. Long the venue with lower funding, short the one with higher -> collect the funding difference,
(near) zero price exposure. Edge exists only if the spread is persistent enough to beat 4 taker fills + basis noise.
Data: Hyperliquid hourly funding (public info API), Binance 8h funding (data.binance.vision), 2023-06..2026-09.
Coins (fixed now): BTC ETH SOL XRP DOGE BNB ADA AVAX LINK LTC DOT TRX BCH NEAR APT ARB OP SUI ATOM AAVE INJ TIA SEI WLD ENA.
Both series converted to per-8h rates aligned to Binance funding times (HL: sum of the 8 hourly rates in the window).
Rule: signal = trailing 7-day mean of (HL - Binance) per-8h spread. Enter when |signal| > 3 bp per 8h (~33%/yr), direction
short the higher-funding venue; exit when |signal| < 1 bp or sign flips. Positions evaluated every 8 h, one unit notional per
leg. P&L = realized next-period spread received; costs 4 x 5 bp = 20 bp per round trip (taker both venues), basis risk
ignored in the main run (daily-close cross-venue basis reported separately). Context 2023-06..2024-12, test 2025-01..2026-09.
Report per coin and pooled: annualized net return on notional per leg, share of time invested, month-clustered t.
