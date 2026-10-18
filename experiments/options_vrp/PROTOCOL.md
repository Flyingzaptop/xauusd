# Option-selling study ("where price will NOT go" monetized directly) — protocol, fixed before looking at results

Why: a range forecast pays nothing on a perp (zero-drift martingale), but it is exactly what an option seller is paid for.
If implied vol (what buyers pay) exceeds realized vol and/or our range forecast is better than the market's, selling
options with strikes outside the predicted range earns the variance risk premium. This is a risk premium with crash
(tail) risk, not a free edge; results must show tails and drawdowns, not only means.

Data: Deribit public history (history.deribit.com): every BTC/ETH option trade 08:00-10:00 UTC each day 2021-01..2026-10
(price in coin, mark, IV, index, taker direction); Deribit delivery (settlement) prices. Binance 1m klines for forecasts.
Instruments: daily options expiring next day 08:00 UTC (~22-24 h to expiry).
Fills (we are taker): sell = VWAP of taker-SELL trades in that instrument in the window (they hit the bid); buy (wings) =
VWAP of taker-BUY trades (they lift the ask). Instruments without such trades that day are skipped.
P&L per unit underlying notional, USD, premium converted at entry index: prem*S0 - intrinsic(S_T), in bp of S0.
Fees: 3 bp of underlying per leg capped at 12.5% of the option price, plus 1.5 bp delivery fee per leg (same cap).
Strategies: (S0) short ATM straddle every day; (S1) short strangle at strikes nearest k*sigma_hat (sigma_hat = forecast
24 h vol), k in {0.5,1,1.5}; (S2) S0/S1 only when entry IV > forecast vol*(1+m), m in {0,0.1,0.2}; (S3) iron fly / condor:
S0/S1 plus bought wings at 2*k*sigma_hat. Forecast: HAR (daily/weekly/monthly RV from 1m) fitted on fit period only.
Split: fit 2021-2023, validation 2024 (choose k, m, wings), test 2025-01..2026-10 (once).
Report: mean bp/day, t, Sharpe (annualized), worst day, max drawdown, by year, for BTC and ETH.
