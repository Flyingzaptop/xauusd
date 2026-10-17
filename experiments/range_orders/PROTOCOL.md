# Range / resting-order study ("where price will NOT go") — protocol, fixed before looking at results

Question (user's idea): even if direction is unpredictable, can we predict where price will *not* go within a horizon and
monetize it with resting orders (limit / stop / stop-limit) instead of market entries?

Prior, stated up front: range/volatility is predictable (seen in all earlier studies). But on a martingale price, any
entry/exit rule built from resting orders has zero expected gross P&L (optional stopping). Order types change the *shape*
of P&L (win rate vs payoff) and the *fee* (maker vs taker), not the mean. So an edge exists only if touches of the
predicted range edges are followed by non-martingale moves (reversion -> fade with limits, continuation -> stop breakouts),
larger than costs. This study measures exactly that.

Data: Binance USD-M 1-minute klines BTCUSDT, ETHUSDT 2020-01 .. 2026-09 (high/low/close, taker-buy volume).
Split: fit 2020-01..2023-12, validation 2024-01..2024-12 (choose quantile level / exit variant), test 2025-01..2026-09 (once).

Model: LightGBM quantile regression of up-excursion U=log(max high[t+1..t+H]/close_t) and down-excursion
D=log(close_t/min low[t+1..t+H]) for H in {15, 60, 240} min, alpha in {0.7, 0.8, 0.9}. Features at minute close t only:
realized vol over 15/60/240/1440/10080 min, returns 15/60/240/1440, taker-buy share 60/240, distance to rolling 1440 high/low,
volume z-score, hour, weekday. Naive control bands: k*sigma_60 scaled to the same validation fill rate.

Strategies (decision every H minutes, non-overlapping):
- FADE: limit sell at close*exp(qU), limit buy at close*exp(-qD), OCO. Fill only on trade-through by 1 bp (queue-conservative).
  Exit (a) at horizon end, taker; (b) take-profit limit back at decision price / stop-loss at 2x band distance, conservative
  intrabar ordering (stop first). Costs: maker 2 bp entry, taker 5 bp exits (TP limit exit = maker 2 bp).
- BREAKOUT (stop orders): buy-stop at the upper band, sell-stop at the lower band, exit at horizon end; costs taker 5+5 bp
  plus 1 bp stop slippage. Both bands touched in one bar -> assume the adverse fill.
Success = validation-chosen variant net > 0 on test for both symbols with t > 2, and beats the naive-band control.
