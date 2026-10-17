# Mean reversion to lines and support/resistance levels — XAUUSD M1 2016–2026 (declared before running, 17.10.2026)

Execution model: one position at a time per test; target and stop are FIXED prices set at entry; M1 high/low decide which is hit first;
if both inside the same bar → counted as a loss (conservative); timeout 240 min or session gap > 30 min → exit at last close.
Cost $0.25/oz per round trip. Results reported per year and overall.

A. Reversion to a line. Lines: session VWAP (tick-volume weighted, reset at the 1st bar after a ≥30-min gap), SMA20, SMA100 (=SMA20 on M5),
   SMA300 (=SMA20 on M15). sigma = rolling std of (close − line) over the same window (Bollinger width). Entry at the close when
   |close − line| ≥ k·sigma, k ∈ {2, 3}, trade toward the line. Target = line value at entry (distance d). Stop distance a ∈ {d, 2d}.
   Random-walk null: P(target first) = a/(a+d) (0.50 for a=d, 0.667 for a=2d). Edge requires P above the null AND positive mean PnL after cost.
   Also reported (descriptive only): how often the *moving* line is crossed later — to show why "always crosses" is not an edge.
B. Levels. Per day (known before the day starts): previous-day high, low, close; previous-day tick-volume POC ($0.5 bins);
   round numbers every $10; swing highs/lows (M15 5-bar fractals in the previous 2 days, last 3 of each).
   Event: first touch of a level today, approached from one side (previous close on that side). Bet on a bounce: limit entry at the level,
   target b away in the bounce direction, stop b beyond the level, b ∈ {1, 2}× mean 1-min range of the last 60 min (≥ $0.30).
   PLACEBO: the same procedure on fake levels = each real level shifted by ±U($3, $8). Edge requires bounce rate real > placebo
   (difference with bootstrap CI excluding 0) AND positive mean PnL after cost.
