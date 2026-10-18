# Real XAUUSD exact meta-gradient pilot — protocol (declared before any run, 17.10.2026)

Source: Spotware/cTrader `XAUUSD_41.parquet` from the user's Drive (id 1S3AjTWTWzq76bUyHxlgTqNhZzVsOtgwf),
3 505 124 M1 rows, timestamps = UTC candle open, bar t available at t+1 min.

Episodes: one per UTC weekday, session 08:00–18:00. Decisions every 5 minutes (08:00 … 17:55, N=120).
Decision at minute t sees only bars with open < t. Actions WAIT / LONG / SHORT, one position, no new decision while open.
Fixed exit: min(t+120 min, 18:00). Entry = Open of minute t (decision masked to WAIT if that bar is missing).
Exit = Open of first available bar at/after the planned exit within the session, else last available Close ≤ 18:00
(documented simplification; no carrying across nights/weekends).
Money in basis points of entry price; roundtrip cost $0.25/oz converted to bp at entry. Not account % return.

Student input: causal gold-only features (returns 1..1440 min, realized vol, range position, quote age, clock),
standardized with train-period stats, context = last 8 decision steps. No price level.
Teacher additionally sees the realized relative path of the next 24 decision steps (120 min) — training days only.

Splits (calendar): train 2017-01-01..2022-12-31 (inner and outer meta-training); selection 2023-01-01..2023-12-31;
test 2025-07-01..2026-09-04. 2024-01..2025-06 unused (purge + previously-used periods).
Test is a *research* holdout: gold history was examined by earlier work (LightGBM evaluated 2025–2026);
no teacher/meta method in this repo has been evaluated on it. Test is run once with frozen code.

Learner: exact expected return over all WAIT/OPEN trajectories (known historical prices), K=10 SGD steps,
reward RMS-normalized, LR tuned for money on train-period calibration students.
Comparisons on 48 fresh students: learned teacher vs money, money_big (8x inner days), supervised_critic
(ridge regression of money on student features over all train days), constant, shuffled, initial teacher,
plus fixed policies always_long / always_short / flat. Paired bootstrap 95% CI over students.
Success criterion (pre-declared): learned − money CI > 0 AND learned − supervised_critic CI > 0 on test.
Profit claim requires additionally learned > max(flat, always_long, always_short) with CI > 0.
