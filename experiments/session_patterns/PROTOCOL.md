# Session / time-window patterns in XAUUSD M1 — protocol (declared before running, 17.10.2026)

Data: Spotware `XAUUSD_41.parquet`, M1, 2016-09 … 2026-09, UTC candle-open timestamps.
Window return: log(Open[first bar at/after end] / Open[first bar at/after start]) in bp, both bars within 5 min of the
nominal time and on the same trading day; otherwise the day is skipped. Weekdays only.

## Windows
A. Sessions (fixed UTC, as given by the user): Sydney-only 23–00 (previous evening), Sydney–Tokyo 00–07, Tokyo–London 07–09,
   pure London 09–12, London–NY 12–16, pure NY 16–20, late NY 20–21.
B. Anchored events (local time, DST-aware, converted to UTC per day), each with pre (−30→0) and post (0→+15, 0→+30, 0→+60):
   SGE AM benchmark 10:15 Beijing, SGE PM benchmark 14:15 Beijing, London open 08:00 London, LBMA AM 10:30 London,
   US data 08:30 New York, US equity open 09:30 New York, NY 10:00 data, LBMA PM 15:00 London, WM/R 16:00 London,
   COMEX settlement 13:30 New York, NY close 17:00 New York.

## Tests
1. Drift: mean window return.
2. Session→later-session (same day): sign(return of A) predicts B.
3. Session open→rest: return of the first 30 min of a session predicts the rest of that session.
4. Event pre→post: pre-window return predicts the post-window return (continuation or reversal).

## Out-of-sample rule (no in-sample selection)
Walk-forward by year 2018…2026: the trading direction for year Y (fixed sign for drift; slope sign for predictive tests)
is estimated from all prior years only. Trade = one position per day per test, cost $0.25/oz roundtrip (in bp at entry).
A pattern is called a candidate only if walk-forward net PnL is positive in ≥ 7 of 9 years AND the t-stat of daily walk-forward
net PnL ≥ 3.3 (≈ Bonferroni for ~100 tests). Drift results are additionally compared with the gold buy-and-hold drift over the
same duration (bull market 2016–2026), so that "always long" in a window is not mistaken for timing.
All windows/tests above are fixed; no new windows are added after seeing results without being reported as such.

# Part 2 — event structure (declared before running, 17.10.2026, after the user clarified the question)
Goal: recurring *structure* around daily events, not unconditional drift.
Events: user session boundaries 00,07,09,12,16,20 UTC + DST-aware anchors (SGE 10:15/14:15 BJ, London open 08:00, LBMA 10:30/15:00,
US 08:30/09:30/10:00 NY, COMEX settle 13:30 NY, WM/R 16:00 London, CME reopen 18:00 NY).
A. Profile (descriptive): per-minute |return| and tick volume from −60 to +90 min vs the −60…−31 baseline: which events are real activity spikes.
B. Conditional pre→post: z = pre-30-min return / its trailing 250-day std at the same event; bins (<−2, −2…−1, −1…1, 1…2, >2);
   trade only |z|>1 and |z|>2, direction = sign of the past mean post-return in that bin (walk-forward); horizons +15/+30/+60.
C. Ranges: (1) Asian range 00–07 UTC, first break 07–10 UTC → continuation/reversal over +30/+60/+120 min and to 16:00;
   (2) "sweep": break then close back inside within 30 min → trade from re-entry; (3) opening-range (first 30 min) break after
   London open 08:00 LDN and US 08:30 NY, held to 16:00 / 20:00 UTC; (4) previous-day high/low taken in London/NY → continuation/reversal.
   Entry = Open of the bar after the signal bar (no same-bar fill). (5) Descriptive: time-of-day of daily high and low.
D. Volatility (descriptive): corr(pre-60-min realized vol, post-60-min |return|).
Same walk-forward (2018–2026), cost $0.25/oz, candidate = t≥3.3 and ≥7/9 positive years.
