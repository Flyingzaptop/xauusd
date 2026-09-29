# Pretraining volume and execution budget

Fixed before research fitting. Full exact counts and timing calculations:
`reports/v3/frozen-budget.json`; raw timing and long-stream numeric checks:
`reports/v3/speed-stability.json`; normalization/range checks:
`reports/v3/data-preflight.json`.

| Split | Observed candles | Scored labels | Contiguous segments | B16/T128 chunks |
|---|---:|---:|---:|---:|
| Train | 522437 | 491102 | 462 | 294 |
| Validation | 177888 | 167691 | 165 | 108 |
| Exposed evaluation | 234102 | 221343 | 188 | 119 |

All 294 train chunks contain labels. Four full epochs give **1176 optimizer
updates and 1964408 supervised label visits per fit**. Fifteen fits give
**17640 updates, 29466120 label visits and 31346220 raw observation visits**.
No train observations are subsampled. Every valid label is used once per epoch.
The unique train-label count is approximately 120 times v1's 4096 windows.

The measured CPU train-chunk medians are approximately 0.074 s (A), 0.111 s (B),
0.733 s (C), 0.683 s (no transfer), and 0.735 s (no open). Forward-only rates
were measured separately. The budget JSON applies these to all training,
validation and evaluation passes and adds a 20% overhead allowance. This is an
aggregate work estimate, not a promised wall-clock deadline. The fixed 15 fits
will be scheduled with at most five independent one-thread workers; CPU
contention and cache/metric overhead must be acknowledged in final timings.

Tests verified the same observations and labels occur exactly once in each
packed pass. Stream states cross the 128-minute gradient truncation boundary.
On train, a contiguous segment has median length 1319 and maximum length 1380
minutes; resets are caused by observed minute gaps, not by block length.

Before fitting, C was checked on 16384 uninterrupted synthetic observations
and on 16384 real train input rows with the prescribed 17 real resets. The
check used fixed throwaway weights, independent dummy labels and **zero optimizer
updates**. All states/gradients remained finite. Across 127 chunk boundaries,
states were carried, not silently zeroed. Real-input maxima were about 0.989 for
GRU state and 0.955 for tree state; maximum preclip gradient norm was 0.667.
Training metadata clipping affected 45 of 12538488 entries; feature clipping
affected 741 of 2089748 entries. These are numeric and throughput checks, not
market-learning results.
