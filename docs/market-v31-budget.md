# V3.1 prefit volume and runtime budget

This budget was set before inspecting v3.1 calibration or evaluation scores. The exact mask and lane counts are in [budget.json](../reports/v31/budget.json). V3.1 keeps v3's 16 lanes, 128 observations per block, 491,102 train labels and 294 optimizer updates per full epoch. No train candle is subsampled.

| Split | Input rows | Valid targets | Segments | Packed blocks |
| --- | ---: | ---: | ---: | ---: |
| Train | 522,437 | 491,102 | 462 | 294 |
| Selection | 90,583 | 85,536 | 70 | 54 |
| Calibration | 87,305 | 82,081 | 96 | 54 |
| Exposed evaluation | 234,102 | 221,343 | 188 | 119 |

Every train and selection block contains at least one valid target. The selection/calibration split falls between adjacent observed minutes at 2025-10-01 00:00 UTC. Selection's final ten targets are purged, and calibration begins with fresh router and neural state plus its own 64-observation warmup. Train and exposed evaluation cache files are byte-identical to v3. These counts describe the prepared population; they do not reveal class frequencies or evaluation outcomes.

The preregistered controller fits **at least 4 and at most 20 full epochs**. It waits through epochs 1–4, then allows five nonimproving epochs, so the earliest stop is epoch 9. A meaningful improvement for patience requires selection log loss to fall by at least `0.00001` from its reference. The selected `full_best` checkpoint is instead the exact lowest selection loss seen, even if its gain was smaller than `0.00001`; `short_best` is the exact best of epochs 1–4. These may be the same weights. A stop at epoch 20 is a budget cap and does not establish a plateau or adequate optimization. Final reporting must show actual epochs, selection curves, last meaningful improvement, last-five-epoch trend and stopping reason for every fit.

| Work | Per fit, 9–20 epochs | All 15 fits, 9–20 epochs |
| --- | ---: | ---: |
| Optimizer updates | 2,646–5,880 | 39,690–88,200 |
| Supervised train-label visits | 4,419,918–9,822,040 | 66,298,770–147,330,600 |
| Selection forward blocks | 486–1,080 | 7,290–16,200 |

For planning only, the v3 one-worker median block timings applied to v3.1's 294 train and 54 selection blocks imply about **5.4–12.1 aggregate worker-hours** across the fifteen fits. This sums measured training and forward-only selection time over the 9–20 epoch range; it is not a wall-clock forecast. Nine independent one-thread processes can overlap, while CPU contention, checkpoint writes, optimizer restart, calibration and evaluation add variable time. All 30 selected checkpoint identities are reported; identical short/full weights may share inference, so the number of distinct forward passes can be lower. Actual time and work should replace these estimates after completion.

The source file, thresholds, features, model architecture and original v3 results are fixed. This iteration tests training adequacy and positive affine calibration; its January–August 2026 evaluation remains exposed research, not independent confirmation.
