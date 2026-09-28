# Crossing Memory v1 — acceptance report

Repository: https://github.com/Flyingzaptop/xauusd
Branch: `crossing-memory-v1`
Frozen executable code, pushed before fitting: `a8d7f4c88e92dd3171c4173d01562d52e5d5a682`.
The final report commit is a descendant containing results; its full published SHA is supplied in the GitHub review and delivery message.

## Conclusion

This fixed-budget run provides **no evidence of benefit from the crossing representation or the learned operator**. B is worse than A in mean test log loss; C is worse than B, with a paired weekly interval including zero. C is worse than A with a positive paired weekly interval. The constant predictor has the lowest mean test log loss. Neither ablation shows a resolved positive contribution from transfer or open readout under this protocol. Complex nested-history synthetic learning also failed. These are negative findings for this implementation and budget, not a universal impossibility result for the hypothesis.

## What was implemented and run

A bounded, trainable multiscale Crossing Memory operator with GRU updates of open and completed events, learned child-to-parent transfer, and learned readout. Fixed spatial grids use only observed closes; simultaneous crossings are atomic and processed fine-to-coarse. No reconstructed intraminute price path is used. A uses candle features, B adds causal crossing statistics, C adds learned memory on top of B. Two C ablations remove transfer and open-state readout. A train-frequency/mean-return constant predictor is included.

The complete source dataset contains 3,505,124 M1 candles from 2016-09-04 to 2026-09-04 UTC. Embedded metadata says cTrader Open API demo, UTC candle-open timestamps and tick volume. Volume is not used. Only M1 is accessed by the experiment; no MBO, depth, other instrument or Cohort Memory was added.

The frozen bounded run uses train 2024-01-01 through June 2025, validation July–December 2025, and test January–August 2026. The actual sample counts are 4096/2048/4096, deterministically subsampled from eligible windows. Every model gets exactly 64 contiguous candles and fresh state per window. Targets are ten-minute close direction (zero = not-up) and log return. Windows with gaps or cross-split targets are excluded. Thresholds and normalization come from train only. See `docs/experiment-protocol.md` for exact rules and candidate counts in `reports/market_metrics.json`.

Five trainable variants × three seeds × six epochs = 15 market fits, 90 epochs. One configuration, no hyperparameter search; lowest validation log-loss epoch selected separately for each fit. All market fitting completed before the first test evaluation. Market compute used four CPU threads, despite an available RTX 3060 Laptop GPU. Total recorded training time: 297.0 seconds. Times exclude preprocessing, final evaluation and synthetic work. Auxiliary regression uses a train-scaled squared-error loss and a separate head.

## Test results

Mean across three seeds; ± is sample standard deviation across seeds, not a confidence interval. MAE is in log-return units; training time is mean seconds per seed. Ablations retain parameter tensors but disable their designated paths, so their nominal parameter counts are equal to C.

| Model | Log loss mean ± SD | ROC-AUC | Brier | ECE | Return MAE | Parameters | Train s/seed |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 0.693984 ± 0.000000 | 0.5000 | 0.250418 | 0.0215 | 0.0010057 | 0 | 0.0 |
| A | 0.694401 ± 0.000275 | 0.5101 | 0.250625 | 0.0264 | 0.0010054 | 2210 | 3.1 |
| B | 0.695480 ± 0.000369 | 0.5080 | 0.251154 | 0.0282 | 0.0010076 | 3506 | 3.4 |
| C | 0.696490 ± 0.000986 | 0.5020 | 0.251656 | 0.0384 | 0.0010071 | 8954 | 31.0 |
| C_no_transfer | 0.696483 ± 0.000983 | 0.5022 | 0.251652 | 0.0386 | 0.0010071 | 8954 | 30.4 |
| C_no_open | 0.696321 ± 0.001059 | 0.5017 | 0.251571 | 0.0362 | 0.0010075 | 8954 | 31.2 |

Paired test loss differences (negative favors C), using 2,000 resamples of whole UTC calendar weeks. Seed-mean per-observation loss is paired before block resampling; these intervals measure block variability, not full uncertainty over training or market regimes:

- C - A: +0.0020890; 95% weekly-block interval [+0.0008772, +0.0032908] (36 weeks).
- C - B: +0.0010096; 95% weekly-block interval [-0.0001159, +0.0020930] (36 weeks).
- C - C_no_transfer: +0.0000068; 95% weekly-block interval [-0.0000135, +0.0000266] (36 weeks).
- C - C_no_open: +0.0001684; 95% weekly-block interval [-0.0000739, +0.0004132] (36 weeks).

Per-seed, validation and monthly outcomes and ten-bin calibration counts are published in `reports/market_metrics.json` and the companion CSV. They include negative and unfavorable results without selecting a favorable seed or period.

## Mechanism checks

Ten pytest checks passed before market training: manual return/closure/jump paths, nested grid validation, future perturbation causality, independent windows, prefix routing equality, neural streaming/batch equality, finite gradients including child-to-parent flow, transfer/open readout effects, optimization on short loops, UTC unit handling, gap/split-target exclusion, zero-horizon rejection and known metric values. Several assertions share a test function. The short-loop optimization check proves that parameters can learn; it does not establish successful learning of complex nested structure.

The separately frozen generated-data benchmark completed 15 fits across three seeds, all with finite losses and gradients, in 172.4 seconds. Each fit used 320 train and 160 independently generated test paths. The target depends on the order of finer moves within the last large excursion, with an earlier distractor and random padding.

- ordered_nested_excursions/full: balanced accuracy 0.4777; seeds 0.5000, 0.4425, 0.4906.
- ordered_nested_excursions/no_transfer: balanced accuracy 0.4714; seeds 0.5000, 0.4364, 0.4779.
- ordered_nested_excursions/no_open: balanced accuracy 0.4777; seeds 0.5000, 0.4425, 0.4906.
- ordered_nested_excursions/gru: balanced accuracy 0.4951; seeds 0.5117, 0.4731, 0.5003.
- independent_random_labels/full: balanced accuracy 0.5141; seeds 0.5222, 0.5035, 0.5167.

The complex nested-history learnability check **did not succeed** under the fixed budget. Random-label control was near chance. Full and no-open synthetic results are equal because final open states are zero after the terminal closures; this endpoint cannot diagnose open-readout value. The open-readout unit check does exercise unfinished events, and the market ablation is reported independently. No synthetic result is evidence of market usefulness.

Post-run descriptive routing counts on train windows (`reports/routing-diagnostics.json`) found 15,629 fine closures, of which 4,183 (26.8%) were discarded by an idle parent; at the next link 5,082 of 14,558 (34.9%) were discarded. Open states were available at the final readout in 91.9%, 85.9% and 74.6% of train windows across the scales. Thus market open-readout exposure is substantial, but the transfer convention drops some messages. These counts suggest design limitations; they do not prove the cause of the negative result and were not used to change the model.

## Limitations and interpretation boundary

- This v1 has nested spatial grids, not a stack of strictly temporally contained events. A coarse event can close before a finer one; child closure messages are discarded if the parent is idle. Consequently the run tests this bounded multiscale design, not every possible nested-event memory.
- The floor grid anchored at the initial close is asymmetric around that boundary: an arbitrarily small negative move can enter a negative bin while a positive move must reach the next boundary. This convention was fixed before training, but can influence representation quality.
- Parameters differ between A/B/C; update budgets, optimizer, history and examples are matched. This is not a parameter-matched scaling study. Six epochs and 4096 train windows are a feasibility budget, not evidence of convergence or exhaustive use of the decade-long dataset.
- Failed complex synthetic learning limits mechanistic interpretation. Successful causality and gradient checks do not repair that failure.
- Prior outside-project exposure to these data is unknown. Test is nominally held out within this run, not certified untouched. No architecture or configuration was changed after viewing its predictions.
- No spread, fees, execution, sizing or trading strategy was tested. Predictive metrics cannot establish profitability.
- Raw data provenance/semantics are supported by embedded metadata only; scheduled closures versus missing trading minutes were not independently classified. Reproducing exact bytes requires the owner's existing file.

## Reproduction and provenance

Config SHA-256: `54a43cf25f2072ec047f67a5c30a88afac5eb7fdbcfef466fb9cbcb9e5c0a48a`.
M1 SHA-256: `a9dd281e0100047d56732540168af6e971ef23d858043e3e64991849a2fa6539`.
Commands, environment versions and local artifact retrieval/regeneration are in `docs/reproduction.md`. Data audit and generator hashes are included. Local selected weights and predictions remain excluded from Git; their manifest is `reports/artifact-manifest.json`.

```powershell
python -m pytest -q
python scripts/synthetic_checks.py
python scripts/run_experiment.py --phase fit
python scripts/run_experiment.py --phase evaluate
```

No additional source or architectural phase should start before this stage is reviewed by Astra. A new model after these test results requires a new evaluation claim; this period is now exposed.
