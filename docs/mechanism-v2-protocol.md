# V2 synthetic-only protocol, fixed before training

No market files or market training. The exposed January–August 2026 test will not be reused. V1 files and results remain intact. Architecture specification and hand traces were pushed as f2bc339f47b0d7d2022d880168982823f25ed954 before implementation.

## Diagnosis and schedule

First validate the old task with an independent parser of observed increments, not router events or generator labels. Exhaustively check grammar collisions and sample observed float32 prices. Check new task labels against the independent oracle and enforce exact path-disjoint train/validation/test splits. Generated repeats with independent labels are not a generalization test; hash exclusion avoids them.

Fixed seeds 7,19,43. One configuration in configs/mechanism-v2.json; no favorable seed selection. Tiny memorization uses 16 distinct original hard closed paths, 300 full-batch updates, and reports training fit (its evaluation set is deliberately identical). Then use independent sets of 128 train, 64 validation, 128 test examples for 80 full-batch updates. This is a mechanism diagnostic budget, not a converged benchmark. Each task initializes fresh weights; increasing complexity is a sequence of separate experiments, not curriculum transfer or test-driven adaptation.

- easy_closed: one original outer excursion, six neutral loop pairs around it; recover the finer excursion order.
- medium_closed: two outer excursions (first is a distractor), four neutral loop pairs distributed across gaps.
- hard_closed: unchanged v1 grammar: two excursions and ten neutral loop pairs, 73 observed closes.
- hard_open: original hard path truncated after the first three steps of the final seven-step outer return; leading repeated-close padding produces 69 observations while keeping the outer event open at displacement ±4. Both classes receive the same construction, without consulting router success. Independent parser must recover the target. Pretraining inspection rejected a cut immediately after the finer structure because its final increment exposed the label through endpoint features; the later cut removes that particular shortcut without inventing new movements. This correction precedes all model fitting.
- random_hard: hard closed paths with fresh independent random train/validation/test labels; near-chance generalization is the negative control.

Tiny and hard closed also run the untouched V1 model under the same current data/update budget, labelled as new diagnostic runs, not replacements of old results. Full v2 and GRU run all tasks. Transfer-content and open-readout ablations run hard closed and hard open. Closed-endpoint full/no-open equality is expected if all events close, and must not be sold as a diagnostic of open readout. Open endpoint tests address that separately.

## Equal inputs and budgets

All variants see the same observed log-close sequence through base features [one-step change, displacement from first close / 10] in synthetic step units. V2 also routes that same history with the specified tree; v1 retains its old [1,3,6] grid. No target or latent generator annotation is given to either model. GRU hidden width is chosen algorithmically to minimize total parameter-count difference from the full tree model, before labels are evaluated. Report exact counts, time and mismatch. V1 size stays its documented hidden width 12 and is a diagnostic, not a capacity-matched contender. Models have the same 32-unit tanh classification head and optimizer schedule; updates, clipping, data and seed budgets are matched per task.

Training uses Adam 0.003, gradient clipping 5, one CPU thread, deterministic PyTorch operations. Record train/validation loss and balanced accuracy at every epoch, finite gradient checks and parameter changes. Independent test is evaluated in one post-training phase per variant, for both the final and validation-best checkpoint. Main comparative results use validation-best, while tiny memorization uses the final checkpoint. No architecture/generator/hyperparameter changes are permitted in response to these outcomes. Source, configuration and dataset hashes must accompany results.

## Required conclusions and failure localization

1. Exact routing accounting and oracle reconstruction establish symbolic routing/target correctness; bounded neural vectors do not guarantee information-theoretic losslessness.
2. Tiny-set fit distinguishes failure to optimize even a small known mapping from poor independent generalization. Compare untouched v1 and matched GRU.
3. Separate easy, distractor, noisy closed and open tasks identify where learning degrades. Plot every seed and disclose all outcomes.
4. Compare ablations on appropriate endpoints. No-transfer still retains structural timing/metadata and cannot be interpreted as removing every possible clue about events.
5. If the complex task fails, stop the progression to market work. Diagnose evidence-supported ambiguity, optimization, generalization, state ownership or information loss; do not claim an unsupported single cause or expand market data.

Any additional fixed-input probes after outcomes are descriptive diagnostics and must be labelled post hoc, with no model fitting or hidden retuning.
