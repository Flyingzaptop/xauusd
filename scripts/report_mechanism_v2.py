"""Assemble complete synthetic mechanism evidence, never train or select models."""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

TASKS=['tiny_hard','easy_closed','medium_closed','hard_closed','hard_open','random_hard']
FROZEN='104022f5cabe9dda532380604a6d86ea584c1080'


def main():
    reports={task:json.loads((Path('reports/v2')/(task+'.json')).read_text()) for task in TASKS}
    assert all(r['complete'] for r in reports.values())
    runs=[run for report in reports.values() for run in report['runs']]
    assert len(runs)==54
    assert all(report['provenance']['commit']==FROZEN for report in reports.values())
    table=[]
    for task,report in reports.items():
        for variant in report['config']['tasks'][task]:
            subset=sorted([run for run in report['runs'] if run['variant']==variant],key=lambda r:r['seed'])
            scores=[r['final_train' if task=='tiny_hard' else 'selected_test']['balanced_accuracy'] for r in subset]
            train=[r['final_train']['balanced_accuracy'] for r in subset]
            params=subset[0]['parameters']
            table.append(f"| {task} | {variant} | {' / '.join(f'{v:.3f}' for v in scores)} | {np.mean(scores):.3f} ± {np.std(scores,ddof=1):.3f} | {np.mean(train):.3f} | {params} |")
    means={}
    for task,report in reports.items():
        means[task]={variant:float(np.mean([r['selected_test']['balanced_accuracy'] for r in report['runs'] if r['variant']==variant])) for variant in report['config']['tasks'][task]}
    total_time=sum(r['training_seconds'] for r in runs)
    probe=json.loads(Path('reports/v2/hidden-state-probe.json').read_text())
    probe_rows=[]
    for task in ['easy_closed','medium_closed','hard_closed','hard_open']:
        pr=[r for r in probe['runs'] if r['task']==task]
        zero_scores=' / '.join(format(r['zero_hidden_at_readout']['balanced_accuracy'],'.3f') for r in pr)
        probe_rows.append(f"| {task} | {np.mean([r['baseline']['balanced_accuracy'] for r in pr]):.3f} | {zero_scores} |")
    text=f'''# Crossing Memory v2: mechanism acceptance report

Repository: https://github.com/Flyingzaptop/xauusd
Branch: `crossing-memory-mechanism-v2`.
Frozen training code/config: `{FROZEN}`.
V1 preserved at `b160b4e523064c6c90990c22c60a5715c6a093aa` on `crossing-memory-v1`.
The final published report SHA is supplied in the GitHub review and delivery message.

## Conclusion for acceptance

The repaired operator satisfies the defined tree/streaming/accounting invariants and learns the intended order-of-nested-history dependency in this finite synthetic family. Full v2 reached 1.000 independent-test balanced accuracy on all four supervised stages and every seed; all three models memorized the separate tiny set. Transfer content matters for completed-event retention (hard closed falls to {means['hard_closed']['no_transfer']:.3f} without it), while open readout matters before the outer event closes (hard open falls to {means['hard_open']['no_open']:.3f} without it). Transfer adds no measured benefit on this particular open task. GRU also solves several tasks and scores {means['hard_closed']['gru']:.3f} on hard closed under the same update/parameter budget; this is not a general superiority claim.

The untouched v1 reaches {means['hard_closed']['v1']:.3f} on hard closed under the new diagnostic schedule. Thus the old failure was not label ambiguity or an intrinsic impossibility of encoding the mapping. Its exact optimization/generalization cause is not isolated by the changed-budget comparison. Structural defects in v1 were independently demonstrated and repaired; synthetic improvement alone cannot be attributed exclusively to those repairs. No market benefit or profitability is established.

## Scope and invariant evidence

No market training was run and no market dataset was used by the v2 runner. January–August 2026 remains exposed. V1 source, market protocol and negative results are preserved unchanged.

The pre-implementation tree specification and manual traces were published in `f2bc339f47b0d7d2022d880168982823f25ed954`. V2 uses a permanent root and immutable event/parent IDs, with genuinely contained lifetimes. Every child closes into its recorded live parent once. An ancestor return forces unfinished descendants to finalize deepest-first at the observed endpoint, explicitly labelled as forced finalization. A jump produces one raw observation and at most one birth. It never invents intraminute prices. Symmetric amplitude comparisons replace floor grids; tolerance scales with the smallest threshold.

Only the deepest live node consumes a raw observation. Completed child state is merged into its parent through a learned GRU. Root consumes raw input only outside a coarse event, avoiding an unrestricted raw-sequence bypass around the tree. Open state and completed state have defined places in the active chain/root. Operational state has at most four nodes and four hidden vectors; an optional unbounded debugging ledger is disabled by default.

All 37 pretraining tests passed: temporal containment and unique ownership, mirrored random trajectories, jumps and simultaneous/forced closures, bounded state, full/step/chunk outputs, serialization resume, full/chunk gradients, causality, finite gradients, oracle labels, disjoint generated samples and matched model sizes. Exact additive-cell instrumentation verifies that the sum of live states contains each distinct raw payload exactly once at every step, with derivative one for every payload. See `docs/mechanism-v2-invariants.md` for the induction argument and its limits. Bounded learned vectors can forget content: message accounting is not an information-theoretic losslessness claim.

## Independent investigation of the v1 failure

An independent parser of observed log-close increments decoded all 1440 original examples and another 4000 sampled paths. Exhaustive enumeration covered 1,081,344 original closed-path derivations with zero opposite-label collisions. Thus ambiguity or a missing observable target does not explain v1's old near-chance result. The original open-prefix construction also exposed the label via its final increment; the v2 open cut was moved three *existing* outer-return observations later before fitting. Exhaustive terminal-feature balancing covered 327,696 observable prefixes. Neither target order nor the original closed hard generator was changed to favor Crossing Memory.

V1 reruns below are new diagnostics, not replacements for its published negative experiment. Their examples, input normalization and update count differ from the old 320-example/10-epoch run. Consequently success here rules out an inherent inability to encode this mapping, but does not isolate a single cause of the old optimizer/generalization failure. It would be incorrect to claim that the routing repair alone caused all improvement, especially if the untouched v1 or GRU also learns.

## Frozen training schedule and complete results

54 completed fits, seeds 7/19/43, one fixed configuration. Tiny memorization: 16 examples, 300 updates. Other tasks: 128 train, 64 validation, 128 test paths, 80 updates, all paths disjoint across splits. Every fit starts from scratch. Easy uses one outer excursion, medium adds an earlier distractor, hard retains the original noisy two-excursion grammar. Open ends at ±4 during the outer return; no end-of-input flush is applied. Random labels are independent of paths in each split.

Tree and GRU have 14,370 and 14,358 parameters (difference 0.084%). Update/data/optimizer budgets are matched. V1 has 8,630 parameters and is a smaller diagnostic comparator. Ablations retain nominal parameter tensors even when a disabled path receives no useful gradient. All computed gradients remained finite; per-parameter actual changes are recorded. The 54 fits record {total_time:.1f} aggregate training seconds, including concurrent CPU contention; these are not isolated speed measurements. Scheduling overlapped independent tasks, with no cross-task model state or selection.

Balanced accuracy; columns list every seed in order 7/19/43. Tiny rows show **training memorization**, not independent test performance. All other rows use the validation-selected checkpoint's independent test. Raw JSON retains every epoch's train/validation loss and balanced accuracy; CSV exports retain both final and selected checkpoint results per fit.

| Task | Variant | Seeds 7 / 19 / 43 | Mean ± SD | Final train mean | Parameters |
|---|---|---:|---:|---:|---:|
{chr(10).join(table)}

Full curves are shown in `plots/` (PNG and SVG), with per-seed traces and their mean. No unfavorable seed or final-checkpoint result is discarded. `plots/seed_metrics.csv` and `plots/aggregate_summary.csv` provide tabular exports. Each task JSON stores its source/config hashes, generated dataset hashes, optimizer update counts, selected epoch, gradients and parameter changes.

![Hard closed learning curves](plots/hard_closed.png)

![Hard open learning curves](plots/hard_open.png)

## What transfer and open memory contribute

Hard closed means: full tree {means['hard_closed']['tree']:.3f}, no-transfer {means['hard_closed']['no_transfer']:.3f}, no-open {means['hard_closed']['no_open']:.3f}, GRU {means['hard_closed']['gru']:.3f}, untouched v1 {means['hard_closed']['v1']:.3f}.

Hard open means: full tree {means['hard_open']['tree']:.3f}, no-transfer {means['hard_open']['no_transfer']:.3f}, no-open {means['hard_open']['no_open']:.3f}, GRU {means['hard_open']['gru']:.3f}.

No-transfer zeros learned child content while retaining structural timing and parent metadata. No-open hides live non-root slots only at readout; completed-event transfers are unchanged. Equality of full and no-open at closed endpoints is expected, not proof that open readout is useless. The separate open task tests that mechanism where the target event remains alive. These contrasts describe this known synthetic family, not every temporal dependency or market process.

The additional fixed-checkpoint intervention in `hidden-state-probe.json` is explicitly **post hoc and nontraining**: it zeros hidden channels at readout while leaving router metadata untouched, checks original predictions against saved results, and reevaluates the same examples. It tests whether these trained predictions rely on learned state rather than metadata alone. It was not used for architecture, seed, checkpoint or generator selection. Exact per-seed drops and checkpoint hashes are preserved in that report.

| Task | Original mean BA | Zero hidden: seeds 7 / 19 / 43 |
|---|---:|---:|
{chr(10).join(probe_rows)}

These models rely on their learned hidden state. Zeroing it is an intervention outside the usual activation distribution and does not prove that a separately retrained metadata-only model could not solve the task.

## Limits and next-stage boundary

- This establishes structural correctness on tested paths and empirical learnability on a finite, exactly specified grammar. Neural compression is not universally lossless, and there is no proof of out-of-distribution generalization.
- Three seeds and 128 independent test examples per task/seed support mechanism diagnosis, not broad ranking of recurrent architectures. These generated paths have no market microstructure.
- Root/child thresholds, one-active-child capacity and forced ancestor finalization are explicit design choices. They preserve temporal containment by construction and may sacrifice other useful organizations of real history.
- New-vs-old results change several conditions; the old negative market result remains valid for its original implementation/protocol. There was no market rerun or attempt to rehabilitate it with these synthetic scores.
- Random-label results and final-vs-selected discrepancies are disclosed. Training fit on random labels is memorization, not evidence of signal.
- Subsequent market protocol must be reviewed separately after acceptance. No new market period is claimed untouched here.

## Reproduction

See `docs/mechanism-v2-reproduction.md`, `docs/mechanism-v2-protocol.md`, `docs/operator-v2.md`, and `docs/v1-synthetic-diagnosis.md`.

```powershell
python -m pytest -q
python scripts/diagnose_v1_synthetic.py
python scripts/run_mechanism_v2.py --task all
python scripts/probe_memory_v2.py
python scripts/plot_mechanism_v2.py
python scripts/report_mechanism_v2.py
python scripts/verify_mechanism_v2.py
```

Training source is frozen at the SHA above; postprocessing scripts live in the final report commit. Local weights remain under `artifacts/mechanism-v2/`, excluded from Git and reproducible without external data. Their sizes/checksums are in `artifact-manifest.json`. Raw v1 data and large external artifacts are not uploaded.
'''
    Path('reports/v2/acceptance-report.md').write_text(text,encoding='utf-8')
    artifacts=[dict(path=p.as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(Path('artifacts/mechanism-v2').glob('*.pt'))]
    assert len(artifacts)==54
    Path('reports/v2/artifact-manifest.json').write_text(json.dumps(dict(training_commit=FROZEN,local_root=str(Path.cwd()),artifacts=artifacts),indent=2))
    Path('reports/v2/aggregate-means.json').write_text(json.dumps(means,indent=2))
    print(json.dumps(means,indent=2))


if __name__=='__main__':
    main()
