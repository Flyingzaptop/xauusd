"""Render completed frozen-run results; does not fit or choose models."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    market = json.loads(Path('reports/market_metrics.json').read_text())
    synthetic = json.loads(Path('reports/synthetic_metrics.json').read_text())
    frame = pd.DataFrame(market['metrics'])
    runs = pd.DataFrame(market['runs'])
    rows = []
    for name in ['constant', 'A', 'B', 'C', 'C_no_transfer', 'C_no_open']:
        values = frame[(frame.model == name) & (frame.period == 'test')]
        training = runs[runs.model == name]
        means = values[['log_loss', 'roc_auc', 'brier', 'ece', 'mae']].mean()
        sd = values.log_loss.std() if len(values) > 1 else 0.
        params = int(training.parameters.iloc[0]) if len(training) else 0
        seconds = training.training_seconds.mean() if len(training) else 0.
        rows.append(f'| {name} | {means.log_loss:.6f} ± {sd:.6f} | {means.roc_auc:.4f} | {means.brier:.6f} | {means.ece:.4f} | {means.mae:.7f} | {params} | {seconds:.1f} |')
    intervals = '\n'.join(f"- {c['comparison']}: {c['difference']:+.7f}; 95% weekly-block interval [{c['ci95'][0]:+.7f}, {c['ci95'][1]:+.7f}] ({c['weeks']} weeks)." for c in market['paired_week_bootstrap'])
    syn = '\n'.join(f"- {name}: balanced accuracy {entry['mean_balanced_accuracy']:.4f}; seeds {', '.join(f'{v:.4f}' for v in entry['per_seed'])}." for name,entry in synthetic['summary'].items())
    provenance = market['provenance']
    text = f'''# Crossing Memory v1 — acceptance report

Repository: https://github.com/Flyingzaptop/xauusd
Branch: `crossing-memory-v1`
Frozen executable code, pushed before fitting: `{provenance['code_commit']}`.
The final report commit is a descendant containing results; its full published SHA is supplied in the GitHub review and delivery message.

## Conclusion

This fixed-budget run provides **no evidence of benefit from the crossing representation or the learned operator**. B is worse than A in mean test log loss; C is worse than B, with a paired weekly interval including zero. C is worse than A with a positive paired weekly interval. The constant predictor has the lowest mean test log loss. Neither ablation shows a resolved positive contribution from transfer or open readout under this protocol. Complex nested-history synthetic learning also failed. These are negative findings for this implementation and budget, not a universal impossibility result for the hypothesis.

## What was implemented and run

A bounded, trainable multiscale Crossing Memory operator with GRU updates of open and completed events, learned child-to-parent transfer, and learned readout. Fixed spatial grids use only observed closes; simultaneous crossings are atomic and processed fine-to-coarse. No reconstructed intraminute price path is used. A uses candle features, B adds causal crossing statistics, C adds learned memory on top of B. Two C ablations remove transfer and open-state readout. A train-frequency/mean-return constant predictor is included.

The complete source dataset contains 3,505,124 M1 candles from 2016-09-04 to 2026-09-04 UTC. Embedded metadata says cTrader Open API demo, UTC candle-open timestamps and tick volume. Volume is not used. Only M1 is accessed by the experiment; no MBO, depth, other instrument or Cohort Memory was added.

The frozen bounded run uses train 2024-01-01 through June 2025, validation July–December 2025, and test January–August 2026. The actual sample counts are 4096/2048/4096, deterministically subsampled from eligible windows. Every model gets exactly 64 contiguous candles and fresh state per window. Targets are ten-minute close direction (zero = not-up) and log return. Windows with gaps or cross-split targets are excluded. Thresholds and normalization come from train only. See `docs/experiment-protocol.md` for exact rules and candidate counts in `reports/market_metrics.json`.

Five trainable variants × three seeds × six epochs = 15 market fits, 90 epochs. One configuration, no hyperparameter search; lowest validation log-loss epoch selected separately for each fit. All market fitting completed before the first test evaluation. Market compute used four CPU threads, despite an available RTX 3060 Laptop GPU. Total recorded training time: {runs.training_seconds.sum():.1f} seconds. Times exclude preprocessing, final evaluation and synthetic work. Auxiliary regression uses a train-scaled squared-error loss and a separate head.

## Test results

Mean across three seeds; ± is sample standard deviation across seeds, not a confidence interval. MAE is in log-return units; training time is mean seconds per seed. Ablations retain parameter tensors but disable their designated paths, so their nominal parameter counts are equal to C.

| Model | Log loss mean ± SD | ROC-AUC | Brier | ECE | Return MAE | Parameters | Train s/seed |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Paired test loss differences (negative favors C), using 2,000 resamples of whole UTC calendar weeks. Seed-mean per-observation loss is paired before block resampling; these intervals measure block variability, not full uncertainty over training or market regimes:

{intervals}

Per-seed, validation and monthly outcomes and ten-bin calibration counts are published in `reports/market_metrics.json` and the companion CSV. They include negative and unfavorable results without selecting a favorable seed or period.

## Mechanism checks

Ten pytest checks passed before market training: manual return/closure/jump paths, nested grid validation, future perturbation causality, independent windows, prefix routing equality, neural streaming/batch equality, finite gradients including child-to-parent flow, transfer/open readout effects, optimization on short loops, UTC unit handling, gap/split-target exclusion, zero-horizon rejection and known metric values. Several assertions share a test function. The short-loop optimization check proves that parameters can learn; it does not establish successful learning of complex nested structure.

The separately frozen generated-data benchmark completed 15 fits across three seeds, all with finite losses and gradients, in {synthetic['elapsed_seconds']:.1f} seconds. Each fit used 320 train and 160 independently generated test paths. The target depends on the order of finer moves within the last large excursion, with an earlier distractor and random padding.

{syn}

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

Config SHA-256: `{provenance['config_sha256']}`.
M1 SHA-256: `{provenance['data_sha256']}`.
Commands, environment versions and local artifact retrieval/regeneration are in `docs/reproduction.md`. Data audit and generator hashes are included. Local selected weights and predictions remain excluded from Git; their manifest is `reports/artifact-manifest.json`.

```powershell
python -m pytest -q
python scripts/synthetic_checks.py
python scripts/run_experiment.py --phase fit
python scripts/run_experiment.py --phase evaluate
```

No additional source or architectural phase should start before this stage is reviewed by Astra. A new model after these test results requires a new evaluation claim; this period is now exposed.
'''
    Path('reports/crossing-memory-v1.md').write_text(text, encoding='utf-8')
    artifacts = []
    for file in sorted(Path('artifacts/crossing-memory-v1').glob('*')):
        if file.is_file():
            artifacts.append(dict(path=file.as_posix(), bytes=file.stat().st_size,
                                  sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    Path('reports/artifact-manifest.json').write_text(json.dumps(dict(code_commit=provenance['code_commit'],
                  local_root=str(Path.cwd()), retrieval='Owner local workspace, or regenerate using docs/reproduction.md',
                  artifacts=artifacts), indent=2), encoding='utf-8')
    print('\n'.join(rows))
    print(intervals)


if __name__ == '__main__':
    main()
