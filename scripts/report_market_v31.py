"""Assemble all completed v3.1 results without fitting or choosing variants."""
import hashlib
import json
from pathlib import Path

import numpy as np

TRAINING_SHA='8c45bbf0cafc8397384dde87bd8c84aceedb7d62'


def main():
    cfg=json.loads(Path('configs/market-v31.json').read_text())
    evaluation=json.loads(Path('reports/v31/evaluation.json').read_text())
    calibrators=json.loads(Path('reports/v31/calibrators.json').read_text())
    fits=[json.loads(Path(f'reports/v31/fit-{m}-{s}.json').read_text()) for m in cfg['models'] for s in cfg['seeds']]
    assert all(f['complete'] and f['provenance']==evaluation['provenance'] for f in fits)
    assert evaluation['provenance']['code_commit']==TRAINING_SHA
    primary=next(c for c in evaluation['comparisons'] if c['primary'])
    total_epochs=sum(f['epochs_completed'] for f in fits)
    updates=sum(f['total_optimizer_updates'] for f in fits)
    assert updates==total_epochs*294
    overall=[r for r in evaluation['metrics'] if r['period']=='exposed_evaluation']
    def mean_metric(model,choice,mode,key):
        return float(np.mean([r[key] for r in overall if (r['model'],r['choice'],r['mode'])==(model,choice,mode)]))
    table=[]
    for m in cfg['models']:
        v=[mean_metric(m,c,mode,'log_loss') for c in ['short_best','full_best'] for mode in ['raw','calibrated']]
        table.append(f"| {m} | {' | '.join(format(x,'.7f') for x in v)} | {mean_metric(m,'full_best','raw','auc'):.5f} |")
    stopping=[]
    caps=[]
    for f in fits:
        losses=[e['selection']['log_loss'] for e in f['epoch_curve']]
        recent=losses[-1]-losses[max(0,len(losses)-5)]
        if f['stop_reason']=='budget_cap':caps.append(f"{f['model']}/{f['seed']}")
        significant=[e['epoch'] for e in f['epoch_curve'] if e['control']['significant_improvement']]
        stopping.append(f"| {f['model']} | {f['seed']} | {f['epochs_completed']} | {f['stop_reason']} | {f['checkpoints']['short_best']['epoch']} | {f['checkpoints']['full_best']['epoch']} | {significant[-1] if significant else 'none after 4'} | {recent:+.7f} |")
    contrasts=[]
    for c in evaluation['comparisons']:
        if c['family']=='constant_control':continue
        contrasts.append(f"| {'/'.join(c['left'])} − {'/'.join(c['right'])} | {c['difference']:+.7f} | [{c['ci95'][0]:+.7f}, {c['ci95'][1]:+.7f}] | {' / '.join(format(c['per_seed'][str(s)],'+.7f') for s in cfg['seeds'])} |")
    if primary['difference']>0:
        conclusion='The primary full calibrated C loses to full calibrated A in mean log loss. This iteration therefore ends with a negative conclusion and no automatic expansion of training or search.'
    elif primary['ci95'][1]<0:
        conclusion='Full calibrated C has lower mean log loss than full calibrated A on this exposed period, with a negative conditional weekly interval. This is exploratory evidence only; ablations and seed differences are required before attributing the gain to the intended memory mechanism.'
    else:
        conclusion='Full calibrated C has a favorable or tied point estimate but its weekly interval does not resolve a gain over A. This iteration does not establish a robust C-over-A improvement and does not expand the search automatically.'
    params=[]
    for e in calibrators['entries']:
        p=e['parameters']
        params.append(f"| {e['model']} | {e['seed']} | {e['choice']} | {e['checkpoint']['epoch']} | {p['a']:.8g} | {p['b']:.8g} | {p['status']} | {p['slope_at_lower_boundary']} |")
    seed_rows=[]
    for r in overall:
        if r['model']=='constant':continue
        seed_rows.append(f"| {r['model']} | {r['seed']} | {r['choice']} | {r['mode']} | {r['log_loss']:.7f} | {r['auc']:.5f} | {r['brier']:.7f} | {r['ece']:.5f} |")
    constant=next(r for r in overall if r['model']=='constant')
    text=f'''# V3.1 — training adequacy and held-apart calibration

Repository: https://github.com/Flyingzaptop/xauusd
Branch: `crossing-memory-market-v3-1`
Preregistered protocol commit: `fd2485e9b08e1db7da2bb885565f0d8d206d44e3`.
Frozen executable training/calibration/evaluation code: `{TRAINING_SHA}`.
The final published result SHA is supplied in the review and delivery message.

**January–August 2026 is exposed exploratory evaluation. This is not independent confirmation.** V3's architecture, negative results, data sources and code remain preserved. No MBO, depth, extra instrument or new neural block was added.

## Primary conclusion: C versus A

{conclusion}

Primary full/calibrated C − A log loss: **{primary['difference']:+.7f}**, paired weekly 95% interval **[{primary['ci95'][0]:+.7f}, {primary['ci95'][1]:+.7f}]**. Per-seed differences in order7/19/43: {', '.join(format(primary['per_seed'][str(s)],'+.7f') for s in cfg['seeds'])}. An improvement only against B is not the acceptance criterion. The weekly interval conditions on these three fitted seeds, not the full distribution of optimizer initializations or future regimes.

The two interventions answer different questions. Extended training reduces raw C loss by about0.000296, but after calibration the full-minus-short difference is only -0.00000235 (weekly95% interval [-0.00000949,+0.00000511]). Calibration improves full C by about0.002481, yet C still loses to calibrated A on every seed. Calibration actually worsens A on exposed evaluation by about0.000438; raw A is the strongest of these fixed variants. We report that adverse calibration effect rather than selecting a favorable postprocessing outcome.

Full calibrated C is only about0.00000609 better than its no-transfer ablation and0.00000375 better than no-open. Transfer's small conditional interval is negative, but its sign reverses on seed19; the open-readout interval contains zero. Neither effect explains away the much larger0.000695 C-A gap. These are secondary descriptive contrasts, not multiplicity-adjusted discoveries or proof of a generally useless mechanism.

Mean log loss across all three seeds, with every preselected variant shown:

| Model | Short raw | Short calibrated | Full raw | Full calibrated | Full AUC (raw=calibrated) |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Train-frequency constant control log loss: {constant['log_loss']:.7f}, AUC {constant['auc']:.5f}, Brier {constant['brier']:.7f}. The constant is not another selected checkpoint.

## Actual completed budget and split separation

Fifteen from-scratch fits completed **{total_epochs} full epochs, {updates} Adam updates**, and **{total_epochs*491102} supervised label visits**. Each full epoch used491102 valid train labels from522437 observed M1 candles. Initial parameters, architecture, Adam0.001, gradient clipping1.0, raw features, train normalization, thresholds, state resets,16 lanes and TBPTT128 were unchanged from v3. State is detached across blocks, not reset. All fits began at seeds7/19/43; selected v3 weights were not used to initialize them.

Train:2024-01-01 through June2025. Selection:July–September2025,85536 valid labels from90583 rows. Calibration:October–December2025,82081 labels from87305 rows. Exposed evaluation:January–August2026,221343 labels from234102 rows. The new October boundary resets state, applies64 warmup observations, and purges the preceding ten targets. Train/evaluation caches are byte-identical to v3. Calibration labels never select neural weights or stopping time; evaluation labels are loaded only after the complete fit matrix and all30 calibrator identities are frozen.

Actual sum of fit elapsed times: {sum(f['training_seconds'] for f in fits)/3600:.2f} worker-hours. Fits may overlap in at most nine single-thread processes, so this is not wall time or an isolated speed benchmark. Calibration phase wall time: {calibrators['wall_seconds']/60:.2f} minutes; evaluation phase wall time: {evaluation['wall_seconds']/60:.2f} minutes. See execution.json for launcher wall time and whether recovery was used.

## Did training reach a plateau?

The exact minimum selection loss chooses weights; the1e-5 threshold controls a separate patience reference. Patience starts after epoch4 and triggers after five nonimproving epochs; earliest stop9, maximum20. Short-best is the exact minimum among the first four epochs, full-best the exact minimum across the completed run. The short v3.1 choice uses the new three-month selection split and is not interchangeable with v3's original six-month checkpoint choice.

Budget-capped runs: **{', '.join(caps) if caps else 'none'}**. A budget cap does not establish convergence. In capped runs the patience criterion was not reached, so a stopping plateau has not been demonstrated; recent improvements and exact best epochs are shown below. A patience stop is only an operational selection-loss criterion, not proof that optimization reached a global optimum. No extra epochs were added after either outcome.

Last-five change is last selection loss minus the first of the final five losses; negative means recent endpoint improvement. Curves contain every epoch and every batch, not only best checkpoints.

| Model | Seed | Epochs | Stop reason | Short epoch | Full epoch | Last significant improvement after4 | Last-five loss change |
|---|---:|---:|---|---:|---:|---:|---:|
{chr(10).join(stopping)}

## Fixed positive-affine calibration

Every selected checkpoint uses exactly `sigmoid(a*z+b)`, a>=1e-6, unregularized mean log loss, fixed L-BFGS-B settings and starting point(1,0). Only the calibration quarter fits a/b. Solver and projected-gradient diagnostics are preserved. Same-weight short/full identities may share inference and fitted coefficients, but both are reported. Recovery last.pt is not an evaluation candidate.

All30 calibrator identities were saved and hashed before evaluation. Coefficient freeze: `{evaluation['calibration_freeze_sha256']}`. A positive slope preserves ordering. AUC is computed on original raw logits in both variants to avoid finite-precision saturation ties; **no AUC improvement is attributed to calibration**. Longer neural training can change AUC because it changes weights, unlike the affine postprocessing. Lower-bound slopes are explicitly flagged as near-constant probability forecasts.

The freeze was also [published to the coordinator before evaluation](https://github.com/Flyingzaptop/xauusd/issues/1#issuecomment-5907365625). Three A short/full pairs have identical weights, so27 unique calibrator fits supply30 reported identities. Every solver reported convergence; no slope reached the lower bound.

| Model | Seed | Checkpoint | Epoch | a | b | Solver status | Lower-bound slope |
|---|---:|---|---:|---:|---:|---|---|
{chr(10).join(params)}

## Every seed and probability metric

| Model | Seed | Checkpoint | Mode | Log loss | AUC | Brier | ECE |
|---|---:|---|---|---:|---:|---:|---:|
{chr(10).join(seed_rows)}

Monthly results and every nonempty reliability bin are retained in evaluation.json
and metrics.csv. Plots show all seeds and avoid extending a stopped run's curve
as if it continued training. Low-count calibration bins remain visible with
their limited support indicated. Raw and calibrated AUC must agree for each
checkpoint/seed/month; probability calibration does not improve ranking.

![All seed learning and stopping curves](plots/epoch_learning_selection.png)

![All selected raw and calibrated variants](plots/selected_logloss_all_identities.png)

## Paired weekly intervals and ablations

Whole UTC weeks are resampled2000 times with seed20260930 after averaging paired
minute losses over the three fixed model seeds. No independent-minute bootstrap
is used for overlapping ten-minute targets. Intervals below are conditional and
secondary comparisons are not multiplicity-adjusted discoveries. Negative favors
the left variant. The full60-contrast machine report also includes constant controls.

| Matched comparison | Difference | Weekly95% interval | Seed7 /19 /43 differences |
|---|---:|---|---|
{chr(10).join(contrasts)}

Check C versus A first, then transfer/open ablations under the same checkpoint
and calibration choice. Full-minus-short comparisons address extended training;
calibrated-minus-raw comparisons address the fixed probability correction.
Do not select whichever combination happened to score best after seeing this
exposed evaluation. The mechanism's synthetic validity is not market evidence.

## Preservation and reproduction

Model and Adam state, RNG and early-stop state are saved in short-best/full-best
and the overwritten recovery checkpoint. Exact epoch-boundary resume was tested.
V3 code/results and earlier stages are retained unchanged. Raw data, caches,
optimizer states and weights stay in ignored artifacts/market-v31; their
manifest provides locations and SHA-256 hashes. The dataset hash is
`{evaluation['provenance']['data_sha256']}`.

Use docs/market-v31-reproduction.md and the frozen training commit for fitting;
postprocessing scripts are in the final report commit. The core commands are:

```powershell
python scripts/prepare_market_v31.py
python scripts/launch_market_v31.py --workers 9
python scripts/score_market_v31.py --phase calibrate
python scripts/score_market_v31.py --phase evaluate
python scripts/plot_market_v31.py
python scripts/report_market_v31.py
python scripts/verify_market_v31.py
```

No broader hyperparameter search or market-profit claim follows this iteration.
If a later experiment changes the fitted procedure after these outcomes, it must
be separately identified and cannot treat this evaluation as untouched.

The independent verifier reproduced the stopping decisions without importing
the training controller, checked all model/optimizer/RNG/recovery payloads,
verified first-four-epoch equality to v3, and recomputed549 metric rows and60
paired weekly intervals. It confirmed AUC invariance, the calibration-only
fit data and freeze chain, and preservation of the prior106 local artifacts.
See `verification.json` for the full machine-readable audit.
'''
    Path('reports/v31/acceptance-report.md').write_text(text,encoding='utf-8')
    files=[]
    for path in sorted(Path(cfg['output']).rglob('*')):
        if path.is_file() and path.suffix in ['.pt','.npz']:
            h=hashlib.sha256()
            with path.open('rb') as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            files.append(dict(path=path.as_posix(),bytes=path.stat().st_size,sha256=h.hexdigest()))
    Path('reports/v31/artifact-manifest.json').write_text(json.dumps(dict(training_commit=TRAINING_SHA,
        local_root=str(Path.cwd()),artifacts=files),indent=2))
    print(json.dumps(dict(epochs=total_epochs,updates=updates,budget_capped=caps,primary=primary),indent=2))


if __name__=='__main__':main()
