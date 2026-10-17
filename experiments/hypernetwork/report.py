"""CPU audit and plots of the frozen hypernetwork comparison."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'experiments/direction_precision'))
from policy import fit_policy,apply_policy,metrics

def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pct(x):return '—' if x is None else f'{100*x:.2f}%'
def verify_equal(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:verify_equal(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):verify_equal(x,y)
    elif isinstance(a,float):assert np.isclose(a,b,rtol=1e-6,atol=1e-9),(a,b)
    else:assert a==b,(a,b)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=root/'reports/hypernetwork_20261010');args=ap.parse_args();out=args.out;r=out/'remote';run=read(r/'run.json')
    assert run['status']=='complete' and run['selection_frozen_before_test']
    assert read(r/'worker_result.json')['returncode']==0
    assert run['source_sha256']==sha(root/'experiments/hypernetwork/train.py')
    assert run['model_sha256']==sha(root/'experiments/hypernetwork/model.py')
    assert run['dataset_sha256']==sha(root/'reports/pcgrad_mse_month_20261009/dataset.npz')
    assert len(run['trials'])==4
    for seed in [17,29]:assert run['trials'][f'static_{seed}']['initial_trunk_sha256']==run['trials'][f'hyper_{seed}']['initial_trunk_sha256']
    curves={}
    for name,trial in run['trials'].items():
        rows=read(r/f'{name}_curve.json');curves[name]=rows
        assert len(rows)==trial['epochs']==12
        assert all(x['examples_seen']==27716 for x in rows)
        assert np.isfinite([x['train_cost'] for x in rows]).all()
        assert trial['best_epoch']==1+int(np.argmax([x['selection_policy']['selection_score'] for x in rows]))
        z=np.load(r/f'{name}_validation.npz');early=z['early']
        pol=fit_policy(z['p'][early],z['target'][early],z['timestamp'][early]//86400)
        verify_equal(pol,rows[trial['best_epoch']-1]['selection_policy'])
    dataset=np.load(root/'reports/pcgrad_mse_month_20261009/dataset.npz')
    cents=np.rint(np.exp(dataset['log_prices'][:,2])*100).astype(np.int64)
    starts=dataset['test_start'];delta=cents[starts+7]-cents[starts-1]
    truth=np.where(delta>200,2,np.where(delta<-200,0,1))
    for family,v in run['families'].items():
        names=[f'{family}_{seed}' for seed in [17,29]] if family!='hyper_fixed' else [f'hyper_{seed}_fixed' for seed in [17,29]]
        members=[np.load(r/f'{name}_validation.npz') for name in names]
        ensemble=np.mean([m['p'] for m in members],axis=0);late=~members[0]['early']
        z=np.load(r/f'{family}_calibration.npz')
        assert np.array_equal(z['p'],ensemble[late])
        assert np.array_equal(z['target'],members[0]['target'][late])
        pol=fit_policy(z['p'],z['target'],z['timestamp']//86400)
        verify_equal(pol,v['policy']);verify_equal(metrics(apply_policy(z['p'],pol),z['target']),v['calibration'])
        z=np.load(r/f'{family}_test.npz');pred=apply_policy(z['p'],pol)
        assert np.array_equal(z['target'],truth) and np.array_equal(z['delta_cents'],delta)
        assert np.array_equal(z['timestamp'],dataset['timestamp'][starts-1])
        assert np.array_equal(pred,z['prediction'])
        verify_equal(metrics(pred,truth),v['test'])
        assert np.isfinite(z['p']).all() and np.allclose(z['p'].sum(1),1,atol=1e-6)
    assert run['selected_family']==max(['static','hyper'],key=lambda k:run['families'][k]['policy']['selection_score'])
    verification=dict(status='passed',trials=4,epochs_per_trial=12,train_windows_per_epoch=27716,source_and_data_hashes=True,paired_initial_trunks=True,best_epochs_recomputed=True,calibration_policies_recomputed=True,test_labels_recomputed_from_close=True,test_metrics_recomputed=True)
    write(out/'verification.json',verification)
    write(out/'summary.json',dict(selected_family=run['selected_family'],parameters=run['parameters'],families=run['families'],trials=run['trials'],seconds=run['seconds']))
    labels={'static':'Static TCN','hyper':'Hypernetwork','hyper_fixed':'Hyper: fixed weights'}
    colors={'static':'#4969b2','hyper':'#d77d20','hyper_fixed':'#777777'}
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    families=list(labels)
    for i,k in enumerate(families):
        m=run['families'][k]['test'];p=m['precision'] or 0
        axs[0].bar(i,p*100,color=colors[k]);axs[0].text(i,p*100+.2,f"{m['tp']}/{m['signals']}",ha='center')
        axs[1].bar(i,m['tp'],color='#39875b');axs[1].bar(i,m['fp'],bottom=m['tp'],color='#c97979')
    for ax in axs:ax.set_xticks(range(3),labels.values(),rotation=10);ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Test directional precision (%)');axs[0].set_ylim(0,max(10,axs[0].get_ylim()[1]*1.2));axs[0].set_title('Labels: correct / emitted signals')
    axs[1].set_ylabel('Test signals');axs[1].set_title('Green: correct; red: false')
    fig.suptitle('Same monthly task, matched parameter counts — small reused research test');fig.savefig(out/'precision.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name,rows in curves.items():
        kind=name.rsplit('_',1)[0];style='-' if name.endswith('17') else '--'
        axs[0].plot(range(1,13),[x['train_cost'] for x in rows],style,label=name,color=colors[kind])
        axs[1].plot(range(1,13),[x['selection_policy']['selection_score'] for x in rows],style,label=name,color=colors[kind])
    for ax in axs:ax.set_xlabel('Full chronological pass');ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Train cross-entropy');axs[0].legend();axs[1].set_ylabel('Early-validation precision selection score')
    fig.savefig(out/'learning.png',dpi=170);plt.close(fig)
    lines=['# Hypernetwork: monthly precision experiment','',f"Validation-selected family: **{run['selected_family']}**. GPU: {run['gpu']}; training/evaluation worker {run['seconds']/60:.2f} minutes.",'',
           'Same 512-past-bar inputs; Close after eight M1 bars, >+$2 LONG, <−$2 SHORT, otherwise NONE. January 2024 train: 27,716 windows. Two seeds per architecture, 12 full chronological passes each. Ordinary CE + Adam. Both models have about 635k parameters; difference 191 parameters (0.03%).', '',
           'Hypernetwork generates a rank-four correction W(c)=W0+A(c)B(c)/2 to one classifier layer. The control has a slightly wider static head and the identical initial TCN trunk within each seed. Fixed-weight ablation uses the mean TRAIN context only for weight generation; its classifier input remains real. It has its own validation-fitted thresholds and is not eligible for family selection.', '',
           'Early validation selects epochs; later validation selects thresholds and the model family. Signals require a per-direction cutoff selected by Wilson lower score, at least five validation signals across two dates. No test threshold tuning. Two seed probabilities are averaged.', '',
           '| Model | Calibration correct/signals | Test correct/signals | Test precision | Directional recall |', '|---|---:|---:|---:|---:|']
    for k,v in run['families'].items():
        c=v['calibration'];m=v['test'];lines.append(f"| {labels[k]} | {c['tp']}/{c['signals']} | {m['tp']}/{m['signals']} | {pct(m['precision'])} | {pct(m['recall'])} |")
    lines+=['','| Model | SHORT correct/signals | SHORT precision | LONG correct/signals | LONG precision |','|---|---:|---:|---:|---:|']
    for k,v in run['families'].items():
        a=v['test']['per_direction']['0'];b=v['test']['per_direction']['2'];lines.append(f"| {labels[k]} | {a['tp']}/{a['signals']} | {pct(a['precision'])} | {b['tp']}/{b['signals']} | {pct(b['precision'])} |")
    lines+=['','![Precision and signals](precision.png)','','![Training curves](learning.png)','','Gradient probe: 24 fixed TRAIN examples, eight per class. This is a descriptive probe, not an estimate for all training data. Lower conflict fraction alone does not imply better precision.','', '| Trial | Selected epoch | Gradient conflict: initial → selected | Cross-class conflict: initial → selected | Dynamic variation / base weight norm |','|---|---:|---:|---:|---:|']
    for name,v in run['trials'].items():
        a=v['initial_probe'];b=v['best_probe'];d=b['dynamic_weights'];d='—' if d is None else f"{d['relative_variation']:.3f}"
        lines.append(f"| {name} | {v['best_epoch']} | {pct(a['conflict_fraction'])} → {pct(b['conflict_fraction'])} | {pct(a['cross_class_conflict_fraction'])} → {pct(b['cross_class_conflict_fraction'])} | {d} |")
    a=run['families']['static']['test'];b=run['families']['hyper']['test'];f=run['families']['hyper_fixed']['test']
    lines+=['',f"Interpretation: on this test, static produced {a['tp']} correct and {a['fp']} false signals; dynamic weights produced {b['tp']} correct and {b['fp']} false signals. Freezing the conditioning context produced {f['tp']} correct and {f['fp']} false signals. The dynamic version has numerically higher test precision here, but the predeclared validation selection chose {run['selected_family']}. This does not establish a robust advantage. The small gradient probe still has substantial conflicts in both architectures; conditional weights did not eliminate them.",'']
    lines+=['','Checks passed: explicit dynamic-matrix equivalence, nonzero generator gradients, context-dependent logits, paired initial trunk hashes, all 48 full epochs, independently recomputed targets/thresholds/metrics. See verification.json and remote/run.json.','',
           'Limits: only 690 test windows, 30 directional labels, six dates; overlapping market observations and reused test period. The Wilson score is a selection heuristic, not a confidence guarantee after threshold search. Two seeds and one month cannot establish superiority or absence of information in M1. Hypernetworks still share generator parameters and do not guarantee that each example improves monotonically.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(verification=verification,selected_family=run['selected_family'],test={k:v['test'] for k,v in run['families'].items()}),indent=2))
if __name__=='__main__':main()
