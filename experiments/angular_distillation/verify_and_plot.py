"""Independent artifact verification and descriptive plots; never retrains/selects models."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/angular_distillation_20261008'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='complete'
    assert sha(Path(__file__).with_name('run_experiment.py'))==manifest['script_sha256']
    assert sha(Path(__file__).with_name('protocol.md'))==manifest['protocol_sha256']
    assert sha(OUT/'metrics.csv')==manifest['metrics_sha256']
    audit=json.loads((OUT/'data_audit.json').read_text(encoding='utf-8'))
    assert sha(ROOT/'data/M1/xauusd_m1.parquet')==audit['source_sha256']
    m=pd.read_csv(OUT/'metrics.csv');c=pd.read_csv(OUT/'paired_comparisons.csv')
    assert len(m)==manifest['metrics_rows']
    assert len(c)==manifest['comparison_rows']
    assert np.isfinite(m[['mse_bps2','mae_bps','skill_vs_zero','direction_accuracy','correlation']]).all().all()
    ts=pd.read_parquet(ROOT/'data/M1/xauusd_m1.parquet',columns=['timestamp']).timestamp.to_numpy(dtype='datetime64[s]').astype(np.int64)
    splits=np.load(OUT/'market_split_indices.npz')
    bounds=[int(pd.Timestamp(v,tz='UTC').timestamp()) for v in ['2025-01-01','2026-01-01','2026-09-01']]
    for name in splits.files:
        ix=splits[name]
        assert len(ix)==audit['sample_counts'][name]
        for rel in range(-73,41):assert np.all(ts[ix+rel]-ts[ix+rel-1]==60)
        if name=='train':assert np.all(ts[ix+40]<bounds[0])
        if name=='val':assert np.all(ts[ix-74]>=bounds[0]) and np.all(ts[ix+40]<bounds[1])
        if name=='test':assert np.all(ts[ix-74]>=bounds[1]) and np.all(ts[ix+40]<bounds[2])
        if name=='tail':assert np.all(ts[ix-74]>=bounds[2])
    max_relative=0.;checked=0;primary=[];zero_comparisons=[]
    for kind,seeds in [('market',[17,29,43]),('random_walk',[17]),('predictable_ar',[17])]:
        selection=json.loads((OUT/f'{kind}_frozen_lambda.json').read_text(encoding='utf-8'))
        training=json.loads((OUT/f'{kind}_training_selection.json').read_text(encoding='utf-8'))
        for rep,choice in selection.items():
            vs={name:np.mean([training[f'{rep}/{s}'][name]['val_score'] for s in seeds]) for name in ['C25','C75']}
            assert choice['selected']==min(vs,key=vs.get)
            for split in ['test','tail']:
                folder=OUT/'predictions'/kind/rep/split
                if not folder.exists():continue
                truth=np.load(folder/'truth.npz');y=truth['y_bps'].astype(np.float64)
                for seed in seeds:
                    for method in ['A','A_long','B','T','C25','C75']:
                        p=np.load(folder/f'{method}_{seed}.npz')['p_bps'].astype(np.float64)
                        assert p.shape==y.shape and np.isfinite(p).all()
                        for h in [1,5,10,20]:
                            computed=np.mean((p[:,h-1,3]-y[:,h-1,3])**2)
                            reported=m[(m.kind==kind)&(m.representation==rep)&(m.split==split)&(m.seed==seed)&(m.method==method)&(m.period=='ALL')&(m.horizon==h)].mse_bps2.item()
                            discrepancy=abs(computed-reported)/max(abs(reported),1e-10)
                            max_relative=max(max_relative,discrepancy);assert discrepancy<2e-6;checked+=1
                        jj=np.array([0,4,9,19]);weights=np.array([1,5,10,20])
                        loss=np.mean((p[:,jj,3]-y[:,jj,3])**2/weights)
                        zero=np.mean(y[:,jj,3]**2/weights)
                        primary.append(dict(kind=kind,representation=rep,split=split,seed=seed,method=method,
                            primary_mse_bps2_per_minute=loss,skill_vs_zero=1-loss/zero,
                            selected=method==choice['selected']))
                # Post-fit diagnostics only: C vs zero with day-block intervals.
                jj=np.array([0,4,9,19]);weights=np.array([1,5,10,20])
                ce=np.mean([np.mean((np.load(folder/f'{choice["selected"]}_{s}.npz')['p_bps'][:,jj,3]-y[:,jj,3])**2/weights,axis=1) for s in seeds],axis=0)
                ze=np.mean(y[:,jj,3]**2/weights,axis=1)
                _,inv=np.unique(truth['timestamp']//86400,return_inverse=True)
                cnt=np.bincount(inv);summ=np.bincount(inv,weights=ce-ze)
                rng=np.random.default_rng(2718);idx=rng.integers(0,len(cnt),(2000,len(cnt)))
                boot=summ[idx].sum(1)/cnt[idx].sum(1)
                ci=np.quantile(boot,[.025,.975])
                zero_comparisons.append(dict(kind=kind,representation=rep,split=split,
                    delta_vs_zero=float(np.mean(ce-ze)),ci_low=ci[0],ci_high=ci[1],
                    relative_improvement_pct=float((1-ce.mean()/ze.mean())*100)))
    pd.DataFrame(primary).to_csv(OUT/'primary_scores_by_seed.csv',index=False)
    pd.DataFrame(zero_comparisons).to_csv(OUT/'selected_vs_zero.csv',index=False)
    selection=json.loads((OUT/'market_frozen_lambda.json').read_text(encoding='utf-8'))
    convergence=[]
    for rep in ['log','angular']:
        for seed in [17,29,43]:
            meta=json.loads((OUT/'market_training_selection.json').read_text(encoding='utf-8'))[f'{rep}/{seed}']
            for method in ['A','A_long','T','B','C25','C75']:
                curve=pd.read_csv(OUT/'models/market'/rep/str(seed)/f'{"A_long" if method=="A" else method}_curve.csv')
                if method=='A':curve=curve.iloc[:12]
                assert abs(curve.val_score.min()-meta[method]['val_score'])<1e-8
                convergence.append(dict(representation=rep,seed=seed,method=method,best_epoch=meta[method]['best_epoch'],
                    budget_epochs=len(curve),best_at_last_epoch=meta[method]['best_epoch']==len(curve),
                    first_val=curve.val_score.iloc[0],best_val=curve.val_score.min(),last_val=curve.val_score.iloc[-1]))
    pd.DataFrame(convergence).to_csv(OUT/'convergence_diagnostics.csv',index=False)
    scale=json.loads((OUT/'scale.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    plots=OUT/'plots';plots.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    colors={'A':'#456990','A_long':'#7599aa','B':'#e9a23b','C_selected':'#c84b31','T':'#229b77','ridge':'#7d659c'}
    summary=pd.read_csv(OUT/'summary.csv')
    for ax,rep in zip(axes,['log','angular']):
        for method in ['A','A_long','B','C_selected','ridge']:
            g=summary[(summary.split=='test')&(summary.representation==rep)&(summary.method==method)].sort_values('horizon')
            ax.plot(g.horizon,g.skill_vs_zero*100,'o-',label=method,color=colors[method])
        ax.axhline(0,color='black',linewidth=.8);ax.set(title=f'{rep}: causal forecasts',xlabel='Horizon (minutes)',ylabel='MSE reduction vs zero (%)');ax.set_xticks([1,5,10,20]);ax.legend(fontsize=9)
    fig.suptitle('XAUUSD Jan-Aug 2026 | mean of 3 seeds | exposed research evaluation')
    fig.tight_layout();fig.savefig(plots/'forecast_skill.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,rep in zip(axes,['log','angular']):
        for method in ['A','B','C_selected']:
            z=m[(m.kind=='market')&(m.split=='test')&(m.representation==rep)&(m.method==method)&(m.period!='ALL')&(m.horizon==20)]
            g=z.groupby('period').skill_vs_zero.mean()*100
            ax.plot(g.index,g.values,'o-',label=method,color=colors[method])
        ax.axhline(0,color='black',linewidth=.8);ax.tick_params(axis='x',rotation=35);ax.set(title=f'{rep}: 20-minute horizon',ylabel='MSE reduction vs zero (%)');ax.legend()
    fig.tight_layout();fig.savefig(plots/'monthly_stability.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5));ax.loglog(scale['lags'],scale['MAD'],'o',label='Train MAD');ax.loglog(scale['lags'],scale['fitted'],'-',label=f'Fit H={scale["H"]:.4f}')
    ax.set(xlabel='Lag (minutes)',ylabel='MAD of log-return',title='Train-only scale law; fit does not imply predictability');ax.legend();fig.tight_layout();fig.savefig(plots/'scale_law.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True)
    for row,rep in enumerate(['log','angular']):
        for col,seed in enumerate([17,29,43]):
            ax=axes[row,col]
            for method in ['A_long','B','C25','C75']:
                curve=pd.read_csv(OUT/'models/market'/rep/str(seed)/f'{method}_curve.csv')
                ax.plot(curve.epoch,curve.val_score,label=method)
            ax.axvline(12,color='gray',ls=':',lw=.8);ax.set(title=f'{rep} seed={seed}',xlabel='Epoch',ylabel='Validation score');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(plots/'validation_curves.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    cp=c[(c.kind=='market')&(c.split=='test')&(c.horizon==0)].reset_index(drop=True)
    for i,row in cp.iterrows():
        ax.errorbar(row.delta_mse_bps2,i,xerr=[[row.delta_mse_bps2-row.ci_low],[row.ci_high-row.delta_mse_bps2]],
            fmt='o',color='#c84b31' if row.candidate=='C_selected' else '#e9a23b',capsize=4)
    ax.axvline(0,color='black',lw=1)
    ax.set_yticks(range(len(cp)),[f'{r.representation}: {r.candidate} - {r.baseline}' for r in cp.itertuples()])
    ax.set(xlabel='Primary loss difference (bps squared / minute); negative is better',
        title='Paired day-block bootstrap: 95% diagnostic intervals')
    fig.tight_layout();fig.savefig(plots/'primary_comparisons.png',dpi=170);plt.close(fig)
    verification=dict(status='passed',prediction_metric_checks=checked,max_relative_metric_discrepancy=max_relative,
        source_code_protocol_hashes_match=True,all_split_supports_and_minute_continuity_checked=True,
        lambda_selection_matches_validation=True,best_checkpoints_match_training_curves=True,
        all_predictions_finite=True,verifier_sha256=sha(Path(__file__)))
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps(verification,indent=2));print(pd.DataFrame(zero_comparisons).to_string(index=False))
    print(c[(c.kind=='market')&(c.split=='test')&(c.horizon==0)].to_string(index=False))

if __name__=='__main__':main()
