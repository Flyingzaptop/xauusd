import numpy as np,json
from pathlib import Path
from sklearn.metrics import roc_auc_score
P=Path('/workspace/xauusd/reports/spectral_transformer_20261013');R=P/'remote';d=np.load(R/'evaluation_data.npz');ids=d['test'];keep=np.arange(0,len(ids),121);rows=ids[keep];days=d['timestamp'][rows]//86400;dayidx=days-days.min();L=int(dayidx.max()+1);rng=np.random.default_rng(872)
weights=[]
for _ in range(1000):
 starts=rng.integers(0,L,(L+6)//7);idx=((starts[:,None]+np.arange(7))%L).flatten()[:L];weights.append(np.bincount(idx,minlength=L)[dayidx])
out={}
for kind in ['price','raw','engineered']:
 p=np.load(R/f'{kind}_test.npz');prob=p['event_prob'][keep];event=d['event'][rows]>0;side=d['event'][rows]%2==1;pe=1-prob[:,0];up=prob[:,1::2].sum(1);down=prob[:,2::2].sum(1);ps=up/np.maximum(up+down,1e-8);ea=[];da=[]
 for w in weights:
  ea.append(roc_auc_score(event,pe,sample_weight=w));da.append(roc_auc_score(side[event],ps[event],sample_weight=w[event]))
 out[kind]={'nonoverlap_event_auc':roc_auc_score(event,pe),'event_auc_ci95_week':np.quantile(ea,[.025,.975]).tolist(),'nonoverlap_direction_auc_given_event':roc_auc_score(side[event],ps[event]),'direction_auc_ci95_week':np.quantile(da,[.025,.975]).tolist(),'predicted_mean_std_normalized_by_horizon':p['mean'].std(0).tolist(),'actual_movement_std_normalized_by_horizon':d['y'][ids].std(0).tolist(),'event_probability_range':[float((1-p['event_prob'][:,0]).min()),float((1-p['event_prob'][:,0]).max())]}
(P/'diagnostics.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
