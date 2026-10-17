"""Interpret trained trees. Rule selection uses model structure/counts, never test labels."""
from pathlib import Path
import numpy as np,lightgbm as lgb,json
R=Path('/content/intermarket');O=R/'results';meta=json.loads((R/'prepared.json').read_text());names=meta['feature_names'];mm=np.memmap(R/'features.f32',dtype='float32',mode='r',shape=tuple(meta['shape']));d=np.load(O/'evaluation_data.npz');model=lgb.Booster(model_file=str(O/'all_60_final.txt'));dump=model.dump_model();candidates=[]
def walk(node,path,tree):
 if 'leaf_index' in node:
  symbols={names[p['feature']].split('__')[0] for p in path if '__' in names[p['feature']]}
  if len(symbols)>=2 and node['leaf_count']>=2000:
   candidates.append({'tree':tree,'leaf':node['leaf_index'],'leaf_value':node['leaf_value'],'train_count':node['leaf_count'],'symbols':sorted(symbols),'path':path,'selection_score':abs(node['leaf_value'])*np.sqrt(node['leaf_count'])})
  return
 cond={'feature':node['split_feature'],'name':names[node['split_feature']],'threshold':float(node['threshold']),'default_left':node['default_left'],'decision_type':node['decision_type']}
 walk(node['left_child'],path+[{**cond,'left':True}],tree);walk(node['right_child'],path+[{**cond,'left':False}],tree)
for i,t in enumerate(dump['tree_info']):walk(t['tree_structure'],[],i)
chosen=[];seen=set()
for q in sorted(candidates,key=lambda q:q['selection_score'],reverse=True):
 sig=tuple(q['symbols'])
 if sig in seen:continue
 seen.add(sig);chosen.append(q)
 if len(chosen)==5:break
for q in chosen:
 for part,idx in [('validation',d['val_idx']),('test',d['test_idx'])]:
  mask=np.ones(len(idx),bool)
  for cond in q['path']:
   assert cond['decision_type']=='<='
   v=mm[cond['feature'],idx];left=(v<=cond['threshold'])|(np.isnan(v)&cond['default_left']);mask&=left if cond['left'] else ~left
  yy=d['targets'][idx,2];mask&=np.isfinite(yy);v=yy[mask];gold=np.load(O/('gold_validation.npz' if part=='validation' else 'gold_test.npz'))['prediction'][:,2];q[part]={'rows':int(mask.sum()),'mean_future_normalized_change':float(v.mean()) if len(v) else None,'up_rate':float((v>0).mean()) if len(v) else None,'mean_residual_vs_gold_only':float((v-gold[mask]).mean()) if len(v) else None}
importance=[]
for h in [15,30,60,120]:
 b=lgb.Booster(model_file=str(O/f'all_{h}_final.txt'));g=b.feature_importance('gain');groups={}
 for name,value in zip(names,g):
  group=name.split('__')[0] if '__' in name else 'clock';groups[group]=groups.get(group,0)+float(value)
 total=sum(groups.values());importance.append({'horizon':h,'top_instruments':[{'symbol':k,'gain_share':v/total if total else 0} for k,v in sorted(groups.items(),key=lambda z:z[1],reverse=True)[:20]]})
(O/'patterns.json').write_text(json.dumps({'selection':'top5different symbol combinations by abs leaf contribution*sqrt training count, >=2000 training rows, >=2symbols, 60min final model; no test-label selection','rules_are_not_standalone_trading_policies':True,'rules':chosen,'group_gain_importance':importance},indent=2));print('Patterns and grouped gain saved',len(chosen))
