"""Chronological, baseline-controlled spectral research; no images."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np,json,csv,time
from pathlib import Path
from scipy.stats import rankdata
OUT=Path('/workspace/xauusd/reports/spectral_analysis_20261012')
H=[15,60,120]; SPLITS=['discovery','validation','test'];rng=np.random.default_rng(20261012)
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def save(name,obj):(OUT/name).write_text(json.dumps(clean(obj),indent=2,ensure_ascii=False))
def corr(a,b):
 a=np.asarray(a);b=np.asarray(b);a=a-a.mean();b=b-b.mean();return float(a@b/np.sqrt((a@a)*(b@b))) if len(a)>2 else float('nan')
def auc(y,s):
 yes=y>0;n=yes.sum();m=len(y)-n
 return float((rankdata(s)[yes].sum()-n*(n+1)/2)/(n*m)) if n*m else np.nan

def day_ci(vals,days,B=2000,alpha=.05):
 if not len(vals):return [np.nan,np.nan]
 unique,inv=np.unique(days,return_inverse=True);s=np.bincount(inv,weights=vals);n=np.bincount(inv); draws=rng.integers(0,len(s),(B,len(s)));b=s[draws].sum(1)/n[draws].sum(1)
 return np.quantile(b,[alpha/2,1-alpha/2]).tolist()

def trades(signal,mask,h,m):
 inds=np.flatnonzero(mask&(signal!=0)); picked=[];end=-1
 for t in inds:
  if t>end:picked.append(t);end=t+h
 ids=np.array(picked,dtype=int);sgn=np.sign(signal[ids]);p=m['pnl'][ids,H.index(h)]*sgn;days=m['timestamp'][ids]//86400
 counts=np.bincount(m['months'][ids],minlength=13)[1:]
 return ids,p,days,counts

def trade_stats(signal,mask,h,m,ci=True):
 ids,p,days,counts=trades(signal,mask,h,m)
 if len(ids)==0:return {'trades':0}
 dayunique,iv=np.unique(days,return_inverse=True);dailyp=np.bincount(iv,weights=p);cum=np.cumsum(p)
 return {'trades':len(p),'active_days':len(dayunique),'mean_gross_usd_per_oz':p.mean(),'median_gross':np.median(p),'win_rate_gross':(p>0).mean(),'gross_total':p.sum(),'profit_factor_gross':p[p>0].sum()/max(1e-12,-p[p<0].sum()),'mean_gross_day_cluster_ci95':day_ci(p,days) if ci else None,'mean_gross_day_cluster_ci99':day_ci(p,days,alpha=.01) if ci else None,'cost_scenarios':{str(c):{'mean_net':p.mean()-c,'total_net':(p-c).sum(),'win_rate':(p>c).mean()} for c in [0,.1,.25,.5,1]},'monthly_counts':counts,'monthly_mean_gross':{str(mo):p[m['months'][ids]==mo].mean() for mo in np.unique(m['months'][ids])},'max_drawdown_gross_one_oz':float(np.max(np.maximum.accumulate(np.r_[0,cum])[1:]-cum))}

def analyze_events(m,allx,names,masks):
 n=len(m['close']);split=m['split'];months=m['months'];days=m['timestamp']//86400
 # Shock/reaction study. Use pre-event volatility, no future in detection.
 c=m['close'];vol=m['vol'];N=len(c);shock=np.zeros(N);shock[5:]=1e4*np.log(c[5:]/c[:-5])/(vol[:-5]*np.sqrt(5)); events=[];end=-1
 energy=(allx[:,names.index('wavelet_returns.energy.band4_8')]+allx[:,names.index('wavelet_returns.energy.band8_16')])/2
 for t in np.flatnonzero(abs(shock)>=2.5):
  if t<5 or t<=end or t+65>=N:continue
  end=t+60  # detector cooldown depends only on the shock, never on future label validity
  q=t+5
  if not masks[split[q],q,1] or m['timestamp'][q]-m['timestamp'][t-5]!=600:continue
  direction=int(np.sign(shock[t]));anchor=c[t-5];mag=abs(c[t]-anchor); retr=direction*(c[t]-c[q])/max(mag,.01)
  already=(np.min(m['low'][t+1:q+1])<=anchor) if direction>0 else (np.max(m['high'][t+1:q+1])>=anchor)
  future=(np.min(m['low'][q+1:q+61])<=anchor) if direction>0 else (np.max(m['high'][q+1:q+61])>=anchor)
  events.append({'t':int(t),'decision':int(q),'split':int(split[q]),'month':int(months[q]),'day':int(days[q]),'shock_z':abs(shock[t]),'shock_usd':mag,'side':direction,'vol':vol[t-5],'retracement':retr,'energy_change':energy[q]-energy[t-5],'reaction_energy':energy[q],'already_returned':int(already),'future_return':int(future),'continuation_pnl':direction*m['pnl'][q,1]})
 ev={k:np.array([r[k] for r in events]) for k in events[0]};etr=(ev['split']==0)&(ev['already_returned']==0);edges=np.quantile(ev['energy_change'][etr],[1/3,2/3]);group=np.searchsorted(edges,ev['energy_change']);table=[]
 for s in range(3):
  for g in range(3):
   ok=(ev['split']==s)&(group==g)&(ev['already_returned']==0);f=ev['future_return'][ok]
   table.append({'split':SPLITS[s],'energy_group':['low','middle','high'][g],'n':ok.sum(),'return_to_anchor_rate':f.mean() if ok.any() else np.nan,'rate_ci95':day_ci(f,ev['day'][ok]),'mean_continuation_usd':ev['continuation_pnl'][ok].mean() if ok.any() else np.nan,'mean_shock_usd':ev['shock_usd'][ok].mean() if ok.any() else np.nan,'mean_observed_retracement':ev['retracement'][ok].mean() if ok.any() else np.nan})
 # Conditional linear-probability models; coefficients fitted discovery, assessed later.
 ebase=np.column_stack([np.log(ev['shock_z']),np.log(ev['vol']),ev['retracement'],ev['retracement']**2,ev['side'],np.sin((m['timestamp'][ev['decision']]%86400)*2*np.pi/86400),np.cos((m['timestamp'][ev['decision']]%86400)*2*np.pi/86400)])
 ereports={}
 for label,ex in [('geometry_only',ebase),('geometry_plus_energy',np.column_stack([ebase,ev['energy_change'],ev['reaction_energy']]))]:
  mu=ex[etr].mean(0);sd=np.maximum(ex[etr].std(0),1e-6);ex=np.column_stack([np.ones(len(ex)),np.clip((ex-mu)/sd,-8,8)]);coef=np.linalg.solve(ex[etr].T@ex[etr]+np.eye(ex.shape[1]),ex[etr].T@ev['future_return'][etr]);p=np.clip(ex@coef,.001,.999);ereports[label]={'coefficients':coef,'splits':{}}
  for s in range(3):
   ok=(ev['split']==s)&(ev['already_returned']==0);yy=ev['future_return'][ok];ereports[label]['splits'][SPLITS[s]]={'n':ok.sum(),'brier':np.mean((p[ok]-yy)**2),'auc':auc(yy-.5,p[ok])}
 save('shock_reactions.json',{'detected_events':len(events),'already_returned_excluded':int(ev['already_returned'].sum()),'energy_edges_train':edges,'table':table,'conditional_models':ereports});np.savez_compressed(OUT/'shock_events.npz',**ev)
 with (OUT/'shock_events.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=events[0].keys());w.writeheader();w.writerows(events)

def main():
 started=time.time();meta=json.loads((OUT/'features.json').read_text());m=dict(np.load(OUT/'market.npz'));base=np.load(OUT/'base.npy');spec=np.load(OUT/'spectral.npy',mmap_mode='r');names=meta['base']+meta['spectral'];n=len(base);split=m['split'];months=m['months'];days=m['timestamp']//86400
 masks=np.zeros((3,n,3),bool)
 for s in range(3):
  for j,h in enumerate(H):
   same=np.zeros(n,bool);same[:-h]=split[:-h]==split[h:];masks[s,:,j]=(split==s)&m['valid'][:,j]&same
 save('counts.json',{'total_rows':n,'features_base':len(meta['base']),'features_spectral':len(meta['spectral']),'valid_rows':{SPLITS[s]:{str(h):int(masks[s,:,j].sum()) for j,h in enumerate(H)} for s in range(3)}})
 # Scan correlations for every stored spectral component, ranking on discovery only.
 mm=np.load(OUT/'full_component_moments.npz');screen=[]
 for method in mm.files:
  v=mm[method];nn=v[:,:,:,0];sx=v[:,:,:,1];xx=np.maximum(v[:,:,:,2]-sx*sx/np.maximum(nn,1),1e-20)
  for target,syi,sxyi in [('direction',3,5),('absolute_move',6,7)]:
   sy=v[:,:,:,syi];yy=np.maximum(v[:,:,:,4]-sy*sy/np.maximum(nn,1),1e-20);rr=(v[:,:,:,sxyi]-sx*sy/np.maximum(nn,1))/np.sqrt(xx*yy)
   for j,h in enumerate(H):
    for typ,typname in enumerate(['energy','cos','sin']):
     for k,p in enumerate(meta['axes'][method]):screen.append({'method':method,'period':p,'feature':typname,'horizon':h,'target':target,**{SPLITS[s]+'_corr':rr[s,j,typ,k] for s in range(3)}})
 with (OUT/'all_component_correlations.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(screen[0]));w.writeheader();w.writerows(screen)
 # Only training ranks determine this concise list, not test peaks.
 top={}
 for target in ['direction','absolute_move']:
  for h in H:
   rows=[r for r in screen if r['target']==target and r['horizon']==h];top[f'{target}_{h}']=sorted(rows,key=lambda r:abs(r['discovery_corr']),reverse=True)[:12]
 save('component_discovery_leaders.json',top)
 print('component screen complete',flush=True)
 # Model targets: normalized signed movement, sign, log future realized volatility.
 target=np.column_stack([m['y'],np.sign(m['y']),np.log(np.maximum(m['rv'],1e-10))]);predictions={};modelreport={}
 collections={'price_only':np.arange(len(base[0])),'price_plus_stft':np.r_[np.arange(len(base[0])),len(base[0])+np.array([i for i,x in enumerate(meta['spectral']) if x.startswith('stft')])],'price_plus_wavelets':np.r_[np.arange(len(base[0])),len(base[0])+np.array([i for i,x in enumerate(meta['spectral']) if x.startswith('wavelet')])],'price_plus_all':np.arange(len(names))}
 allx=np.column_stack([base,spec]).astype(np.float32);del spec
 # Common horizon120 training mask prevents any split-crossing label entering fit.
 train=masks[0,:,2];val=masks[1,:,2];test=masks[2,:,2]
 for model,idx in collections.items():
  x=allx[:,idx].copy();mu=x[train].mean(0,dtype=np.float64);sd=x[train].std(0,dtype=np.float64);sd=np.maximum(sd,1e-6);x-=mu;x/=sd;np.clip(x,-8,8,out=x);x=np.column_stack([np.ones(n,dtype=np.float32),x]);xt=x[train];yt=target[train].copy();yt[:,:3]=np.clip(yt[:,:3],-10,10)
  gram=(xt.T@xt/len(xt)).astype(np.float64);cross=xt.T@yt/len(xt);del xt; best=np.full(9,np.inf);bestp=np.empty((n,9));bestalpha=np.zeros(9);params=[]
  for alpha in [.001,.01,.1,1,10]:
   reg=np.eye(len(gram))*alpha;reg[0,0]=0;coef=np.linalg.solve(gram+reg,cross);pv=x[val]@coef;loss=np.mean((pv-target[val])**2,axis=0);improve=loss<best
   if improve.any():best[improve]=loss[improve];bestp[:,improve]=x@coef[:,improve];bestalpha[improve]=alpha
  del x;predictions[model]=bestp.astype(np.float32);report={'selected_alpha':bestalpha,'metrics':{}}
  for s in range(3):
   stats={}
   for j,h in enumerate(H):
    ok=masks[s,:,j];yy=m['y'][ok,j];p=bestp[ok,j];sp=bestp[ok,3+j];lr=target[ok,6+j];vp=bestp[ok,6+j]
    stats[str(h)]={'n':ok.sum(),'return_corr':corr(yy,p),'return_mse':np.mean((yy-p)**2),'zero_return_mse':np.mean(yy**2),'direction_accuracy':np.mean((sp>0)==(yy>0)),'direction_auc':auc(yy,sp),'long_fraction_actual':np.mean(yy>0),'sign_mse':np.mean((np.sign(yy)-sp)**2),'log_rv_mse':np.mean((lr-vp)**2),'log_rv_corr':corr(lr,vp)}
   report['metrics'][SPLITS[s]]=stats
  modelreport[model]=report;print(json.dumps({'model':model,'seconds':round(time.time()-started,1)}),flush=True)
 save('models.json',modelreport);np.savez_compressed(OUT/'predictions.npz',**predictions)
 # Daily paired heldout error differences against price-only; units return/sign/log-RV MSE.
 comparisons={}
 for model,p in predictions.items():
  if model=='price_only':continue
  stats={}
  for j,h in enumerate(H):
   ok=masks[2,:,j];stat={}
   for kind,col in [('return',j),('sign',j+3),('log_rv',j+6)]:
    delta=(predictions['price_only'][ok,col]-target[ok,col])**2-(p[ok,col]-target[ok,col])**2
    stat[kind]={'mse_improvement':delta.mean(),'day_cluster_ci95':day_ci(delta,days[ok]),'positive_months':sum(np.mean(delta[months[ok]==mo])>0 for mo in [10,11,12])}
   stats[str(h)]=stat
  comparisons[model]=stats
 save('paired_model_comparisons.json',comparisons)
 # Quantile descriptive energy/activity relationships, thresholds discovery-only.
 energy_names=['base.logvol_60','wavelet_returns.energy.band4_8','wavelet_returns.energy.band16_32','wavelet_returns.energy.band64_128','wavelet_returns.energy.band256_512','wavelet_returns.energy.band2048_4096','stft_8192.energy.band4096_8192']
 activity={}
 for name in energy_names:
  if name not in names:continue
  v=allx[:,names.index(name)];edges=np.quantile(v[train],[.2,.4,.6,.8]);groups=np.searchsorted(edges,v);rows=[]
  for s in range(3):
   for q in range(5):
    ok=masks[s,:,1]&(groups==q);rows.append({'split':SPLITS[s],'quintile':q+1,'n':ok.sum(),'mean_future_rv_bps':m['rv'][ok,1].mean(),'median_future_abs_move_usd':np.median(abs(m['pnl'][ok,1])),'up_fraction':np.mean(m['pnl'][ok,1]>0),'mean_signed_move_usd':m['pnl'][ok,1].mean()})
  activity[name]={'edges':edges,'rows':rows}
 save('energy_activity.json',activity)
 # Trade candidates: validation selects one configuration for each model; final test untouched.
 strategies={};candidate_rows=[]
 for model,p in predictions.items():
  candidates=[]
  for j,h in enumerate(H):
   for kind,col in [('return',j),('sign',j+3)]:
    for quantile in [.5,.8,.9,.95]:
     threshold=np.quantile(abs(p[masks[1,:,j],col]),quantile);sig=np.where(abs(p[:,col])>=threshold,np.sign(p[:,col]),0);ids,profits,dd,_=trades(sig,masks[1,:,j],h,m)
     if len(ids)<80:continue
     _,iv=np.unique(dd,return_inverse=True);daily=np.bincount(iv,weights=profits-.25);score=daily.mean()/max(daily.std(),1e-8)*np.sqrt(252)
     row={'model':model,'horizon':h,'kind':kind,'quantile':quantile,'threshold':threshold,'val_trades':len(ids),'val_mean_gross':profits.mean(),'validation_daily_score_cost025':score};candidates.append((score,row,sig,j));candidate_rows.append(row)
  score,row,sig,j=max(candidates,key=lambda z:z[0]);strategies[model]={'selection':row,'validation':trade_stats(sig,masks[1,:,j],H[j],m),'test':trade_stats(sig,masks[2,:,j],H[j],m)}
  ids,pp,dd,_=trades(sig,masks[2,:,j],H[j],m);np.savez_compressed(OUT/f'trades_{model}.npz',row=ids,timestamp=m['timestamp'][ids],side=sig[ids],gross=pp)
 save('trading_candidates_validation.json',candidate_rows);save('trading.json',strategies)
 # Simple one-feature rules: top/bottom training quintiles, side from discovery mean.
 rules=[]
 for j,h in enumerate(H):
  tr=masks[0,:,j];lo,hi=np.quantile(allx[tr],[.2,.8],axis=0); yy=m['y'][:,j]
  # All features screened, but only top20 train candidates per horizon reach validation.
  chunk=[]
  for k,name in enumerate(names):
   a=tr&(allx[:,k]<=lo[k]);b=tr&(allx[:,k]>=hi[k]);diff=yy[b].mean()-yy[a].mean();score=abs(diff)
   chunk.append((score,k,float(lo[k]),float(hi[k]),int(np.sign(diff) or 1)))
  for score,k,low,high,side in sorted(chunk,reverse=True)[:20]:
   sig=np.where(allx[:,k]>=high,side,np.where(allx[:,k]<=low,-side,0));v=trade_stats(sig,masks[1,:,j],h,m,False)
   rules.append({'feature':names[k],'index':k,'horizon':h,'lo':low,'hi':high,'side_high':side,'discovery_normalized_spread':score,'validation':v})
 # Lock FIVE best validation gross mean/sqrt variability surrogate uses mean, min80 trades.
 rules.sort(key=lambda r:r['validation'].get('mean_gross_usd_per_oz',-1e9),reverse=True)
 save('rule_candidates_validation.json',rules)
 selected=[]
 for r in rules[:5]:
  k=r['index'];j=H.index(r['horizon']);sig=np.where(allx[:,k]>=r['hi'],r['side_high'],np.where(allx[:,k]<=r['lo'],-r['side_high'],0));r['test']=trade_stats(sig,masks[2,:,j],r['horizon'],m);selected.append(r)
 save('selected_rules.json',selected)
 analyze_events(m,allx,names,masks)
 save('analysis_complete.json',{'complete':True,'seconds':time.time()-started,'all_rows':n,'images_created':False,'gpu_used':False})
 print('COMPLETE',time.time()-started,flush=True)
if __name__=='__main__':main()
