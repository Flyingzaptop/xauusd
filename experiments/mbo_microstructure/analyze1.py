"""Stage-1: does the 100 ms book/flow state predict future mid-price moves, in ticks, and does it beat the spread?
Train on the first 60% of trading days, test on the last 40% (purged). Ridge + LightGBM."""
import sys,glob,json,time,numpy as np
import lightgbm as lgb
D=sys.argv[1];OUT=sys.argv[2];TICK=0.1
t0=time.time()
Z=[np.load(f) for f in sorted(glob.glob(D+'/chunk_*.npz'))]
ts=np.concatenate([z['ts'] for z in Z]);px=np.concatenate([z['px'] for z in Z]);F=np.concatenate([z['f'] for z in Z]);del Z
N=len(ts);bid=px[:,0];ask=px[:,1];mid=(bid+ask)/2;sp=(ask-bid)/TICK
sec=(ts//10**9)%86400;hour=sec/3600.0
sess=~((hour>=22)&(hour<23))&np.isfinite(mid)&(sp>0)&(sp<=100)
day=ts//(86400*10**9)
print('rows',N,'valid session rows',sess.sum(),'%.0fs'%(time.time()-t0),flush=True)
cs=lambda x:np.concatenate([[0.],np.cumsum(x.astype(np.float64))])
def win(c,w):
    out=np.zeros(N);out[w:]=c[w+1:N+1]-c[1:N+1-w][:N-w] if False else c[w:N]-c[:N-w];return out
bsz1,asz1=F[:,0].astype(np.float64),F[:,5].astype(np.float64)
b5=F[:,:5].sum(1).astype(np.float64);a5=F[:,5:10].sum(1).astype(np.float64)
imb1=(bsz1-asz1)/(bsz1+asz1+1e-9);imb5=(b5-a5)/(b5+a5+1e-9)
micro=(bid*asz1+ask*bsz1)/(bsz1+asz1+1e-9)
feat={}
def add(k,v):feat[k]=v.astype(np.float32)
for k_,v_ in {'spread':sp,'imb1':imb1,'imb5':imb5,'micro_mid':(micro-mid)/TICK,'ldepth1':np.log1p(bsz1+asz1),'ldepth5':np.log1p(b5+a5)}.items():add(k_,v_)
ofi=cs(F[:,10]);bv=cs(F[:,11]);sv=cs(F[:,12]);nt=cs(F[:,13]);adb=cs(F[:,14]);ada=cs(F[:,15]);cxb=cs(F[:,16]);cxa=cs(F[:,17])
midf=np.where(np.isfinite(mid),mid,np.nan)
for w in [1,10,50,300]:
    add(f'ofi{w}',win(ofi,w));add(f'tv{w}',win(bv,w)-win(sv,w));add(f'nt{w}',win(nt,w))
    add(f'addimb{w}',win(adb,w)-win(ada,w));add(f'cxlimb{w}',win(cxb,w)-win(cxa,w))
    m=np.full(N,np.nan);m[w:]=(midf[w:]-midf[:-w])/TICK;add(f'mom{w}',m)
names=list(feat);X=np.empty((N,len(names)),np.float32)
for j,k in enumerate(names):X[:,j]=feat[k]
del feat,F,ofi,bv,sv,nt,adb,ada,cxb,cxa,bsz1,asz1,b5,a5,imb1,imb5,micro,midf
print('features',len(names),'%.0fs'%(time.time()-t0),flush=True)
# train/test split by trading day
udays=np.unique(day[sess]);cut=udays[int(len(udays)*0.6)]
print('days',len(udays),'cut',cut)
rows=np.arange(0,N,10)                       # 1 s sampling
import os
MAXSP=float(os.environ.get('MAXSP','100'))
rows=rows[sess[rows]&(sp[rows]<=MAXSP)&np.isfinite(X[rows]).all(1)]
tr=rows[day[rows]<cut-0];te=rows[day[rows]>cut]
# purge: drop train rows within 120 s of cut handled by excluding day==cut
mu=X[tr].mean(0);sd=X[tr].std(0)+1e-9
def targ(r,L,h):
    e=r+L;x=e+h;ok=(x<N)
    ok&=(ts[np.minimum(x,N-1)]-ts[r])==(L+h)*100_000_000
    ok&=sess[np.minimum(x,N-1)]&sess[np.minimum(e,N-1)]
    xi=np.minimum(x,N-1);ei=np.minimum(e,N-1)
    dm=(mid[xi]-mid[ei])/TICK;pl=(bid[xi]-ask[ei])/TICK;ps=(bid[ei]-ask[xi])/TICK    # taker pnl in ticks (long, short), no fees
    return ok&np.isfinite(dm),dm,pl,ps
res={}
def evaluate(name,pred_tr,pred_te,L,h):
    okt,dmt,plt,pst=targ(te,L,h);p=pred_te[okt];y=dmt[okt]
    out={'n_test':int(okt.sum()),'ic':float(np.corrcoef(p,y)[0,1]),'spread_med_ticks':float(np.median(sp[te][okt]))}
    okr,_,_,_=targ(tr,L,h);thr_src=np.abs(pred_tr[okr])
    for q in [0.9,0.99]:
        thr=np.quantile(thr_src,q);sel=np.abs(p)>thr
        sgn=np.sign(p[sel]);pnl=np.where(sgn>0,plt[okt][sel],pst[okt][sel])
        out[f'top{int((1-q)*100)}pct']={'n':int(sel.sum()),'mean_signed_mid_move_ticks':float(np.mean(sgn*y[sel])) if sel.any() else None,
                                       'taker_net_ticks_per_trade':float(pnl.mean()) if sel.any() else None,'mean_spread_ticks':float(sp[te][okt][sel].mean()) if sel.any() else None}
    res[f'{name}_L{L}_h{h}']=out;return out
Xtr=(X[tr]-mu)/sd;Xte=(X[te]-mu)/sd
for L in [0,1,5,10]:
    for h in [10,50,300,600]:
        okr,dmr,_,_=targ(tr,L,h);okt,dmt,_,_=targ(te,L,h)
        A=Xtr[okr];A=np.c_[A,np.ones(len(A))];W=np.linalg.solve(A.T@A+1e3*np.eye(A.shape[1]),A.T@dmr[okr])
        ptr=np.c_[Xtr,np.ones(len(Xtr))]@W;pte=np.c_[Xte,np.ones(len(Xte))]@W
        o=evaluate('ridge',ptr,pte,L,h)
        if L in(0,10):
            m=lgb.LGBMRegressor(n_estimators=300,learning_rate=.05,num_leaves=63,min_child_samples=200,subsample=.7,subsample_freq=1,colsample_bytree=.8,verbose=-1,n_jobs=4)
            m.fit(Xtr[okr],dmr[okr]);ptr=m.predict(Xtr);pte=m.predict(Xte);o2=evaluate('lgbm',ptr,pte,L,h)
            if L==0 and h==50:res['lgbm_top_features_L0_h50']=[names[i] for i in np.argsort(m.booster_.feature_importance('gain'))[::-1][:10]]
            print(f'L={L*100}ms h={h/10}s ridge IC {o["ic"]:+.4f} top1%: move {o["top1pct"]["mean_signed_mid_move_ticks"]} taker {o["top1pct"]["taker_net_ticks_per_trade"]} | lgbm IC {o2["ic"]:+.4f} top1%: move {o2["top1pct"]["mean_signed_mid_move_ticks"]} taker {o2["top1pct"]["taker_net_ticks_per_trade"]} spread {o["spread_med_ticks"]}',flush=True)
        else:
            print(f'L={L*100}ms h={h/10}s ridge IC {o["ic"]:+.4f} top1%: move {o["top1pct"]["mean_signed_mid_move_ticks"]} taker {o["top1pct"]["taker_net_ticks_per_trade"]}',flush=True)
json.dump(res,open(OUT,'w'),indent=1);print('done %.0fs'%(time.time()-t0))
