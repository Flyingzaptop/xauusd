"""Range / resting-order study (see PROTOCOL.md). usage: run.py OUT.json"""
import sys,json,numpy as np,pandas as pd,lightgbm as lgb,numba as nb
HS=[15,60,240];ALPHAS=[0.7,0.8,0.9];MK,TK,SLIP=2.0,5.0,1.0
FIT=('2020-01-01','2024-01-01');VAL=('2024-01-01','2025-01-01');TEST=('2025-01-01','2026-10-01')
def load(s):
    d=pd.read_parquet(f'/home/user/crypto/{s}_1m.parquet');d.index=pd.to_datetime(d.open_time,unit='ms');d=d[~d.index.duplicated()]
    d=d.reindex(pd.date_range(d.index[0],d.index[-1],freq='1min'));d['close']=d.close.ffill()
    for c in ('open','high','low'):d[c]=d[c].fillna(d.close)
    d[['volume','taker_buy_volume']]=d[['volume','taker_buy_volume']].fillna(0);return d
def features(d):
    c=np.log(d.close);r=c.diff().fillna(0);X=pd.DataFrame(index=d.index)
    for m in (15,60,240,1440,10080):X[f'rv{m}']=np.sqrt((r**2).rolling(m,min_periods=m//2).mean())
    for m in (15,60,240,1440):X[f'ret{m}']=c-c.shift(m)
    for m in (60,240):X[f'tb{m}']=d.taker_buy_volume.rolling(m).sum()/(d.volume.rolling(m).sum()+1e-9)-.5
    X['dhi']=np.log(d.high.rolling(1440).max())-c;X['dlo']=c-np.log(d.low.rolling(1440).min())
    lv=np.log1p(d.volume);X['vz']=(lv.rolling(60).mean()-lv.rolling(10080).mean())/(lv.rolling(10080).std()+1e-9)
    X['hour']=d.index.hour+d.index.minute/60;X['wd']=d.index.weekday
    return X
def targets(d,H):
    c=np.log(d.close.values);hi=np.log(d.high.values);lo=np.log(d.low.values);n=len(c)
    mh=pd.Series(hi[::-1]).rolling(H,min_periods=H).max().values[::-1];ml=pd.Series(lo[::-1]).rolling(H,min_periods=H).min().values[::-1]
    U=np.full(n,np.nan);D=np.full(n,np.nan);U[:-1]=mh[1:]-c[:-1];D[:-1]=c[:-1]-ml[1:]   # excursions over bars t+1..t+H
    return U,D
@nb.njit(cache=False)
def sim(o,h,l,c,idx,up,dn,H,mode,tp_sl):
    """For each decision bar i (row idx[k]) with band distances up/dn (log): returns (pnl_bp gross, side, filled) per decision.
    mode 0 FADE (limit, trade-through 1bp), 1 BREAKOUT (stop orders). tp_sl: FADE exit at decision price / stop at 2x band."""
    K=len(idx);pnl=np.zeros(K);side=np.zeros(K,np.int8);exm=np.zeros(K,np.int8)
    for k in range(K):
        i=idx[k];c0=c[i];U=c0*np.exp(up[k]);D=c0*np.exp(-dn[k]);thr=1e-4
        pos=0;ent=0.;j0=-1
        for j in range(i+1,i+H+1):
            if mode==0:
                tu=h[j]>=U*(1+thr);td=l[j]<=D*(1-thr)
                if tu and td:
                    if c[j]>c0:pos=-1;ent=U
                    else:pos=1;ent=D
                elif tu:pos=-1;ent=U
                elif td:pos=1;ent=D
            else:
                tu=h[j]>=U;td=l[j]<=D
                if tu and td:
                    if c[j]>c0:pos=-1;ent=D
                    else:pos=1;ent=U
                elif tu:pos=1;ent=U
                elif td:pos=-1;ent=D
            if pos!=0:j0=j;break
        if pos==0:continue
        side[k]=pos;ex=c[i+H];em=0
        if mode==0 and tp_sl:
            sl=ent*np.exp(-pos*2*(up[k] if pos<0 else dn[k]));tp=c0
            # from the fill bar onward: stop first (conservative); the fill bar itself only checks the stop
            for j in range(j0,i+H+1):
                if (pos>0 and l[j]<=sl) or (pos<0 and h[j]>=sl):ex=sl;em=1;break
                if j>j0 and ((pos>0 and h[j]>=tp*(1+thr)) or (pos<0 and l[j]<=tp*(1-thr))):ex=tp;em=2;break
        pnl[k]=pos*(ex/ent-1)*1e4;exm[k]=em
    return pnl,side,exm
def cost(mode,tp_sl,exm):
    if mode==1:return np.full(len(exm),2*TK+SLIP)
    return np.where(exm==2,MK+MK,MK+TK)
def stats(p):
    n=len(p);return {'n':n,'mean':float(p.mean()) if n else 0.,'t':float(p.mean()/(p.std()/np.sqrt(n))) if n>2 and p.std()>0 else 0.}
def main():
    out={}
    for s in ('BTCUSDT','ETHUSDT'):
        d=load(s);X=features(d);o,h,l,c=(d[k].values for k in ('open','high','low','close'));T=d.index;out[s]={}
        for H in HS:
            U,Dn=targets(d,H);ok=X.notna().all(1).values&~np.isnan(U)
            per={nm:np.where(ok&(T>=a)&(T<b))[0] for nm,(a,b) in zip(('fit','val','test'),(FIT,VAL,TEST))}
            fi=per['fit'][::5];res={}
            sig=X['rv60'].values*np.sqrt(H)
            for al in ALPHAS:
                q={}
                for nm,y in (('U',U),('D',Dn)):
                    m=lgb.train(dict(objective='quantile',alpha=al,learning_rate=0.05,num_leaves=31,min_data_in_leaf=500,feature_fraction=.8,bagging_fraction=.7,bagging_freq=1,verbose=-1),lgb.Dataset(X.values[fi],y[fi]),300)
                    q[nm]=m
                for nm in ('val','test'):
                    ii=per[nm];ii=ii[ii%H==0];ii=ii[ii+H<len(c)]
                    qu=np.maximum(q['U'].predict(X.values[ii]),1e-5);qd=np.maximum(q['D'].predict(X.values[ii]),1e-5)
                    cov={'U':float((U[ii]<=qu).mean()),'D':float((Dn[ii]<=qd).mean())}
                    if nm=='val':   # naive control: k*sigma with the same empirical coverage as the model on validation
                        ku=np.quantile(U[ii]/sig[ii],cov['U']);kd=np.quantile(Dn[ii]/sig[ii],cov['D'])
                    bands={'model':(qu,qd),'naive':(np.maximum(ku*sig[ii],1e-5),np.maximum(kd*sig[ii],1e-5))}
                    for bn,(bu,bd) in bands.items():
                        for mode,tps,lab in ((0,False,'fade_hold'),(0,True,'fade_tpsl'),(1,False,'breakout')):
                            p,sd,em=sim(o,h,l,c,ii.astype(np.int64),bu,bd,H,mode,tps);f=sd!=0
                            g=p[f];nt=g-cost(mode,tps,em[f])
                            res[f'{nm}|a{al}|{bn}|{lab}']={'coverage':cov,'fill_rate':float(f.mean()),'gross':stats(g),'net':stats(nt),
                                'win':float((nt>0).mean()) if f.any() else 0.,'by_year':{str(y):stats(nt[T[ii[f]].year==y]) for y in sorted(set(T[ii[f]].year))}}
            out[s][f'H{H}']=res
            best=sorted(((v['net']['mean'],k) for k,v in res.items() if k.startswith('val|')),reverse=True)[:3]
            print(s,H,'top val:',[(k,round(m,2),round(res[k.replace('val|','test|')]['net']['mean'],2)) for m,k in best],flush=True)
    json.dump(out,open(sys.argv[1],'w'),indent=1)
main()
