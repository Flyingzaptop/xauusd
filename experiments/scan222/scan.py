"""Same cheap battery for every instrument (gross edges, before costs). Tests:
 P15/P60: walk-forward (yearly) ridge on own-past features -> fwd return over next 15/60 bars, entry Open[t+2] (1-bar delay);
          OOS corr per year; top-5% |pred| (threshold from train) non-overlapping trades: gross bp/trade.
 VT/VR: trend-away-from / revert-to session VWAP after 3-sigma deviation (fixed target/stop, conservative same-bar rule): gross bp/trade.
Output: one row per instrument."""
import sys,json,glob,os,numpy as np,pandas as pd
sys.path.insert(0,os.path.dirname(__file__));import kernels as K
from multiprocessing import Pool
RAW='/home/user/data_raw/all222';OUT='/home/user/xauusd/reports/scan222'
def one(path):
    name=os.path.basename(path).replace('.parquet','')
    try:
        d=pd.read_parquet(path);d=d.sort_values('timestamp').drop_duplicates('timestamp')
        t=pd.to_datetime(d.timestamp,utc=True);O,H,L,C=[d[c].values.astype(float) for c in ['open','high','low','close']];V=d.volume.values.astype(float)
        ok=(O>0)&(H>0)&(L>0)&(C>0);O,H,L,C,V,t=O[ok],H[ok],L[ok],C[ok],V[ok],t[ok]
        tm=t.values.astype('datetime64[m]').astype(np.int64);N=len(C)
        if N<200_000:return {'name':name,'skip':'short history','bars':int(N)}
        sess=np.cumsum(np.r_[True,np.diff(tm)>30]);year=t.dt.year.values
        lc=np.log(C);r1=np.r_[0,np.diff(lc)];r1[np.r_[True,np.diff(sess)>0]]=0
        cs=np.cumsum(r1);cs2=np.cumsum(r1**2)
        def back(w):x=np.full(N,np.nan);x[w:]=cs[w:]-cs[:-w];return x
        def vol(w):m=np.full(N,np.nan);m[w:]=np.sqrt(np.maximum((cs2[w:]-cs2[:-w])/w-((cs[w:]-cs[:-w])/w)**2,0));return m
        tod=(tm%1440)/1440.0
        F=np.stack([back(1),back(5),back(15),back(60),back(240),back(1440),vol(60),vol(240),np.sin(2*np.pi*tod),np.cos(2*np.pi*tod)],1)
        res={'name':name,'bars':int(N),'first':str(t.iloc[0].date()),'last':str(t.iloc[-1].date()),'med_abs_1m_bp':float(np.median(np.abs(r1[r1!=0]))*1e4)}
        lo=np.log(O);dec=np.arange(1500,N-70,5)
        for h in (15,60):
            e=dec+2;x=e+h;okk=(x<N)&(sess[np.minimum(x,N-1)]==sess[dec])&np.isfinite(F[dec]).all(1)
            dd=dec[okk];y=(lo[x[okk]]-lo[e[okk]])*1e4;X=F[dd];yr=year[dd];cors={};gross=[];per={}
            for Y in range(int(yr.min())+1,int(yr.max())+1):
                tr=yr<Y;te=yr==Y
                if tr.sum()<20000 or te.sum()<2000:continue
                mu=X[tr].mean(0);sd=X[tr].std(0)+1e-12;A=(X[tr]-mu)/sd;W=np.linalg.solve(A.T@A+1e3*np.eye(A.shape[1]),A.T@(y[tr]-y[tr].mean()))
                p=((X[te]-mu)/sd)@W;cors[Y]=float(np.corrcoef(p,y[te])[0,1])
                thr=np.quantile(np.abs(A@W),.95);s=np.where(np.abs(p)>thr)[0];s=s[K.nonoverlap(dd[te][s],h)]
                g=np.sign(p[s])*y[te][s];gross.append(g);per[Y]=float(g.sum())
            if cors:
                g=np.concatenate(gross) if gross else np.array([0.]);v=np.array(list(cors.values()))
                res[f'P{h}_corr']=float(v.mean());res[f'P{h}_corr_pos_years']=f'{int((v>0).sum())}/{len(v)}'
                res[f'P{h}_gross_bp']=float(g.mean());res[f'P{h}_t']=float(g.mean()/(g.std()/np.sqrt(len(g))+1e-12));res[f'P{h}_trades']=int(len(g))
                res[f'P{h}_pos_years']=f'{sum(1 for z in per.values() if z>0)}/{len(per)}'
        vw=(pd.Series(C*np.maximum(V,1)).groupby(sess).cumsum()/pd.Series(np.maximum(V,1)).groupby(sess).cumsum()).values
        dev=C-vw;sd_=pd.Series(dev).rolling(120,min_periods=60).std().values
        for mode,lab in ((1,'VT'),(0,'VR')):
            a=(dev>=3*sd_)&(sd_>0);b=(dev<=-3*sd_)&(sd_>0);out=np.zeros((N,2));n=K.run_line(a,b,np.nan_to_num(vw),H,L,C,sess,240,mode,out)
            g=out[:n,1];yy=year[out[:n,0].astype(int)]
            if n>100:
                per={int(Y):float(g[yy==Y].sum()) for Y in np.unique(yy)}
                res[f'{lab}_gross_bp']=float(g.mean());res[f'{lab}_t']=float(g.mean()/(g.std()/np.sqrt(n)));res[f'{lab}_trades']=int(n);res[f'{lab}_pos_years']=f'{sum(v>0 for v in per.values())}/{len(per)}'
        return res
    except Exception as ex:
        return {'name':name,'error':repr(ex)[:200]}
if __name__=='__main__':
    files=sorted(glob.glob(RAW+'/*.parquet'))
    with Pool(4) as pool:rows=pool.map(one,files,chunksize=1)
    json.dump(rows,open(OUT+'/scan.json','w'),indent=1);pd.DataFrame(rows).to_csv(OUT+'/scan.csv',index=False);print('done',len(rows))
