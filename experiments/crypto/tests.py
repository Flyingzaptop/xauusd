"""Crypto test battery C1-C4 (see PROTOCOL.md)."""
import sys,os,json,numpy as np,pandas as pd
sys.path.insert(0,'/home/user/xauusd/experiments/scan222');import kernels as K
COSTS=(4,7,10);OUT='/home/user/xauusd/reports/crypto';os.makedirs(OUT,exist_ok=True)
def summ(name,idx,g,year,yrs=range(2021,2027)):
    m=np.isin(year[idx],list(yrs));g=g[m];idx=idx[m]
    if len(g)<30:return None
    r={'test':name,'n':int(len(g)),'gross_bp':float(g.mean()),'t_gross':float(g.mean()/(g.std()/np.sqrt(len(g))))}
    for c in COSTS:
        v=g-c;per={int(Y):float(v[year[idx]==Y].sum()) for Y in yrs if (year[idx]==Y).any()}
        r[f'net{c}']=float(v.mean());r[f't{c}']=float(v.mean()/(v.std()/np.sqrt(len(v))));r[f'pos{c}']=f"{sum(x>0 for x in per.values())}/{len(per)}"
    r['gross_per_year']={int(Y):float(g[year[idx]==Y].mean()) for Y in yrs if (year[idx]==Y).any()}
    r['candidate']=any(r[f'net{c}']>0 and r[f't{c}']>=3 and int(r[f'pos{c}'].split('/')[0])>=5 for c in COSTS)
    return r
def wf_dir(x_sign,y,year,yrs=range(2021,2027)):
    """direction per year from the past: d_Y = sign(mean(x_sign*y) over years < Y)."""
    out=np.full(len(y),np.nan)
    for Y in yrs:
        p=year<Y;c=year==Y
        if p.sum()<30 or c.sum()==0:continue
        out[c]=np.sign(np.nanmean(x_sign[p]*y[p]))*x_sign[c]*y[c]
    return out
RES=[]
for S in ('BTCUSDT','ETHUSDT'):
    k=pd.read_parquet(f'/home/user/crypto/{S}_1m.parquet');t=pd.to_datetime(k.open_time,unit='ms',utc=True)
    O,H,L,C,V,TB=[k[c].values for c in ['open','high','low','close','volume','taker_buy_volume']];N=len(C);year=t.dt.year.values
    dayid=(k.open_time.values//86_400_000);one=np.zeros(N,np.int64)
    # ---- C1 VWAP
    vw=(pd.Series(C*V).groupby(dayid).cumsum()/pd.Series(V).groupby(dayid).cumsum().replace(0,np.nan)).values
    dev=C-vw;sd=pd.Series(dev).rolling(120,min_periods=60).std().values
    for kk in (2,3,4):
        a=(dev>=kk*sd)&(sd>0);b=(dev<=-kk*sd)&(sd>0)
        for mode,lab in ((1,'trend'),(0,'revert')):
            out=np.zeros((N,2));n=K.run_line(a,b,np.nan_to_num(vw),H,L,C,one,240,mode,out)
            r=summ(f'{S} C1 VWAP {lab} {kk}sd',out[:n,0].astype(int),out[:n,1],year);r and RES.append(r)
    # ---- C2 order flow model
    lc=np.log(C);r1=np.r_[0,np.diff(lc)];cs=np.cumsum(r1);cs2=np.cumsum(r1**2);imb=np.where(V>0,(2*TB-V)/np.maximum(V,1e-12),0);ci=np.cumsum(imb*V);cv=np.cumsum(V)
    def back(c,w):x=np.full(N,np.nan);x[w:]=c[w:]-c[:-w];return x
    def vol(w):m=np.full(N,np.nan);m[w:]=np.sqrt(np.maximum((cs2[w:]-cs2[:-w])/w-((cs[w:]-cs[:-w])/w)**2,0));return m
    def oimb(w):return back(ci,w)/np.maximum(back(cv,w),1e-9)
    tod=(k.open_time.values//60000%1440)/1440
    F=np.stack([back(cs,1),back(cs,5),back(cs,15),back(cs,60),back(cs,240),back(cs,1440),vol(60),vol(240),oimb(1),oimb(5),oimb(15),oimb(60),np.sin(2*np.pi*tod),np.cos(2*np.pi*tod)],1)
    lo=np.log(O);dec=np.arange(1500,N-70,5)
    for h in (15,60):
        e=dec+2;x=e+h;ok=(x<N)&np.isfinite(F[dec]).all(1);dd=dec[ok];y=(lo[x[ok]]-lo[e[ok]])*1e4;X=F[dd];yr=year[dd];I=[];G=[];cors={}
        for Y in range(2021,2027):
            tr=yr<Y;te=yr==Y
            mu=X[tr].mean(0);s_=X[tr].std(0)+1e-12;A=(X[tr]-mu)/s_;W=np.linalg.solve(A.T@A+1e3*np.eye(A.shape[1]),A.T@(y[tr]-y[tr].mean()))
            p=((X[te]-mu)/s_)@W;cors[Y]=float(np.corrcoef(p,y[te])[0,1]);thr=np.quantile(np.abs(A@W),.95)
            sel=np.where(np.abs(p)>thr)[0];sel=sel[K.nonoverlap(dd[te][sel],h)];I.append(dd[te][sel]);G.append(np.sign(p[sel])*y[te][sel])
        r=summ(f'{S} C2 orderflow ridge h{h}',np.concatenate(I),np.concatenate(G),year);r['oos_corr_by_year']=cors;RES.append(r)
    # ---- C3 funding
    fr=pd.read_parquet(f'/home/user/crypto/{S}_funding.parquet').sort_values('calc_time');ft=fr.calc_time.values;fv=fr.last_funding_rate.values.astype(float)
    z=fv/pd.Series(fv).rolling(270,min_periods=90).std().shift(1).values
    fi=np.searchsorted(k.open_time.values,ft)+1;okf=(fi+1440<N)
    for hh in (480,1440):
        y=np.full(len(ft),np.nan);y[okf]=(lo[fi[okf]+hh]-lo[fi[okf]])*1e4;fy=year[np.minimum(fi,N-1)]
        m=okf&(np.abs(z)>2)&np.isfinite(z);g=wf_dir(np.sign(fv[m]),y[m],fy[m]);gi=fi[m]
        keep=np.isfinite(g);r=summ(f'{S} C3 funding |z|>2 -> {hh//60}h',gi[keep],g[keep],year);r and RES.append(r)
    # ---- C4 shock bars
    sd1=pd.Series(r1).rolling(1440,min_periods=720).std().shift(1).values;mv=pd.Series(V).rolling(1440,min_periods=720).median().shift(1).values
    sh=np.where((np.abs(r1)>=5*sd1)&(V>=5*mv))[0];sh=sh[sh+62<N];sh=sh[K.nonoverlap(sh,60)]
    for hh in (5,15,60):
        y=(lo[sh+1+hh]-lo[sh+1])*1e4;g=wf_dir(np.sign(r1[sh]),y,year[sh]);keep=np.isfinite(g)
        r=summ(f'{S} C4 shock bar -> {hh}m',sh[keep],g[keep],year);r and RES.append(r)
json.dump(RES,open(OUT+'/tests.json','w'),indent=1)
print(f"{'test':42} {'n':>6} {'gross':>7} {'t':>5} | net@4 t pos | net@7 t pos | net@10 t pos")
for r in RES:print(f"{r['test']:42} {r['n']:6d} {r['gross_bp']:+7.2f} {r['t_gross']:+5.1f} | {r['net4']:+6.2f} {r['t4']:+5.1f} {r['pos4']} | {r['net7']:+6.2f} {r['t7']:+5.1f} {r['pos7']} | {r['net10']:+6.2f} {r['t10']:+5.1f} {r['pos10']}{'  <== CANDIDATE' if r['candidate'] else ''}")
