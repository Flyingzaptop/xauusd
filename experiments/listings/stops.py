"""H2: short new listings with a protective buy-stop (see PROTOCOL.md amendment). usage: stops.py DATA_DIR OUT.json"""
import sys,glob,json,numpy as np,pandas as pd
D,OUT=sys.argv[1:3];SL=[0.2,0.3,0.5,1.0,None];SLIP=0.02;COST=30
rows=[]
for fn in glob.glob(f'{D}/*USDT.parquet'):
    s=fn.split('/')[-1][:-8]
    if s=='BTCUSDT':continue
    k=pd.read_parquet(fn).sort_values('t').reset_index(drop=True);f=pd.read_parquet(fn.replace('.parquet','_fund.parquet'))
    if len(k)<35:continue
    L=pd.to_datetime(k.t.iloc[0],unit='ms')
    if not (pd.Timestamp('2021-01-01')<=L<=pd.Timestamp('2026-08-31')):continue
    o0=k.o.iloc[1];t0=k.t.iloc[1]
    for h in (7,14,30):
        for sl in SL:
            ex=k.o.iloc[1+h];tex=k.t.iloc[1+h];stopped=False
            if sl is not None:
                st=o0*(1+sl)
                for j in range(1,1+h):
                    if k.o.iloc[j]>=st:ex=k.o.iloc[j]*(1+SLIP);tex=k.t.iloc[j];stopped=True;break
                    if k.h.iloc[j]>=st:ex=st*(1+SLIP);tex=k.t.iloc[j]+86400000;stopped=True;break
            fund=f.r[(f.t>=t0)&(f.t<tex)].sum()*1e4 if len(f) else 0.
            rows.append(dict(sym=s,L=L,h=h,sl=-1 if sl is None else sl,pnl=-(ex/o0-1)*1e4+fund-COST,stopped=stopped))
R=pd.DataFrame(rows);R.to_parquet(f'{D}/rows_stops.parquet');out={}
for h in (7,14,30):
    for sl in [-1]+SL[:-1]:
        q=R[(R.h==h)&(R.sl==sl)]
        for per,(a,b) in (('2021-23',('2021','2024')),('2024-26',('2024','2027'))):
            x=q[(q.L>=a)&(q.L<b)];m=x.groupby(x.L.dt.to_period('M')).pnl.sum()     # 1 unit per listing, monthly P&L
            mm=x.groupby(x.L.dt.to_period('M')).pnl.mean();eq=m.cumsum()
            r={'n':int(len(x)),'mean':float(x.pnl.mean()),'median':float(x.pnl.median()),'stopped':float(x.stopped.mean()),'worst':float(x.pnl.min()),
               't_months':float(mm.mean()/mm.std()*np.sqrt(len(mm))),'monthly_sharpe_ann':float(m.mean()/m.std()*np.sqrt(12)),'maxdd_units_bp':float((eq.cummax()-eq).max()),
               'by_year':{str(y):float(x.pnl[x.L.dt.year==y].mean()) for y in sorted(set(x.L.dt.year))}}
            out[f'{h}d|sl{sl}|{per}']=r
            print(h,'sl',sl,per,r['n'],'mean %.0f med %.0f stop %.2f t %.2f shp %.2f'%(r['mean'],r['median'],r['stopped'],r['t_months'],r['monthly_sharpe_ann']),{k:round(v) for k,v in r['by_year'].items()},flush=True)
json.dump(out,open(OUT,'w'),indent=1)
