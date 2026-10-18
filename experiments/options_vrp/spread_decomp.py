"""Decompose short-ATM-straddle P&L on daily Deribit options: P&L at mark (pure premium vs realized) vs half-spread paid as taker."""
import sys,numpy as np,pandas as pd
sys.argv=['vrp.py','/home/user/crypto/deribit','/dev/null'];src=open('vrp.py').read().split('out={}')[0];exec(src)
for cur in ('BTC','ETH'):
    t,dl=load(cur);t=t[t.direction=='sell'];rows=[]
    for day,g in t.groupby('day'):
        e=day+pd.Timedelta(days=1)
        if e not in dl.index:continue
        S0=g.index_price.mean();ST=dl[e];r={}
        for cp in ('C','P'):
            h=g[g.cp==cp]
            if not len(h):break
            K=h.K.values[np.argmin(np.abs(h.K.values-S0))];h=h[h.K==K]
            r[cp]=(K,np.average(h.price,weights=h.amount),np.average(h.mark_price,weights=h.amount))
        if len(r)<2:continue
        intr=max(ST-r['C'][0],0)+max(r['P'][0]-ST,0)
        rows.append(dict(day=day,mark=((r['C'][2]+r['P'][2])*S0-intr)/S0*1e4,fill=((r['C'][1]+r['P'][1])*S0-intr)/S0*1e4,prem_mark=(r['C'][2]+r['P'][2])*1e4))
    R=pd.DataFrame(rows);R['halfspread']=R.mark-R.fill
    print(cur,len(R));print(R.groupby(R.day.dt.year)[['prem_mark','mark','fill','halfspread']].mean().round(1).to_string())
