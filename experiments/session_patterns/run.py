"""Session / event-window pattern scan on XAUUSD M1 with yearly walk-forward. See PROTOCOL.md."""
import json,sys,numpy as np,pandas as pd
from zoneinfo import ZoneInfo
SRC=sys.argv[1] if len(sys.argv)>1 else '/home/user/data_raw/XAUUSD_41.parquet'
OUT=sys.argv[2] if len(sys.argv)>2 else '/home/user/xauusd/reports/session_patterns/results.json'
d=pd.read_parquet(SRC,columns=['timestamp','open']);ts=pd.to_datetime(d.timestamp,utc=True).values.astype('datetime64[ns]').astype(np.int64);op=d.open.values
o=np.argsort(ts,kind='stable');ts,op=ts[o],op[o]
M=60*10**9
def px(t):
    i=np.searchsorted(ts,t);i=np.minimum(i,len(ts)-1);ok=(ts[i]>=t)&(ts[i]-t<=5*M);return np.where(ok,op[i],np.nan)
days=pd.bdate_range('2016-09-06','2026-09-03',tz='UTC');D0=days.values.astype('datetime64[ns]').astype(np.int64);year=days.year.values
def win(s_ns,e_ns):
    a=px(s_ns);b=px(e_ns);r=1e4*np.log(b/a);return r,a
def utc_h(h):return D0+int(h*60)*M
def local(hh,mm,tz):
    z=ZoneInfo(tz);return np.array([pd.Timestamp(year=x.year,month=x.month,day=x.day,hour=hh,minute=mm,tz=z).tz_convert('UTC').value for x in days],np.int64)
COST=0.25
SES=[('Sydney_only_23-00',utc_h(-1),utc_h(0)),('Sydney_Tokyo_00-07',utc_h(0),utc_h(7)),('Tokyo_London_07-09',utc_h(7),utc_h(9)),('London_09-12',utc_h(9),utc_h(12)),
     ('London_NY_12-16',utc_h(12),utc_h(16)),('NY_16-20',utc_h(16),utc_h(20)),('lateNY_20-21',utc_h(20),utc_h(21))]
ANCH=[('SGE_AM_10:15_BJ',local(10,15,'Asia/Shanghai')),('SGE_PM_14:15_BJ',local(14,15,'Asia/Shanghai')),('London_open_08:00',local(8,0,'Europe/London')),
      ('LBMA_AM_10:30_LDN',local(10,30,'Europe/London')),('US_data_08:30_NY',local(8,30,'America/New_York')),('US_equity_open_09:30_NY',local(9,30,'America/New_York')),
      ('NY_10:00_data',local(10,0,'America/New_York')),('COMEX_settle_13:30_NY',local(13,30,'America/New_York')),('LBMA_PM_15:00_LDN',local(15,0,'Europe/London')),
      ('WMR_16:00_LDN',local(16,0,'Europe/London')),('NY_close_17:00_NY',local(17,0,'America/New_York'))]
YEARS=range(2018,2027)
def wf(name,kind,y,a,x=None):
    """walk-forward: kind 'drift' (fixed sign from past mean) or 'pred' (sign of past slope * sign(x))."""
    cost=COST/a*1e4;ok=np.isfinite(y)&np.isfinite(cost)
    if x is not None:ok&=np.isfinite(x)&(x!=0)
    pnl=np.full(len(y),np.nan);per={}
    for Y in YEARS:
        past=ok&(year<Y);cur=ok&(year==Y)
        if past.sum()<100 or cur.sum()==0:continue
        if kind=='drift':
            s=np.sign(y[past].mean());pnl[cur]=s*y[cur]-cost[cur]
        else:
            b=np.sign(np.cov(x[past],y[past])[0,1]);pnl[cur]=b*np.sign(x[cur])*y[cur]-cost[cur]
        per[Y]=float(np.nansum(pnl[cur]))
    v=pnl[np.isfinite(pnl)];g=v+(COST/a*1e4)[np.isfinite(pnl)]
    t=float(v.mean()/(v.std()/np.sqrt(len(v)))) if len(v)>10 else 0
    res={'test':name,'kind':kind,'oos_days':int(len(v)),'net_bp_per_trade':float(v.mean()),'gross_bp_per_trade':float(g.mean()),'t_net':t,
         'positive_years':int(sum(1 for z in per.values() if z>0)),'years':len(per),'per_year_net_bp':per,
         'candidate':bool(t>=3.3 and sum(1 for z in per.values() if z>0)>=7)}
    if x is not None:res['corr_full']=float(np.corrcoef(x[ok],y[ok])[0,1])
    return res
R=[]
# overall gold drift per minute (for comparison with "always long in a window")
allr=1e4*np.log(op[-1]/op[0])/((ts[-1]-ts[0])/M)
S={}
for n,s,e in SES:
    r,a=win(s,e);S[n]=(r,a,s,e)
    res=wf(n,'drift',r,a);res['mean_bp']=float(np.nanmean(r));res['buyhold_share_bp']=float(allr*(e[0]-s[0])/M);R.append(res)
names=[n for n,_,_ in SES]
for i in range(len(names)):
    for j in range(i+1,len(names)):
        ra=S[names[i]][0];rb,ab=S[names[j]][0],S[names[j]][1];R.append(wf(f'{names[i]} -> {names[j]}','pred',rb,ab,ra))
for n,s,e in SES:
    if (e[0]-s[0])>60*M:
        r1,a1=win(s,s+30*M);r2,a2=win(s+30*M,e);R.append(wf(f'{n}: first30 -> rest','pred',r2,a2,r1))
for n,t in ANCH:
    pre,ap=win(t-30*M,t)
    for h in (15,30,60):
        post,a=win(t,t+h*M)
        R.append(wf(f'{n}: drift 0->+{h}','drift',post,a));R.append(wf(f'{n}: pre30 -> post{h}','pred',post,a,pre))
cand=[r for r in R if r['candidate']]
R_sorted=sorted(R,key=lambda r:-r['t_net'])
json.dump({'n_tests':len(R),'candidates':cand,'all':R_sorted},open(OUT,'w'),indent=1)
print('tests',len(R),'candidates',len(cand))
print(f"{'test':58} {'net bp':>7} {'gross':>6} {'t':>6} {'+yrs':>5} {'days':>5}")
for r in R_sorted[:25]:print(f"{r['test'][:58]:58} {r['net_bp_per_trade']:7.2f} {r['gross_bp_per_trade']:6.2f} {r['t_net']:6.2f} {r['positive_years']:>2}/{r['years']} {r['oos_days']:5d}")
print('--- worst 5');[print(f"{r['test'][:58]:58} {r['net_bp_per_trade']:7.2f} {r['t_net']:6.2f}") for r in R_sorted[-5:]]
print('--- session drifts (mean bp vs buy-and-hold share)')
for r in R:
    if r['kind']=='drift' and 'mean_bp' in r:print(f"{r['test']:22} mean {r['mean_bp']:+.2f} bp  buy&hold share {r['buyhold_share_bp']:+.2f} bp  wf net {r['net_bp_per_trade']:+.2f} t {r['t_net']:+.2f} +yrs {r['positive_years']}/{r['years']}")
