"""Event-structure study on XAUUSD M1 (Part 2 of PROTOCOL.md)."""
import json,sys,numpy as np,pandas as pd
from zoneinfo import ZoneInfo
OUT='/home/user/xauusd/reports/session_patterns/event_study.json'
d=pd.read_parquet('/home/user/data_raw/XAUUSD_41.parquet');t=pd.to_datetime(d.timestamp,utc=True)
day0=t.dt.floor('D');cal=pd.date_range(day0.min(),day0.max(),freq='D',tz='UTC');di=((day0-cal[0]).dt.days).values;mi=(t.dt.hour*60+t.dt.minute).values
ND=len(cal);sh=(ND,1440)
O=np.full(sh,np.nan,np.float32);H=O.copy();L=O.copy();C=O.copy();V=np.zeros(sh,np.float32)
O[di,mi]=d.open;H[di,mi]=d.high;L[di,mi]=d.low;C[di,mi]=d.close;V[di,mi]=d.volume
Of=O.ravel();Hf=H.ravel();Lf=L.ravel();Cf=C.ravel();Vf=V.ravel();NT=len(Of)
wd=cal.dayofweek.values;year=cal.year.values;rows=np.where(wd<5)[0]   # trading days = UTC weekdays
def anchor(h,m,tz):
    z=ZoneInfo(tz);return np.array([int((pd.Timestamp(year=x.year,month=x.month,day=x.day,hour=h,minute=m,tz=z).tz_convert('UTC')-x).total_seconds()//60) for x in cal[rows]])
def fixed(h):return np.full(len(rows),h*60)
EV={'UTC00 Tokyo':fixed(0),'UTC07 TokLon':fixed(7),'UTC09 London':fixed(9),'UTC12 LonNY':fixed(12),'UTC16 NY':fixed(16),'UTC20 lateNY':fixed(20),
    'SGE 10:15 BJ':anchor(10,15,'Asia/Shanghai'),'SGE 14:15 BJ':anchor(14,15,'Asia/Shanghai'),'London open 08:00':anchor(8,0,'Europe/London'),
    'LBMA AM 10:30':anchor(10,30,'Europe/London'),'US data 08:30':anchor(8,30,'America/New_York'),'US equity 09:30':anchor(9,30,'America/New_York'),
    'NY 10:00':anchor(10,0,'America/New_York'),'COMEX 13:30':anchor(13,30,'America/New_York'),'LBMA PM 15:00':anchor(15,0,'Europe/London'),
    'WMR 16:00':anchor(16,0,'Europe/London'),'CME reopen 18:00':anchor(18,0,'America/New_York')}
gidx=lambda m:rows*1440+m          # flat index; minute can be negative / >1439 (crosses into neighbour day)
def px_open(f):                     # open at flat index f, or first bar within 5 min after
    out=np.full(len(f),np.nan)
    for k in range(6):
        g=np.clip(f+k,0,NT-1);v=Of[g];m=np.isnan(out)&~np.isnan(v)&(f+k<NT)&(f+k>=0);out[m]=v[m]
    return out
def ret(f0,f1):return 1e4*np.log(px_open(f1)/px_open(f0))
YR=year[rows];COST=.25
def wf(name,y,a,sig=None,binned=None):
    """walk-forward. sig: per-day predictor (sign used); binned: per-day bin labels -> direction per bin from the past."""
    cost=COST/a*1e4;ok=np.isfinite(y)&np.isfinite(cost)
    if sig is not None:ok&=np.isfinite(sig)&(sig!=0)
    if binned is not None:ok&=(binned>=0)
    pnl=np.full(len(y),np.nan);per={}
    for Y in range(2018,2027):
        p=ok&(YR<Y);c=ok&(YR==Y)
        if p.sum()<60 or c.sum()==0:continue
        if binned is not None:
            for b in np.unique(binned[c]):
                pb=p&(binned==b);cb=c&(binned==b)
                if pb.sum()>=30:pnl[cb]=np.sign(y[pb].mean())*y[cb]-cost[cb]
        elif sig is not None:
            s=np.sign(np.mean(np.sign(sig[p])*y[p]));pnl[c]=s*np.sign(sig[c])*y[c]-cost[c]
        else:pnl[c]=np.sign(y[p].mean())*y[c]-cost[c]
        per[Y]=float(np.nansum(pnl[c]))
    v=pnl[np.isfinite(pnl)]
    if len(v)<30:return None
    tt=float(v.mean()/(v.std()/np.sqrt(len(v))));py=sum(1 for z in per.values() if z>0)
    return {'test':name,'n':int(len(v)),'net_bp':float(v.mean()),'gross_bp':float(v.mean()+np.nanmean(cost[np.isfinite(pnl)])),'t':tt,'pos_years':py,'years':len(per),'candidate':bool(tt>=3.3 and py>=7),'per_year':per}
RES=[];PROF={}
r1=np.full(NT,np.nan,np.float32);r1[1:]=1e4*np.log(Cf[1:]/Cf[:-1])  # 1-min close returns (NaN across gaps)
for name,m in EV.items():
    f=gidx(m)
    # A. profile
    offs=np.arange(-60,91);ar=np.array([np.nanmean(np.abs(r1[np.clip(f+o,0,NT-1)])) for o in offs]);vv=np.array([np.nanmean(Vf[np.clip(f+o,0,NT-1)]) for o in offs])
    base_r=ar[:30].mean();base_v=vv[:30].mean();k=np.nanargmax(ar[55:75])+55
    PROF[name]={'peak_abs_ret_ratio':float(ar[k]/base_r),'peak_offset_min':int(offs[k]),'vol_ratio_0_5':float(vv[60:66].mean()/base_v),'abs_ret_ratio_0_15':float(ar[60:75].mean()/base_r)}
    # B. conditional pre->post
    pre=ret(f-30,f)
    sd=pd.Series(pre).rolling(250,min_periods=100).std().shift(1).values;z=pre/sd
    for h in (15,30,60):
        post=ret(f,f+h);a=px_open(f)
        for thr in (1,2):
            b=np.where(np.isfinite(z),np.where(z>thr,1,np.where(z<-thr,0,-1)),-1)
            r=wf(f'{name}: pre30 |z|>{thr} -> post{h}',post,a,binned=b);r and RES.append(r)
    # D. volatility
    vol_pre=np.array([np.nanstd(r1[np.clip(f+o,0,NT-1)]) for o in []]) if False else None
# C. ranges
f00=gidx(fixed(0));f07=gidx(fixed(7))
def window_hl(fa,fb):
    hh=np.full(len(fa),-np.inf);ll=np.full(len(fa),np.inf)
    for k in range(int((fb-fa).max())):
        g=np.clip(fa+k,0,NT-1);m=(fa+k)<fb;hh=np.where(m,np.fmax(hh,Hf[g]),hh);ll=np.where(m,np.fmin(ll,Lf[g]),ll)
    hh[~np.isfinite(hh)]=np.nan;ll[~np.isfinite(ll)]=np.nan;return hh,ll
def first_break(fa,fb,hh,ll):
    side=np.zeros(len(fa));when=np.full(len(fa),-1)
    for k in range(int((fb-fa).max())):
        g=np.clip(fa+k,0,NT-1);m=(side==0)&((fa+k)<fb)
        up=m&(Hf[g]>hh);dn=m&(Lf[g]<ll)&~up
        side[up]=1;when[up]=fa[up]+k;side[dn]=-1;when[dn]=fa[dn]+k
    return side,when
def add_break_tests(tag,side,when,ends):
    ok=side!=0;e=np.where(ok,when+1,0)   # entry: open of next bar
    a=px_open(e)
    for lab,x in ends.items():
        y=np.where(ok,ret(e,x if np.ndim(x) else e+x),np.nan);r=wf(f'{tag} -> {lab}',y,a,sig=np.where(ok,side,np.nan));r and RES.append(r)
ah,al=window_hl(f00,f07);s,w=first_break(f07,gidx(fixed(10)),ah,al)
add_break_tests('Asian range 00-07 first break 07-10 UTC',s,w,{'+30':30,'+60':60,'+120':120,'16:00 UTC':gidx(fixed(16))})
# sweep: after first break, a close back inside the range within 30 min
sw=np.zeros(len(s));sww=np.full(len(s),-1)
for k in range(1,31):
    g=np.clip(w+k,0,NT-1);m=(s!=0)&(sw==0)&(w>=0)
    back=m&(((s>0)&(Cf[g]<ah))|((s<0)&(Cf[g]>al)));sw[back]=-s[back];sww[back]=w[back]+k
add_break_tests('Asian range sweep (break, close back inside <=30m)',sw,sww,{'+30':30,'+60':60,'+120':120,'16:00 UTC':gidx(fixed(16))})
for nm,m0,endh in [('London open 08:00 LDN',EV['London open 08:00'],16),('US 08:30 NY',EV['US data 08:30'],20)]:
    fa=gidx(m0);oh,ol=window_hl(fa,fa+30);s2,w2=first_break(fa+30,fa+150,oh,ol)
    add_break_tests(f'Opening range 30m {nm}, break within 2h',s2,w2,{'+60':60,f'{endh}:00 UTC':gidx(fixed(endh))})
# previous-day high/low taken during London (07-12) / NY (12-17)
prev_h,prev_l=window_hl(gidx(fixed(-24)),gidx(fixed(0)))
for nm,a0,b0,endh in [('London 07-12',7,12,16),('NY 12-17',12,17,20)]:
    s3,w3=first_break(gidx(fixed(a0)),gidx(fixed(b0)),prev_h,prev_l)
    add_break_tests(f'Previous-day H/L taken in {nm}',s3,w3,{'+30':30,'+120':120,f'{endh}:00 UTC':gidx(fixed(endh))})
# C5 descriptive: hour of daily high / low (UTC 00-21)
hi_h=[];lo_h=[]
for r_ in rows:
    hh=H[r_,:21*60];ll=L[r_,:21*60]
    if np.isfinite(hh).sum()>600:hi_h.append(np.nanargmax(hh)//60);lo_h.append(np.nanargmin(ll)//60)
dist={'high':np.bincount(hi_h,minlength=21).tolist(),'low':np.bincount(lo_h,minlength=21).tolist()}
cand=[r for r in RES if r['candidate']];RES.sort(key=lambda r:-r['t'])
json.dump({'n_tests':len(RES),'candidates':cand,'tests':RES,'profiles':PROF,'hour_of_daily_high_low':dist},open(OUT,'w'),indent=1)
print('== A. activity profile (|1-min return| ratio vs -60..-31 baseline)')
for k,v in sorted(PROF.items(),key=lambda x:-x[1]['abs_ret_ratio_0_15']):print(f"{k:20} first-15m |ret| x{v['abs_ret_ratio_0_15']:.2f}  peak x{v['peak_abs_ret_ratio']:.2f} at {v['peak_offset_min']:+d}m  tick-vol 0-5m x{v['vol_ratio_0_5']:.2f}")
print(f'== B/C tests: {len(RES)}, candidates: {len(cand)}')
for r in RES[:15]:print(f"{r['test'][:70]:70} net {r['net_bp']:+6.2f} gross {r['gross_bp']:+6.2f} t {r['t']:+5.2f} +yrs {r['pos_years']}/{r['years']} n {r['n']}")
print('== C5 hour (UTC) of daily high:',dist['high']);print('   hour (UTC) of daily low :',dist['low'])
