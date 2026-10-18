"""Deribit public history: all option trades in a daily entry window (default 08:00-10:00 UTC) + delivery prices.
usage: download.py CUR FIRST_DAY LAST_DAY OUTDIR   (one parquet per month; resumable)"""
import sys,os,time,json,urllib.request,datetime as dt,pandas as pd
cur,first,last,out=sys.argv[1:5];os.makedirs(out,exist_ok=True)
H='https://history.deribit.com/api/v2/public/'
def get(url,tries=6):
    for k in range(tries):
        try:return json.load(urllib.request.urlopen(url,timeout=60))['result']
        except Exception as e:
            if k==tries-1:raise
            time.sleep(2**k)
def window(t0,t1):
    rows=[];s=t0
    while True:
        r=get(f'{H}get_last_trades_by_currency_and_time?currency={cur}&kind=option&start_timestamp={s}&end_timestamp={t1}&count=1000&sorting=asc')
        tr=r['trades'];rows+=tr
        if not r.get('has_more') or not tr:break
        s=tr[-1]['timestamp']+1          # may drop a few same-millisecond trades at page edges; harmless here
    return rows
d=dt.date.fromisoformat(first);end=dt.date.fromisoformat(last);buf={}
while d<=end:
    mo=d.strftime('%Y-%m');fn=f'{out}/{cur}_opt_{mo}.parquet'
    if os.path.exists(fn):d=(d.replace(day=1)+dt.timedelta(days=32)).replace(day=1);continue
    t0=int(dt.datetime(d.year,d.month,d.day,8,tzinfo=dt.timezone.utc).timestamp()*1000);rows=window(t0,t0+2*3600*1000)
    buf.setdefault(mo,[]).extend(rows);nd=d+dt.timedelta(days=1)
    if nd.month!=d.month or nd>end:
        df=pd.DataFrame(buf.pop(mo));df=df.drop_duplicates('trade_id') if len(df) else df
        cols=[c for c in ('timestamp','instrument_name','price','mark_price','iv','index_price','direction','amount','trade_id') if c in df]
        df[cols].to_parquet(fn);print(cur,mo,len(df),flush=True)
    d=nd
# delivery prices (settlement), full history
idx=f'{cur.lower()}_usd';allp=[];off=0
while True:
    r=get(f'https://www.deribit.com/api/v2/public/get_delivery_prices?index_name={idx}&offset={off}&count=1000');allp+=r['data']
    if not r['data']:break
    off+=len(r['data'])
pd.DataFrame(allp).to_parquet(f'{out}/{cur}_delivery.parquet');print('delivery',len(allp))
