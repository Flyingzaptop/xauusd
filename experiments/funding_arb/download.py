"""Hyperliquid hourly funding + Binance funding for the fixed coin list. usage: download.py OUTDIR"""
import sys,os,io,json,time,zipfile,subprocess,urllib.request,pandas as pd
O=sys.argv[1];os.makedirs(O,exist_ok=True)
C='BTC ETH SOL XRP DOGE BNB ADA AVAX LINK LTC DOT TRX BCH NEAR APT ARB OP SUI ATOM AAVE INJ TIA SEI WLD ENA'.split()
def hl(coin):
    fn=f'{O}/hl_{coin}.parquet'
    if os.path.exists(fn):return
    rows=[];t=int(pd.Timestamp('2023-05-01').timestamp()*1000);end=int(pd.Timestamp('2026-10-01').timestamp()*1000)
    while t<end:
        for k in range(6):
            try:
                req=urllib.request.Request('https://api.hyperliquid.xyz/info',data=json.dumps({'type':'fundingHistory','coin':coin,'startTime':t}).encode(),headers={'Content-Type':'application/json'})
                r=json.load(urllib.request.urlopen(req,timeout=60));break
            except Exception:time.sleep(2**k)
        if not r:break
        rows+=r;nt=r[-1]['time']+1
        if nt<=t:break
        t=nt;time.sleep(0.4)
    pd.DataFrame(rows).to_parquet(fn);print('hl',coin,len(rows),flush=True)
def bn(coin):
    fn=f'{O}/bn_{coin}.parquet'
    if os.path.exists(fn):return
    s=coin+'USDT';parts=[]
    for y in range(2023,2027):
        for m in range(1,13):
            mo=f'{y}-{m:02d}'
            if not('2023-05'<=mo<='2026-09'):continue
            r=subprocess.run(['curl','-sS','-f','--retry','4',f'https://data.binance.vision/data/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{mo}.zip'],capture_output=True)
            if r.returncode:continue
            z=zipfile.ZipFile(io.BytesIO(r.stdout));parts.append(pd.read_csv(z.open(z.namelist()[0])))
    pd.concat(parts).to_parquet(fn);print('bn',coin,flush=True)
for c in C:bn(c);hl(c)
