"""Download Binance USD-M futures public data (data.binance.vision) with CHECKSUM verification.
usage: download.py OUTDIR SYMBOL... ; datasets: monthly 1m klines, monthly fundingRate (2020-01 .. 2026-09)."""
import sys,os,hashlib,subprocess,zipfile
from concurrent.futures import ThreadPoolExecutor
out=sys.argv[1];syms=sys.argv[2:];os.makedirs(out,exist_ok=True)
B='https://data.binance.vision/data/futures/um/monthly'
months=[f'{y}-{m:02d}' for y in range(2020,2027) for m in range(1,13) if (y,m)<=(2026,9)]
jobs=[]
for s in syms:
    for mo in months:
        jobs.append((f'{B}/klines/{s}/1m/{s}-1m-{mo}.zip',f'{out}/{s}/klines'))
        jobs.append((f'{B}/fundingRate/{s}/{s}-fundingRate-{mo}.zip',f'{out}/{s}/funding'))
def get(j):
    url,d=j;os.makedirs(d,exist_ok=True);fn=f'{d}/{url.split("/")[-1]}'
    if os.path.exists(fn.replace('.zip','.csv')):return 'cached'
    r=subprocess.run(['curl','-sS','-f','--retry','3','-o',fn,url],capture_output=True)
    if r.returncode!=0:return 'missing'
    c=subprocess.run(['curl','-sS','-f','--retry','3',url+'.CHECKSUM'],capture_output=True,text=True).stdout.split()
    h=hashlib.sha256(open(fn,'rb').read()).hexdigest()
    if c and c[0]!=h:os.remove(fn);return 'BADSUM'
    zipfile.ZipFile(fn).extractall(d);os.remove(fn);return 'ok'
with ThreadPoolExecutor(8) as ex:res=list(ex.map(get,jobs))
from collections import Counter;print(Counter(res))
