"""Download RAW monthly aggTrades zips (kept zipped, not aggregated) for BTCUSDT/ETHUSDT, newest month first, CHECKSUM-verified.
Stops when free disk < MIN_FREE_GB. usage: download_aggtrades.py OUTDIR FIRST_MONTH(YYYY-MM) LAST_MONTH"""
import sys,os,hashlib,subprocess,shutil,time
out,first,last=sys.argv[1:4];MIN_FREE_GB=3.0
y,m=map(int,last.split('-'));fy,fm=map(int,first.split('-'));months=[]
while (y,m)>=(fy,fm):
    months.append(f'{y}-{m:02d}');m-=1
    if m==0:y-=1;m=12
B='https://data.binance.vision/data/futures/um/monthly/aggTrades'
for mo in months:
    for s in ('BTCUSDT','ETHUSDT'):
        fn=f'{out}/{s}-aggTrades-{mo}.zip'
        if os.path.exists(fn):continue
        if shutil.disk_usage(out).free/1e9<MIN_FREE_GB:print('disk limit reached, stop at',mo,flush=True);sys.exit(0)
        url=f'{B}/{s}/{s}-aggTrades-{mo}.zip';t=time.time()
        r=subprocess.run(['curl','-sS','-f','--retry','5','-o',fn+'.part',url],capture_output=True)
        if r.returncode!=0:print('missing',s,mo,flush=True);continue
        c=subprocess.run(['curl','-sS','-f','--retry','3',url+'.CHECKSUM'],capture_output=True,text=True).stdout.split()
        h=hashlib.sha256(open(fn+'.part','rb').read()).hexdigest()
        if c and c[0]!=h:os.remove(fn+'.part');print('BADSUM',s,mo,flush=True);continue
        os.rename(fn+'.part',fn);print(f'ok {s} {mo} {os.path.getsize(fn)/1e6:.0f} MB {time.time()-t:.0f}s free {shutil.disk_usage(out).free/1e9:.1f} GB',flush=True)
print('done')
