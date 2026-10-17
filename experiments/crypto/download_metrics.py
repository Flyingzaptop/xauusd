import sys,os,hashlib,subprocess,zipfile,datetime
from concurrent.futures import ThreadPoolExecutor
out,syms=sys.argv[1],sys.argv[2:]
B='https://data.binance.vision/data/futures/um/daily/metrics'
d0=datetime.date(2020,9,1);d1=datetime.date(2026,9,30);days=[d0+datetime.timedelta(i) for i in range((d1-d0).days+1)]
jobs=[(s,d) for s in syms for d in days]
def get(j):
    s,d=j;dd=f'{out}/{s}/metrics';os.makedirs(dd,exist_ok=True);fn=f'{dd}/{s}-metrics-{d}.zip'
    if os.path.exists(fn.replace('.zip','.csv')):return 'cached'
    url=f'{B}/{s}/{s}-metrics-{d}.zip';r=subprocess.run(['curl','-sS','-f','--retry','3','-o',fn,url],capture_output=True)
    if r.returncode!=0:return 'missing'
    c=subprocess.run(['curl','-sS','-f','--retry','3',url+'.CHECKSUM'],capture_output=True,text=True).stdout.split()
    if c and c[0]!=hashlib.sha256(open(fn,'rb').read()).hexdigest():os.remove(fn);return 'BADSUM'
    zipfile.ZipFile(fn).extractall(dd);os.remove(fn);return 'ok'
with ThreadPoolExecutor(12) as ex:res=list(ex.map(get,jobs))
from collections import Counter;print(Counter(res))
