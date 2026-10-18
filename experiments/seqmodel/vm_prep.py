"""On the Colab VM: build any missing monthly/final npz directly from data.binance.vision (parallel), then launch training.
Used because chunked upload through the CLI is slower than the VM rebuilding the files (GPU idles during upload)."""
import os,subprocess,sys,numpy as np
from multiprocessing import Pool
DD=os.environ.get('DATA_DIR','/content/data');RAW='/content/raw';OUT=os.environ.get('OUT_DIR','/content/outB');os.makedirs(DD,exist_ok=True);os.makedirs(RAW,exist_ok=True);B='https://data.binance.vision/data/futures/um'
MONTHS=['2025-07','2025-08','2025-09','2025-10','2025-11','2025-12','2026-01','2026-02','2026-03','2026-04','2026-05','2026-06','2026-07','2026-08','2026-09']
def ok(f):
    try:np.load(f)['s_cnt'][:1];return True
    except Exception:return False
def job(a):
    s,mo=a;out=f'{DD}/{s}-{mo}.npz'
    if ok(out):return f'have {out}'
    if mo=='final':urls=[f'{B}/daily/aggTrades/{s}/{s}-aggTrades-2026-10-{d:02d}.zip' for d in range(1,17)]
    else:urls=[f'{B}/monthly/aggTrades/{s}/{s}-aggTrades-{mo}.zip']
    fs=[]
    for u in urls:
        f=f'{RAW}/{os.getpid()}_{os.path.basename(u)}';subprocess.run(['curl','-sS','-f','--retry','5','-o',f,u],check=True);fs.append(f)
    r=subprocess.run([sys.executable,'-W','ignore','/content/seq/prep.py',s,*fs,out],capture_output=True,text=True)
    [os.remove(f) for f in fs];return r.stdout.strip()[-120:]+r.stderr[-300:]
if __name__=='__main__':
    jobs=[(s,m) for s in ('BTCUSDT','ETHUSDT') for m in MONTHS+['final']]
    with Pool(6) as p:
        for r in p.imap_unordered(job,jobs):print(r,flush=True)
    bad=[j for j in jobs if not ok(f'{DD}/{j[0]}-{j[1]}.npz')];print('missing',bad,flush=True)
    if not bad:
        f=lambda ms:' '.join(f'{DD}/{s}-{m}.npz' for s in ('BTCUSDT','ETHUSDT') for m in ms)
        cmd=(f"cd /content/seq && python -W ignore train.py --train {f(MONTHS[:9])} --val {f(MONTHS[9:12])} --test {f(MONTHS[12:])} --final {f(['final'])} "
             f"--d 384 --depth 12 --heads 8 --batch 8 --pre_steps 15000 --ft_steps 15000 --eval_every 2500 --val_stride 600 --eval_stride 120 --eval_batch 64 --out {OUT} 2>&1 | tee -a {OUT}/train.log")
        os.makedirs(OUT,exist_ok=True);print('TRAIN START',flush=True);subprocess.run(cmd,shell=True);print('TRAIN END',flush=True)
