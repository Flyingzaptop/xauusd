import subprocess,time,tarfile,json,hashlib
from pathlib import Path
CLI='/workspace/colab-tools/bin/colab';ROOT=Path('/workspace/xauusd/reports/spectral_transformer_20261013');ARCH=Path('/workspace/scratch/spectral-snapshot.tar.gz');expected=hashlib.sha256(Path('/workspace/xauusd/experiments/spectral_transformer/train.py').read_bytes()).hexdigest();last=None
while True:
 r=subprocess.run([CLI,'download','-s','xauusd-spectral-transformer','/content/spectral-snapshot.tar.gz',str(ARCH)],capture_output=True,text=True,timeout=120)
 if r.returncode==0:
  try:
   with tarfile.open(ARCH) as t:
    if 'remote/run.json' in t.getnames():assert json.load(t.extractfile('remote/run.json'))['source_sha256']['train.py']==expected
    t.extractall(ROOT,filter='data')
   p=ROOT/'remote/status.json';s=json.loads(p.read_text()) if p.exists() else {'status':'waiting_for_training'}
   if s!=last:print(json.dumps(s),flush=True);last=s
   w=ROOT/'remote/worker_result.json'
   if w.exists():
    result=json.loads(w.read_text());print('WORKER',result,flush=True)
    if result['returncode']==0 and (ROOT/'remote/complete.json').exists():
     stop=subprocess.run([CLI,'stop','-s','xauusd-spectral-transformer'],capture_output=True,text=True,timeout=120);print(stop.stdout,stop.stderr,flush=True)
    break
  except (tarfile.TarError,EOFError,OSError) as e:print('snapshot_retry',str(e),flush=True)
 else:print('snapshot_not_ready',flush=True)
 time.sleep(45)
