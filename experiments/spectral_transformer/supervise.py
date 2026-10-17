import time,subprocess,os,json,tarfile
from pathlib import Path
p=Path('/content/spectral');r=p/'results';r.mkdir(exist_ok=True);start=time.time()
def archive():
 with tarfile.open('/content/spectral-snapshot.tmp.tar.gz','w:gz',compresslevel=1) as t:
  for f in r.iterdir():
   if f.is_file() and f.suffix!='.tmp':t.add(f,arcname='remote/'+f.name)
  for n in ['prepared.json','prepare.log','training.log']:
   if (p/n).exists():t.add(p/n,arcname='remote/'+n)
 os.replace('/content/spectral-snapshot.tmp.tar.gz','/content/spectral-snapshot.tar.gz')
while not (p/'prepared.json').exists():
 if (p/'prepare.log').exists() and 'Traceback (most recent call last)' in (p/'prepare.log').read_text():
  (r/'failure.json').write_text(json.dumps({'stage':'prepare','log':(p/'prepare.log').read_text()[-4000:]}));archive();raise SystemExit(1)
 if time.time()-start>1200:raise TimeoutError('Preparation timeout')
 time.sleep(5)
with (p/'training.log').open('w') as log:
 child=subprocess.Popen(['python','-u',str(p/'train.py')],stdout=log,stderr=subprocess.STDOUT,cwd=p)
 last=time.time()
 while child.poll() is None:
  if time.time()-start>4200:child.terminate();child.wait(timeout=30);break
  if time.time()-last>120:archive();last=time.time()
  time.sleep(5)
(r/'worker_result.json').write_text(json.dumps({'returncode':child.returncode,'seconds':time.time()-start}));archive()
