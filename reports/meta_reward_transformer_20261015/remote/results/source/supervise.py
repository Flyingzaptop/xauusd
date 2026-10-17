from pathlib import Path
import subprocess,time,json,tarfile,os,traceback
r=Path('/content/meta_reward');started=time.monotonic()
def archive():
 with tarfile.open('/content/meta-reward-results.tmp.tar.gz','w:gz',compresslevel=1) as t:
  for f in r.glob('*'):
   if f.is_file() and f.suffix in ['.py','.json','.log','.md','.txt']:t.add(f,arcname=f.name)
  for f in (r/'results').rglob('*'):
   if f.is_file() and f.suffix!='.tmp':t.add(f,arcname=str(f.relative_to(r)))
 os.replace('/content/meta-reward-results.tmp.tar.gz','/content/meta-reward-results.tar.gz')
try:
 while not (r/'prepared.json').exists():
  if (r/'prepare_error.txt').exists():raise RuntimeError('Preparation failed')
  time.sleep(5)
 p=subprocess.Popen(['python','-u',str(r/'train.py')],stdout=open(r/'training.log','w'),stderr=subprocess.STDOUT)
 last=time.monotonic()
 while p.poll() is None:
  if time.monotonic()-started>28800:p.terminate();p.wait(timeout=30);raise RuntimeError('Supervisor exceeded eight-hour wall limit')
  if time.monotonic()-last>900:archive();last=time.monotonic()
  time.sleep(5)
 result={'returncode':p.returncode}
except Exception:
 result={'returncode':-1,'error':traceback.format_exc()}
(r/'training_result.json').write_text(json.dumps(result))
archive();(r/'archive_ready.json').write_text(json.dumps({'bytes':Path('/content/meta-reward-results.tar.gz').stat().st_size,**result}))
