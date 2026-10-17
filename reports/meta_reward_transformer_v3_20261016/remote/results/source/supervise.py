from pathlib import Path
import subprocess,time,json,tarfile,os,traceback
r=Path('/content/meta_reward');started=time.monotonic();(r/'results').mkdir(exist_ok=True)
def atomic(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v));os.replace(tmp,p)
def archive():
 with tarfile.open('/content/meta-reward-v3-results.tmp.tar.gz','w:gz',compresslevel=1) as t:
  for f in r.glob('*'):
   if f.is_file() and f.suffix in ['.py','.json','.log','.md','.txt']:t.add(f,arcname=f.name)
  for f in (r/'results').rglob('*'):
   if f.is_file() and f.suffix!='.tmp':
    # Open inode remains valid if training atomically replaces this checkpoint.
    with f.open('rb') as handle:
     info=t.gettarinfo(fileobj=handle,arcname=str(f.relative_to(r)));t.addfile(info,handle)
 os.replace('/content/meta-reward-v3-results.tmp.tar.gz','/content/meta-reward-v3-results.tar.gz')
 atomic(r/'archive_info.json',{'created_epoch':time.time(),'bytes':Path('/content/meta-reward-v3-results.tar.gz').stat().st_size})
p=None;phase='prepare'
try:
 p=subprocess.Popen(['python','-u',str(r/'prepare.py')],stdout=open(r/'prepare.log','w'),stderr=subprocess.STDOUT);last=0
 while True:
  atomic(r/'supervisor_status.json',{'phase':phase,'elapsed_seconds':time.monotonic()-started,'worker_pid':p.pid,'updated_epoch':time.time()})
  if time.monotonic()-started>6.75*3600:p.terminate();p.wait(timeout=30);raise RuntimeError('6.75-hour remote budget limit')
  if time.monotonic()-last>300:
   archive();last=time.monotonic()
  code=p.poll()
  if code is not None:
   if phase=='prepare':
    if code!=0 or not (r/'prepared.json').exists():raise RuntimeError('Preparation failed; inspect prepare.log')
    phase='training';p=subprocess.Popen(['python','-u',str(r/'train.py')],stdout=open(r/'training.log','w'),stderr=subprocess.STDOUT)
   else:
    result={'returncode':code,'complete':code==0};break
  if int(time.monotonic()-started)//30 != globals().get('_last_print',-1):
   _last_print=int(time.monotonic()-started)//30
   path=r/('prepare_status.json' if phase=='prepare' else 'training_status.json')
   print(path.read_text() if path.exists() else phase,flush=True)
  time.sleep(10)
except Exception:
 if p is not None and p.poll() is None:p.terminate()
 result={'returncode':-1,'complete':False,'error':traceback.format_exc()}
atomic(r/'training_result.json',result);archive();atomic(r/'archive_ready.json',{'created_epoch':time.time(),**result});atomic(r/'supervisor_status.json',{'phase':'complete' if result['complete'] else 'failed','updated_epoch':time.time(),**result})
