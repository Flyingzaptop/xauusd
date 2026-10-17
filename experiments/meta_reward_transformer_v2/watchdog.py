"""Status failures never trigger release; hard budget is independent of monitoring."""
from pathlib import Path
import subprocess,time,json,tarfile,hashlib,traceback,os
B=Path('/workspace/xauusd/experiments/meta_reward_transformer_v2');P=Path('/workspace/xauusd/reports/meta_reward_transformer_v2_20261015');CLI='/workspace/colab-tools/bin/colab';SESSION='meta-reward-v2';started=json.loads((P/'watchdog_process.json').read_text())['started_epoch'];last_backup=0;log=open(P/'watchdog.log','a',buffering=1)
def run(args,timeout=90):
 q=subprocess.run([CLI]+args,capture_output=True,text=True,timeout=timeout)
 if q.returncode:raise RuntimeError(q.stdout[-1000:]+q.stderr[-1000:])
 return q.stdout

def fetch(name):
 dest=P/('monitor_'+name);tmp=dest.with_suffix('.download');run(['download','-s',SESSION,'/content/meta_reward/'+name,str(tmp)],timeout=45);v=json.loads(tmp.read_text());os.replace(tmp,dest);return v

def backup(final=False):
 tmp=P/'snapshot.download';run(['download','-s',SESSION,'/content/meta-reward-v2-results.tar.gz',str(tmp)],timeout=300)
 # Verify whole stream before replacing the previous valid archive.
 with tarfile.open(tmp) as t:
  names=t.getnames()
  for member in t:
   if member.isfile():
    with t.extractfile(member) as f:
     while f.read(1024*1024):pass
 os.replace(tmp,P/'latest-snapshot.tar.gz')
 with tarfile.open(P/'latest-snapshot.tar.gz') as t:t.extractall(P/'remote',filter='data')
 h=P/'remote/results/source_hashes.json'
 if h.exists():
  for name,expected in json.loads(h.read_text()).items():assert hashlib.sha256((P/'remote/results/source'/name).read_bytes()).hexdigest()==expected,name
 (P/'snapshot_verification.json').write_text(json.dumps({'verified':True,'epoch':time.time(),'bytes':(P/'latest-snapshot.tar.gz').stat().st_size,'files':len(names),'final':final}))

while True:
 final=None;status={'checked_epoch':time.time()}
 for name in ['supervisor_status.json','prepare_status.json','training_status.json','archive_ready.json']:
  try:
   v=fetch(name);status[name]=v
   if name=='archive_ready.json':final=v
  except Exception as e:status[name]={'monitor_error':str(e)[-600:]}
 (P/'live_status.json').write_text(json.dumps(status,indent=2));log.write(json.dumps(status)+'\n')
 if final is not None or time.time()-last_backup>300:
  try:backup(final is not None);last_backup=time.time()
  except Exception:log.write('BACKUP ERROR '+traceback.format_exc()+'\n');final=None
 if final is not None:
  try:
   run(['stop','-s',SESSION]);(P/'finished.json').write_text(json.dumps(final));(P/'final_usage.txt').write_text(run(['usage']));subprocess.run(['/workspace/colab-tools/bin/python',str(B/'report.py')],stdout=log,stderr=subprocess.STDOUT,timeout=180);break
  except Exception:log.write('RELEASE ERROR '+traceback.format_exc()+'\n')
 if time.time()-started>7*3600:
  try:backup()
  except Exception:log.write('BUDGET BACKUP ERROR '+traceback.format_exc())
  try:run(['stop','-s',SESSION])
  finally:(P/'budget_stop.json').write_text(json.dumps({'epoch':time.time(),'reason':'7-hour wall budget'}))
  break
 time.sleep(45)
