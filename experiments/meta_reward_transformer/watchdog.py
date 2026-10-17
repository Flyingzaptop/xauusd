"""Local guard: periodic snapshots, final download, verify sources, stop own GPU."""
from pathlib import Path
import subprocess,time,json,tarfile,hashlib,traceback,re,os
B=Path('/workspace/xauusd/experiments/meta_reward_transformer');P=Path('/workspace/xauusd/reports/meta_reward_transformer_20261015');CLI='/workspace/colab-tools/bin/colab';P.mkdir(exist_ok=True)
original_start=json.loads((P/'watchdog_process.json').read_text())['started_epoch']
started=time.monotonic()-(time.time()-original_start);last_backup=time.monotonic()-1300;failures=0
log=open(P/'watchdog.log','a',buffering=1)
def run(args,timeout=180):
 q=subprocess.run([CLI]+args,capture_output=True,text=True,timeout=timeout)
 if q.returncode:raise RuntimeError(q.stdout[-1500:]+q.stderr[-1500:])
 return q.stdout
try:
 while True:
  try:
   d={}
   for name in ['prepare_status.json','training_status.json']:
    dest=P/('monitor_'+name);run(['download','-s','meta-reward','/content/meta_reward/'+name,str(dest)],timeout=45);d[name]=json.loads(dest.read_text())
   if d.get('training_status.json',{}).get('phase') in ['complete','failed']:
    try:
     dest=P/'monitor_archive_ready.json';run(['download','-s','meta-reward','/content/meta_reward/archive_ready.json',str(dest)],timeout=45);d['archive_ready.json']=json.loads(dest.read_text())
    except Exception:pass
   (P/'live_status.json').write_text(json.dumps(d,indent=2));log.write(json.dumps(d)+'\n')
   final=d.get('archive_ready.json');phase=d.get('training_status.json',{}).get('phase')
   if final or (phase and time.monotonic()-last_backup>1200):
    try:
     target=P/'latest-snapshot.tar.gz';run(['download','-s','meta-reward','/content/meta-reward-results.tar.gz',str(target)],timeout=300);last_backup=time.monotonic()
     with tarfile.open(target) as t:
      names=t.getnames();assert 'features.f16' not in names;t.extractall(P/'remote',filter='data')
     sources=P/'remote/results/source';h=P/'remote/results/source_hashes.json'
     if h.exists():
      for f,expected in json.loads(h.read_text()).items():assert hashlib.sha256((sources/f).read_bytes()).hexdigest()==expected,f
     (P/'snapshot_verification.json').write_text(json.dumps({'verified':True,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size,'files':len(names),'final':bool(final)},indent=2))
    except Exception:
     log.write('SNAPSHOT ERROR '+traceback.format_exc()+'\n')
     if final:raise
   if final:
    run(['stop','-s','meta-reward']);usage=run(['usage']);(P/'final_usage.txt').write_text(usage);(P/'finished.json').write_text(json.dumps(final,indent=2))
    subprocess.run(['/workspace/colab-tools/bin/python',str(B/'report.py')],stdout=log,stderr=subprocess.STDOUT,timeout=120);break
   failures=0
  except Exception:
   failures+=1;log.write(traceback.format_exc()+'\n')
   if failures>=12:raise RuntimeError('Twelve consecutive runtime or archive failures')
  if time.monotonic()-started>7*3600:raise RuntimeError('Seven-hour local guard exceeded (plus prior preparation); budget stop')
  time.sleep(45)
except Exception:
 (P/'watchdog_error.txt').write_text(traceback.format_exc());log.write(traceback.format_exc())
 # Budget stop even when retrieval fails. Retain any downloaded partial snapshots.
 try:run(['stop','-s','meta-reward'])
 except Exception:log.write(traceback.format_exc())
