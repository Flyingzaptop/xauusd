from pathlib import Path
import requests,json,time,hashlib,os,traceback,concurrent.futures as cf
import numpy as np,pandas as pd,pyarrow.parquet as pq,psutil
from features import encode,NAMES
R=Path('/content/meta_reward');RAW=R/'raw';RAW.mkdir(exist_ok=True);T=time.monotonic()
def atomic(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));os.replace(tmp,p)
def status(**kw):
 kw.update(seconds=round(time.monotonic()-T,1),available_RAM_GB=round(psutil.virtual_memory().available/1e9,1));atomic(R/'prepare_status.json',kw);print(json.dumps(kw),flush=True)
def download(f):
 p=RAW/f['name']
 if p.exists() and p.stat().st_size==f['size']:return p
 for retry in range(4):
  try:
   q=requests.get('https://drive.usercontent.google.com/download',params={'id':f['id'],'export':'download','confirm':'t'},stream=True,timeout=(20,180));q.raise_for_status()
   with p.with_suffix('.part').open('wb') as w:
    for b in q.iter_content(1024*1024):w.write(b)
   assert p.with_suffix('.part').stat().st_size==f['size'];os.replace(p.with_suffix('.part'),p);return p
  except Exception:
   if retry==3:raise
   time.sleep(2*(retry+1))
try:
 files=sorted(json.loads((R/'drive_manifest.json').read_text()),key=lambda x:x['name']);assert len(files)==222
 gold=next(x for x in files if x['name'].startswith('XAUUSD_'));download(gold)
 g=pq.read_table(RAW/gold['name']).to_pandas().sort_values('timestamp').reset_index(drop=True)
 anchors=pd.DatetimeIndex(g.timestamp).as_unit('s').asi8;start=int(anchors[0]);n=(int(anchors[-1])-start)//60+1
 np.save(R/'anchors.npy',anchors);np.save(R/'gold_ohlcv.npy',g[['open','high','low','close','volume']].to_numpy());del g
 shape=(len(files),len(anchors),len(NAMES));expected=int(np.prod(shape))*2
 resume=(R/'features.f16').exists() and (R/'features.f16').stat().st_size==expected and (R/'audit.json').exists()
 mm=np.memmap(R/'features.f16',mode='r+' if resume else 'w+',dtype='float16',shape=shape)
 audits=json.loads((R/'audit.json').read_text()) if resume else [];completed={a['file'] for a in audits}
 status(phase='download',files=222,rows=len(anchors),feature_count=48,bytes=expected)
 with cf.ThreadPoolExecutor(max_workers=8) as ex:
  for k,fut in enumerate(cf.as_completed([ex.submit(download,f) for f in files])):
   fut.result()
   if k%10==0:status(phase='download',done=k+1,total=222)
 def one(pair):
  i,f=pair;block,audit=encode(pq.ParquetFile(RAW/f['name']).read(use_threads=False),start,n,anchors)
  assert block.shape==(48,len(anchors));assert not np.isinf(block).any()
  fp16=block.astype('float16');finite=np.isfinite(block);err=np.abs(fp16.astype('float32')-block)
  audit.update(file=f['name'],sha256=hashlib.sha256((RAW/f['name']).read_bytes()).hexdigest(),fp16_max_abs_error=float(np.max(err[finite])) if finite.any() else 0)
  mm[i]=fp16.T;mm.flush();del block,fp16,err,finite
  return audit
 todo=iter([(i,f) for i,f in enumerate(files) if f['name'] not in completed]);workers=min(8,max(1,int(psutil.virtual_memory().total/1e9)//14))
 with cf.ThreadPoolExecutor(max_workers=workers) as ex:
  pending={}
  for _ in range(workers):
   pair=next(todo,None)
   if pair is not None:pending[ex.submit(one,pair)]=pair
  while pending:
   fut=next(cf.as_completed(pending));pending.pop(fut);audit=fut.result();audits.append(audit);atomic(R/'audit.json',audits);status(phase='features',done=len(audits),total=222,symbol=audit['file'])
   pair=next(todo,None)
   if pair is not None:pending[ex.submit(one,pair)]=pair
 d=pd.to_datetime(anchors,unit='s',utc=True);tod=(d.hour*60+d.minute).to_numpy()/1440;clock=[]
 for k in [1,2,3]:clock.extend([np.sin(2*np.pi*k*tod),np.cos(2*np.pi*k*tod)])
 dow=d.dayofweek.to_numpy()/7;clock.extend([np.sin(2*np.pi*dow),np.cos(2*np.pi*dow)])
 np.save(R/'clock.npy',np.stack(clock,axis=1).astype('float16'))
 manifest={'complete':True,'rows':len(anchors),'instruments':222,'features_per_instrument':48,'shape':shape,'dtype':'float16','bytes':expected,'symbols':[f['name'].rsplit('_',1)[0] for f in files],'names':NAMES,'gold_index':next(i for i,f in enumerate(files) if f['name'].startswith('XAUUSD_')),'first':str(d[0]),'last':str(d[-1]),'seconds':time.monotonic()-T,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.glob('*.py')}}
 atomic(R/'prepared.json',manifest);status(phase='complete',rows=len(anchors),bytes=expected)
except Exception:
 (R/'prepare_error.txt').write_text(traceback.format_exc());status(phase='failed',error=traceback.format_exc());raise
