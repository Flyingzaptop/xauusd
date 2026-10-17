from pathlib import Path
import requests,json,time,hashlib,os,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pyarrow.parquet as pq,pandas as pd
from features import encode,NAMES
R=Path('/content/intermarket');RAW=R/'raw';RAW.mkdir(exist_ok=True)
T=time.monotonic()
try:PRIOR_SECONDS=json.loads((R/'status.json').read_text()).get('seconds',0)
except Exception:PRIOR_SECONDS=0
def status(**kw):
 kw['seconds']=round(PRIOR_SECONDS+time.monotonic()-T,1);(R/'status.json').write_text(json.dumps(kw));print(json.dumps(kw),flush=True)
def download(f):
 p=RAW/f['name']
 if p.exists() and p.stat().st_size==f['size']:return p
 err=''
 for attempt in range(4):
  try:
   q=requests.get('https://drive.usercontent.google.com/download',params={'id':f['id'],'export':'download','confirm':'t'},stream=True,timeout=(20,180));q.raise_for_status()
   with p.with_suffix('.part').open('wb') as w:
    for b in q.iter_content(1024*1024):w.write(b)
   if p.with_suffix('.part').stat().st_size!=f['size']:raise ValueError('Download size mismatch')
   os.replace(p.with_suffix('.part'),p);return p
  except Exception as e:err=str(e);time.sleep(2*(attempt+1))
 raise RuntimeError(f['name']+': '+err)
try:
 files=sorted(json.loads((R/'drive_manifest.json').read_text()),key=lambda x:x['name']);gold=next(f for f in files if f['name'].startswith('XAUUSD_'))
 download(gold);g=pq.read_table(RAW/gold['name']).to_pandas();g=g.sort_values('timestamp').reset_index(drop=True)
 gt=pd.DatetimeIndex(g.timestamp).as_unit('s').asi8;start=int(gt[0]);end=int(gt[-1]);n=(end-start)//60+1
 # Every fifth UTC minute, preserving all observed M1 candles in feature computation.
 anchors=gt[(gt%300==0)&(gt>=start+1440*60)]
 np.save(R/'anchors.npy',anchors);np.savez_compressed(R/'gold.npz',timestamp=gt,ohlcv=g[['open','high','low','close','volume']].to_numpy())
 nfeat=len(files)*len(NAMES)+8
 expected=nfeat*len(anchors)*4
 resume=(R/'features.f32').exists() and (R/'features.f32').stat().st_size==expected and (R/'audit.json').exists()
 mm=np.memmap(R/'features.f32',dtype='float32',mode='r+' if resume else 'w+',shape=(nfeat,len(anchors)))
 audits=json.loads((R/'audit.json').read_text()) if resume else []
 names=[f['name'].rsplit('_',1)[0]+'__'+x for f in files for x in NAMES]
 done=0;status(phase='downloading',files=len(files),bytes=sum(f['size'] for f in files),anchors=len(anchors),features=nfeat)
 with ThreadPoolExecutor(max_workers=8) as ex:
  futures=[ex.submit(download,f) for f in files]
  for fut in as_completed(futures):fut.result();done+=1;status(phase='downloading',done=done,total=len(files))
 def feature_one(pair):
  i,f=pair;tab=pq.ParquetFile(RAW/f['name']).read(use_threads=False);block,audit=encode(tab,start,n,anchors)
  audit['file']=f['name'];audit['sha256']=hashlib.sha256((RAW/f['name']).read_bytes()).hexdigest()
  return i,block,audit
 done_names={x['file'] for x in audits};todo=iter([(i,f) for i,f in enumerate(files) if f['name'] not in done_names])
 with ThreadPoolExecutor(max_workers=3) as ex:
  pending={}
  for _ in range(3):
   pair=next(todo,None)
   if pair is not None:pending[ex.submit(feature_one,pair)]=pair
  while pending:
   fut=next(as_completed(pending));pending.pop(fut);i,block,audit=fut.result();mm[i*39:(i+1)*39]=block;audits.append(audit)
   (R/'audit.json').write_text(json.dumps(audits,indent=2));status(phase='features',done=len(audits),total=len(files),symbol=files[i]['name'])
   del block
   pair=next(todo,None)
   if pair is not None:pending[ex.submit(feature_one,pair)]=pair
 d=pd.to_datetime(anchors,unit='s',utc=True);tod=(d.hour*60+d.minute).to_numpy()/1440
 clock=[]
 for k in [1,2,3]:clock.extend([np.sin(2*np.pi*k*tod),np.cos(2*np.pi*k*tod)])
 dow=d.dayofweek.to_numpy()/7;clock.extend([np.sin(2*np.pi*dow),np.cos(2*np.pi*dow)])
 mm[-8:]=np.stack(clock);names.extend(['clock_'+str(i) for i in range(8)]);mm.flush()
 (R/'prepared.json').write_text(json.dumps({'complete':True,'files':len(files),'features':nfeat,'rows':len(anchors),'shape':[nfeat,len(anchors)],'feature_names':names,'grid_start':start,'grid_n':n,'first_anchor':str(d[0]),'last_anchor':str(d[-1]),'seconds':PRIOR_SECONDS+time.monotonic()-T},indent=2));status(phase='complete',rows=len(anchors),features=nfeat)
except Exception:
 (R/'error.txt').write_text(traceback.format_exc());status(phase='failed',error=traceback.format_exc());raise
