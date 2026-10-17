from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests,pyarrow.parquet as pq,json,time
P=Path('/workspace/xauusd/reports/intermarket_lightgbm_20261015');(P/'samples').mkdir(exist_ok=True)
files=json.loads((P/'drive_manifest.json').read_text());chosen=[f for f in files if f['name'].startswith(('XAUUSD_','EURUSD_','APPLE_'))]
def one(f):
 p=P/'samples'/f['name']
 if not p.exists():
  r=requests.get('https://drive.usercontent.google.com/download',params={'id':f['id'],'export':'download','confirm':'t'},stream=True,timeout=90);r.raise_for_status()
  with p.open('wb') as w:
   for b in r.iter_content(1024*1024):w.write(b)
 assert p.stat().st_size==f['size']
 q=pq.ParquetFile(p);table=q.read_row_group(0).slice(0,3)
 return {'file':f['name'],'rows':q.metadata.num_rows,'row_groups':q.metadata.num_row_groups,'schema':str(q.schema_arrow),'sample':table.to_pylist(),'metadata':{str(k):str(v)[:1500] for k,v in (q.schema_arrow.metadata or {}).items()}}
with ThreadPoolExecutor(max_workers=3) as e:out=list(e.map(one,chosen))
(P/'sample_inspection.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps(out,indent=2,default=str))
