from pathlib import Path
import json,hashlib
r=Path('/content/meta_reward');d={}
for n in ['prepare_status.json','training_status.json','archive_ready.json','training_result.json']:
 p=r/n
 if p.exists():
  try:d[n]=json.loads(p.read_text())
  except Exception:pass
p=r/'train.py'
if p.exists():d['train_source_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
print('META_STATUS '+json.dumps(d))
