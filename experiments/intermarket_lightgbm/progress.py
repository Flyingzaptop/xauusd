from pathlib import Path
import json,psutil
r=Path('/content/intermarket')
for name in ['status.json','train_status.json','training_result.json']:
 p=r/name
 if p.exists():print(name,p.read_text()[-1800:])
p=r/'prepared.json'
if p.exists():
 d=json.loads(p.read_text());print('prepared',{k:d[k] for k in ['complete','rows','features','seconds']})
print('available_RAM_GB',round(psutil.virtual_memory().available/1e9,1))
p=r/'training.log'
if p.exists():print('training log tail',p.read_text(errors='replace')[-900:])
