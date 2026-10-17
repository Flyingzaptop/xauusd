from pathlib import Path
import json,subprocess
r=Path('/content/meta_reward')
for n in ['prepare_status.json','training_status.json','benchmark_patch10.json','result.json','prepare_error.txt']:
 p=r/n
 if p.exists():print(n,p.read_text()[-2500:])
for n in ['bench_patch10.log','training.log','supervisor.log']:
 p=r/n
 if p.exists():print(n,p.read_text()[-1200:])
print(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True))
