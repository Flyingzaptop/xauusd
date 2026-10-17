from pathlib import Path
import importlib.metadata,json,sys,psutil,os,shutil
r=Path('/content/intermarket')
d={'python':sys.version,'versions':{k:importlib.metadata.version(k) for k in ['numpy','pandas','pyarrow','lightgbm','requests','psutil']},'cpus':os.cpu_count(),'ram_bytes':psutil.virtual_memory().total,'disk_free_bytes':shutil.disk_usage('/content').free,'gpu_used':False,'session':'intermarket-prep','compute_rate_cpu_units_per_hour_observed':.26}
(r/'environment.json').write_text(json.dumps(d,indent=2));print(json.dumps(d))
