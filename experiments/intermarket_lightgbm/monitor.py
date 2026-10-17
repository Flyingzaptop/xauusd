from pathlib import Path
import psutil,json
r=Path('/content/intermarket');p=r/'status.json';print(p.read_text() if p.exists() else 'no status');print('RAM GB',round(psutil.virtual_memory().used/1e9,2),'available',round(psutil.virtual_memory().available/1e9,2));print('raw files',len(list((r/'raw').glob('*.parquet'))));print((r/'prepare.log').read_text(errors='replace')[-900:])
