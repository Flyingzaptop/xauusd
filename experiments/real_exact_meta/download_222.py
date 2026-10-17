"""Download the 222 parquet files listed in drive_manifest.json (public Drive links) with size check."""
import json,sys,os,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
man=json.load(open(Path(__file__).resolve().parents[1]/'meta_reward_transformer_v3/drive_manifest.json'))
out=Path(sys.argv[1] if len(sys.argv)>1 else '/home/user/data_raw/all222');out.mkdir(parents=True,exist_ok=True)
def get(f):
    p=out/f['name']
    if p.exists() and p.stat().st_size==int(f['size']):return f['name'],'cached'
    for t in range(4):
        r=subprocess.run(['curl','-sS','-L','--max-time','900','-o',str(p),f"https://drive.usercontent.google.com/download?id={f['id']}&export=download&confirm=t"],capture_output=True,text=True)
        if p.exists() and p.stat().st_size==int(f['size']):return f['name'],'ok'
        time.sleep(2**t)
    return f['name'],'FAIL size=%s'%(p.stat().st_size if p.exists() else None)
with ThreadPoolExecutor(6) as ex:
    res=list(ex.map(get,man))
bad=[r for r in res if not r[1] in('ok','cached')];print(len(res),'files',len(bad),'failed',bad[:5])
