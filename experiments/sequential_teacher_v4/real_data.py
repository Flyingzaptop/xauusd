"""Prepare small causal 222-instrument panel; declared windows are not gap-filtered.
Config windows: [{date:'YYYY-MM-DD', open:'08:00', close:'18:00', split:'train'|'dev'|'test'}].
These are strategy trading windows, not an inferred broker calendar. Planned holiday
exclusions must be supplied explicitly before outcomes are examined.
"""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np,pandas as pd,pyarrow.parquet as pq
from environment import Episode
BASE=Path(__file__).parent.parent/'meta_reward_transformer_v3'
spec=importlib.util.spec_from_file_location('legacy_causal_features',BASE/'features.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def prepare(raw_dir,config,out):
    raw_dir=Path(raw_dir);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((BASE/'drive_manifest.json').read_text());files=sorted([v['name'] for v in manifest]);missing=[f for f in files if not (raw_dir/f).exists()]
    if missing:raise FileNotFoundError(f'Full 222-instrument input required; missing {len(missing)} files. No silent substitution.')
    windows=config['windows'];assert {w['split'] for w in windows}=={'train','dev','test'}
    bounds=[(int(pd.Timestamp(w['date']+'T'+w['open'],tz='UTC').timestamp()),int(pd.Timestamp(w['date']+'T'+w['close'],tz='UTC').timestamp())) for w in windows]
    assert all(b>a for a,b in bounds);order=sorted(range(len(bounds)),key=lambda i:bounds[i][0]);assert all(bounds[order[j]][1]<bounds[order[j+1]][0] for j in range(len(order)-1))
    split_order=[windows[i]['split'] for i in order];assert split_order==sorted(split_order,key={'train':0,'dev':1,'test':2}.get),'Chronological split order required'
    for split in ['train','dev','test']:
        lo=min(bounds[i][0] for i,w in enumerate(windows) if w['split']==split)
        if split!='train':
            earlier=[bounds[i][1] for i,w in enumerate(windows) if {'train':0,'dev':1,'test':2}[w['split']]<{'dev':1,'test':2}[split]]
            assert lo-max(earlier)>=1560*60,'Purge 1440-minute features plus 120-minute horizon across splits'
    start=min(a for a,b in bounds)-1560*60;end=max(b for a,b in bounds);N=(end-start)//60+1;grid=np.unique(np.concatenate([np.arange(a-8*60,b+60,60,dtype=np.int64) for a,b in bounds]));F=len(module.NAMES)
    mm=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype=np.float16,shape=(len(grid),len(files),F));audits=[];gold=None
    for j,name in enumerate(files):
        table=pq.read_table(raw_dir/name);block,audit=module.encode(table,start,N,grid);mm[:,j]=block.T;audits.append({'file':name,**audit})
        if name.startswith('XAUUSD_'):
            df=table.to_pandas().sort_values('timestamp');ts=pd.DatetimeIndex(df.timestamp).as_unit('s').asi8
            idx=np.searchsorted(ts,grid);valid=idx<len(ts);safe=idx.clip(max=len(ts)-1);valid&=ts[safe]==grid;gold=np.full(len(grid),np.nan);gold[valid]=df.open.to_numpy()[safe[valid]]
        del table,block
    mm.flush();assert gold is not None;np.save(out/'opens.npy',gold);np.save(out/'timestamps.npy',grid)
    meta={'config':config,'symbols':files,'feature_names':module.NAMES,'context':8,'feature_transform':'asinh finite values; concatenate finite mask; only t-8 through t-1','normalizer':'causal rolling feature formulas, no global fitted normalizer','source_code_sha256':hashlib.sha256((BASE/'features.py').read_bytes()).hexdigest(),'gold_index':next(i for i,f in enumerate(files) if f.startswith('XAUUSD_')),'windows_bounds':bounds}
    (out/'panel.json').write_text(json.dumps(meta,indent=2));(out/'data_audit.json').write_text(json.dumps(audits,indent=2));return meta

def load_episodes(path,gold_only=False):
    p=Path(path);m=json.loads((p/'panel.json').read_text());grid=np.load(p/'timestamps.npy');price=np.load(p/'opens.npy');raw=np.load(p/'features.npy',mmap_mode='r');episodes={'train':[],'dev':[],'test':[]}
    for w,(lo,hi) in zip(m['config']['windows'],m['windows_bounds']):
        a=int(np.searchsorted(grid,lo));b=int(np.searchsorted(grid,hi));v=np.array(raw[a-8:b],dtype=np.float32);mask=np.isfinite(v)
        if gold_only:
            selector=np.zeros(v.shape[1],dtype=bool);selector[m['gold_index']]=True;mask &= selector[None,:,None]
        z=np.concatenate([np.where(mask,np.arcsinh(np.nan_to_num(v)),0),mask],axis=-1).reshape(len(v),-1).astype(np.float32)
        # strided views avoid duplicating the full feature panel eight times.
        obs=np.lib.stride_tricks.sliding_window_view(z,8,axis=0).transpose(0,2,1)
        ep=Episode(grid[a:b+1],price[a:b+1],obs,int(hi),w['date']);episodes[w['split']].append(ep)
    return episodes,m

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--raw-dir',required=True);a.add_argument('--config',required=True);a.add_argument('--out',required=True);args=a.parse_args();prepare(args.raw_dir,json.loads(Path(args.config).read_text()),args.out)
