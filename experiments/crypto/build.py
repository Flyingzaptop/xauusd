import glob,sys,numpy as np,pandas as pd
cols=['open_time','open','high','low','close','volume','close_time','quote_volume','count','taker_buy_volume','taker_buy_quote_volume','ignore']
for s in sys.argv[1:]:
    parts=[]
    for f in sorted(glob.glob(f'/home/user/crypto/um/{s}/klines/*.csv')):
        x=pd.read_csv(f,header=None,names=cols,low_memory=False)
        x=x[pd.to_numeric(x.open_time,errors='coerce').notna()].astype({'open_time':'int64'})
        parts.append(x[['open_time','open','high','low','close','volume','taker_buy_volume','count']].astype(float).astype({'open_time':'int64'}))
    k=pd.concat(parts).drop_duplicates('open_time').sort_values('open_time')
    k['open_time']=np.where(k.open_time>1e14,k.open_time//1000,k.open_time)   # some recent files use microseconds
    k.to_parquet(f'/home/user/crypto/{s}_1m.parquet',index=False)
    fr=[]
    for f in sorted(glob.glob(f'/home/user/crypto/um/{s}/funding/*.csv')):
        x=pd.read_csv(f);x.columns=[c.strip() for c in x.columns];fr.append(x)
    fr=pd.concat(fr);fr.to_parquet(f'/home/user/crypto/{s}_funding.parquet',index=False)
    gap=np.diff(k.open_time.values)//60000
    print(s,'bars',len(k),pd.to_datetime(k.open_time.iloc[0],unit='ms'),pd.to_datetime(k.open_time.iloc[-1],unit='ms'),'gaps>1m',int((gap>1).sum()),'funding rows',len(fr))
