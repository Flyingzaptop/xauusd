import sys,numpy as np,databento as db,pyarrow.parquet as pq,pandas as pd,time
sys.path.insert(0,'/home/user/mbo_code');import bbo,book
T0=pd.Timestamp('2026-07-07 00:00',tz='UTC').value;T1=pd.Timestamp('2026-07-25 00:00',tz='UTC').value
DO_CME='--broker-only' not in sys.argv
# broker
pf=pq.ParquetFile('/home/user/l2/l2_consolidated.parquet');BS=None;parts=[]
for i in range(pf.metadata.num_row_groups):
    sts=pf.metadata.row_group(i).column(0).statistics
    if pd.Timestamp(sts.max).value<T0 or pd.Timestamp(sts.min).value>T1:continue
    tb=pf.read_row_group(i)
    ts=tb.column('timestamp').cast('int64').to_numpy();q=tb.column('quote_id').to_numpy();bi=tb.column('bid').to_numpy();ak=tb.column('ask').to_numpy();sz=tb.column('size').to_numpy()
    nw=(tb.column('type').to_numpy(zero_copy_only=False)=='new')
    if BS is None:
        p0=int(np.r_[bi[bi>0],ak[ak>0]][0]);base=(p0//1000)*1000-100000*1000
        BS=dict(bl=np.zeros(200000,np.int64),al=np.zeros(200000,np.int64),hkey=np.full(book.HS,-1,np.int64),hpx=np.zeros(book.HS,np.int32),hside=np.zeros(book.HS,np.int8),hsz=np.zeros(book.HS,np.int32),meta=np.array([-1,-1,-2,-2,0,0],np.int64))
    R=len(ts)+1;ot=np.zeros(R,np.int64);ob=np.zeros(R,np.int64);oa=np.zeros(R,np.int64)
    n=bbo.run_broker(ts,q,bi,ak,sz,nw,np.int64(base),BS['bl'],BS['al'],BS['hkey'],BS['hpx'],BS['hside'],BS['hsz'],BS['meta'],ot,ob,oa)
    parts.append(np.stack([ot[:n],ob[:n],oa[:n]],1))
B=np.concatenate(parts);B=B[(B[:,0]>=T0)&(B[:,0]<T1)]
np.savez('/home/user/mbo_out/broker_bbo_jul.npz',ts=B[:,0],bid=(base+B[:,1]*1000)/1e5,ask=(base+B[:,2]*1000)/1e5)
print('resets',BS['meta'][4],'broker bbo changes',len(B),'%.0fs'%(time.time()-t))
