import sys,numpy as np,databento as db,pyarrow.parquet as pq,pandas as pd,time
sys.path.insert(0,'/home/user/mbo_code');import bbo,book
T0=pd.Timestamp('2026-07-06 22:00',tz='UTC').value;T1=pd.Timestamp('2026-07-25 00:00',tz='UTC').value
DO_CME='--broker-only' not in sys.argv
# CME
if DO_CME:
  pass
st=db.DBNStore.from_file('/home/user/mbo/glbx-mdp3-20260701-20260730.mbo.dbn.zst');iid=[int(r['symbol']) for r in st.metadata.mappings['GCQ6']][0]
S=None;parts=[];t=time.time()
for a in st.to_ndarray(count=4_000_000):
    a=a[a['instrument_id']==iid]
    if len(a)==0:continue
    if S is None:S=book.new_state(int(a['price'][(a['action']==b'A')&(a['price']<9e18)][0]));meta=np.array([-1,-1,-2,-2],np.int64)
    R=len(a)+1;o=[np.zeros(R,np.int64) for _ in range(5)]
    n=bbo.run_bbo(a['ts_recv'].astype(np.int64),a['action'].view(np.uint8),a['side'].view(np.uint8),a['price'].astype(np.int64),a['size'],a['order_id'],a['flags'],
                  S['base'],S['bsz'],S['asz'],S['hkey'],S['hidx'],S['hsz'],S['hside'],meta,*o)
    parts.append(np.stack([x[:n] for x in o],1))
C=np.concatenate(parts);C=C[(C[:,0]>=T0)&(C[:,0]<T1)]
np.savez('/home/user/mbo_out/cme_bbo_jul.npz',ts=C[:,0],bid=(S['base']+C[:,1]*book.TICK)/1e9,ask=(S['base']+C[:,2]*book.TICK)/1e9,bq=C[:,3],aq=C[:,4])
print('CME bbo changes',len(C),'%.0fs'%(time.time()-t),flush=True)
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
print('broker bbo changes',len(B),'%.0fs'%(time.time()-t))
