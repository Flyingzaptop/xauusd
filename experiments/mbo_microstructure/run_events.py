"""usage: run_events.py FILE SYMBOL OUT.npz [BIG_ADD=5] [BIG_AGGR=10]"""
import sys,time,numpy as np,databento as db
sys.path.insert(0,'/home/user/mbo_code');import events
f,sym,outp=sys.argv[1:4];BA=int(sys.argv[4]) if len(sys.argv)>4 else 5;BG=int(sys.argv[5]) if len(sys.argv)>5 else 10
st=db.DBNStore.from_file(f);iid=[int(r['symbol']) for r in st.metadata.mappings[sym]][0];S=None;res=[];t0=time.time()
for a in st.to_ndarray(count=4_000_000):
    a=a[a['instrument_id']==iid]
    if len(a)==0:continue
    if S is None:S=events.new_ev_state(int(a['price'][(a['action']==b'A')&(a['price']<9e18)][0]))
    out=np.zeros((len(a)+1,6),np.int64)
    n=events.run_ev(a['ts_recv'].astype(np.int64),a['action'].view(np.uint8),a['side'].view(np.uint8),a['price'].astype(np.int64),a['size'],a['order_id'],a['flags'],
        S['base'],S['bsz'],S['asz'],S['hkey'],S['hidx'],S['hsz'],S['hside'],S['hdisp'],S['hfill'],S['hice'],S['meta'],BA,BG,out)
    res.append(out[:n])
E=np.concatenate(res);np.savez(outp,ev=E)
print('events',len(E),{k:int((E[:,1]==v).sum()) for k,v in [('iceberg',1),('big_add',2),('big_pull',3),('big_aggr',4)]},'%.0fs'%(time.time()-t0))
