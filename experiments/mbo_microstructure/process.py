"""Stream a monthly MBO file, rebuild the book for one outright contract, write 100 ms grid shards.
usage: process.py FILE SYMBOL OUTDIR [MAX_CHUNKS]"""
import sys,re,time,json,numpy as np,databento as db
sys.path.insert(0,'/home/user/mbo_code');import book
f,sym,outd=sys.argv[1:4];maxc=int(sys.argv[4]) if len(sys.argv)>4 else 10**9
from pathlib import Path;Path(outd).mkdir(parents=True,exist_ok=True)
st=db.DBNStore.from_file(f);m=st.metadata
iid=[int(r['symbol']) for r in m.mappings[sym]];assert len(iid)==1,iid;iid=iid[0];print('instrument',sym,iid,flush=True)
CHK=np.zeros(4,np.int64);S=None;t0=time.time();tot=0;crossed=0;rows_total=0;chunk=0
for a in st.to_ndarray(count=4_000_000):
    a=a[a['instrument_id']==iid]
    if len(a)==0:continue
    if S is None:
        first=a['price'][(a['action']==b'A')&(a['price']<9e18)][0];S=book.new_state(int(first))
    R=4_000_000;ots=np.zeros(R,np.int64);opx=np.zeros((R,2));of=np.zeros((R,book.NF),np.float32)
    n=book.run(a['ts_recv'].astype(np.int64),a['action'].view(np.uint8),a['side'].view(np.uint8),a['price'].astype(np.int64),a['size'],a['order_id'],a['flags'],
               S['base'],S['bsz'],S['asz'],S['hkey'],S['hidx'],S['hsz'],S['hside'],S['meta'],S['acc'],S['prev'],S['lasttrade'],ots,opx,of,R,CHK)
    assert n<R-1,n
    ots,opx,of=ots[:n],opx[:n],of[:n]
    ok=np.isfinite(opx).all(1);crossed+=int((opx[ok,0]>=opx[ok,1]).sum());rows_total+=n
    np.savez(f'{outd}/chunk_{chunk:04d}.npz',ts=ots,px=opx,f=of);chunk+=1;tot+=len(a)
    print(f'chunk {chunk} msgs {tot} rows {rows_total} crossed {crossed} orders_open {S["meta"][4]} {time.time()-t0:.0f}s trades {CHK.tolist()}',flush=True)
    if chunk>=maxc:break
json.dump({'msgs':tot,'rows':rows_total,'crossed_rows':crossed,'trade_check':CHK.tolist()},open(f'{outd}/summary.json','w'))
