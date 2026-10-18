"""usage: run_probes.py FILE SYMBOL GRIDDIR OUT.npz LATENCY_MS TTL_S [DECISIONS.npz]
Without DECISIONS: unconditional probes, buy+sell every 2 s inside the session. With DECISIONS (ts,side arrays): conditional probes."""
import sys,time,numpy as np,glob,databento as db
sys.path.insert(0,'/home/user/mbo_code');import sim
f,sym,gd,outp,lat_ms,ttl_s=sys.argv[1:7];dec=sys.argv[7] if len(sys.argv)>7 else None
lat=int(float(lat_ms)*1e6);ttl=int(float(ttl_s)*1e9)
if dec:
    Dz=np.load(dec);dts=Dz['ts'].astype(np.int64);dside=Dz['side'].astype(np.int8)
else:
    Z=[np.load(x) for x in sorted(glob.glob(gd+'/chunk_*.npz'))];ts=np.concatenate([z['ts'] for z in Z]);px=np.concatenate([z['px'] for z in Z])
    sec=(ts//10**9)%86400;ok=~((sec>=22*3600)&(sec<23*3600))&np.isfinite(px).all(1)&((px[:,1]-px[:,0])>0)
    t2=ts[ok][::20];dts=np.repeat(t2,2);dside=np.tile(np.array([1,-1],np.int8),len(t2))
print('decisions',len(dts),flush=True)
st=db.DBNStore.from_file(f);iid=[int(r['symbol']) for r in st.metadata.mappings[sym]][0]
S=None;diag=np.zeros(4,np.int64);res=[];t0=time.time();msgs=0
for a in st.to_ndarray(count=4_000_000):
    a=a[a['instrument_id']==iid]
    if len(a)==0:continue
    if S is None:
        first=a['price'][(a['action']==b'A')&(a['price']<9e18)][0];S=sim.new_sim_state(int(first))
    out=np.zeros((2_000_000,8),np.int64)
    n=sim.run_sim(a['ts_recv'].astype(np.int64),a['action'].view(np.uint8),a['side'].view(np.uint8),a['price'].astype(np.int64),a['size'],a['order_id'],a['flags'],
        S['base'],S['bsz'],S['asz'],S['hkey'],S['hidx'],S['hsz'],S['hside'],S['hseq'],S['meta'],S['pa'],S['pside'],S['pidx'],S['pseq'],S['pahead'],S['ptp'],S['pdl'],S['pid'],S['pahead0'],
        dts,dside,lat,ttl,out,len(out),diag)
    assert n<len(out)-1;res.append(out[:n]);msgs+=len(a)
    if len(res)%5==0:print('msgs',msgs,'results',sum(len(r) for r in res),'%.0fs'%(time.time()-t0),flush=True)
R=np.concatenate(res);np.savez(outp,res=R,dts=dts,dside=dside,base=S['base'],diag=diag)
print('done results',len(R),'fills',(R[:,7]==1).sum(),'ttl',(R[:,7]==0).sum(),'empty',(R[:,7]==-1).sum(),'diag',diag.tolist(),'%.0fs'%(time.time()-t0))
