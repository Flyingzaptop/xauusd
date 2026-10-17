import numpy as np,sys
sys.path.insert(0,'/home/user/mbo_code')
import sim,book
T=book.TICK;L=128;G=10**9
def msgs(rows):
    return (np.array([r[0] for r in rows],np.int64),np.array([ord(r[1]) for r in rows],np.uint8),np.array([ord(r[2]) for r in rows],np.uint8),
            np.array([r[3]*T for r in rows],np.int64),np.array([r[4] for r in rows],np.uint32),np.array([r[5] for r in rows],np.uint64),np.array([r[6] for r in rows],np.uint8))
def go(rows,dts,dside,latency=0,ttl=100*G):
    S=sim.new_sim_state(5000*T);m=msgs(rows);out=np.zeros((100,8),np.int64);diag=np.zeros(4,np.int64)
    n=sim.run_sim(*m,S['base'],S['bsz'],S['asz'],S['hkey'],S['hidx'],S['hsz'],S['hside'],S['hseq'],S['meta'],S['pa'],S['pside'],S['pidx'],S['pseq'],S['pahead'],S['ptp'],S['pdl'],S['pid'],S['pahead0'],
        np.array(dts,np.int64),np.array(dside,np.int8),latency,ttl,out,100,diag)
    return out[:n],diag
# book: bid 5000: order1(10) ahead, then probe(buy) joins; order3(5) arrives after probe.
rows=[(1*G,'A','B',5000,10,1,L),(1*G+1,'A','A',5002,10,2,L),
      (3*G,'A','B',5000,5,3,L),         # arrives after probe (probe placed at 2G)
      (4*G,'C','B',5000,4,1,L),         # ahead order cancels 4 -> ahead 10-4=6
      (5*G,'F','B',5000,6,1,0),(5*G+1,'C','B',5000,6,1,L),   # ahead order fully executed (F then C) -> ahead 0
      (6*G,'F','B',5000,5,3,L),         # behind-probe order executed -> probe filled
      (9*G,'A','A',5003,1,5,L)]
out,diag=go(rows,[2*G],[1])
print(out,diag)
assert out.shape[0]==1 and out[0,7]==1 and out[0,5]==10 and out[0,2]==6*G and out[0,6]==0,out
# ttl: nothing behind us executes -> probe expires
rows2=[(1*G,'A','B',5000,10,1,L),(1*G+1,'A','A',5002,10,2,L),(30*G,'A','A',5003,1,5,L)]
out2,_=go(rows2,[2*G],[1],ttl=10*G);print(out2);assert out2[0,7]==0
# sell probe + latency: places at 2G+0.5G
rows3=[(1*G,'A','B',5000,10,1,L),(1*G+1,'A','A',5002,10,2,L),(3*G,'A','A',5002,2,7,L),(4*G,'F','A',5002,2,7,L),(9*G,'A','A',5009,1,5,L)]
out3,_=go(rows3,[2*G],[-1],latency=G//2);print(out3);assert out3[0,7]==1 and out3[0,3]==-1
print('sim tests passed')
