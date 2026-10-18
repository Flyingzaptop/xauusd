import numpy as np,sys
sys.path.insert(0,'/home/user/mbo_code')
import book
T=book.TICK
def msgs(rows):
    return (np.array([r[0] for r in rows],np.int64),np.array([ord(r[1]) for r in rows],np.uint8),np.array([ord(r[2]) for r in rows],np.uint8),
            np.array([r[3]*T for r in rows],np.int64),np.array([r[4] for r in rows],np.uint32),np.array([r[5] for r in rows],np.uint64),np.array([r[6] for r in rows],np.uint8))
def go(rows,first=5000):
    st=book.new_state(first*T);m=msgs(rows);R=1000
    ots=np.zeros(R,np.int64);opx=np.zeros((R,2));of=np.zeros((R,book.NF),np.float32)
    n=book.run(*m,st['base'],st['bsz'],st['asz'],st['hkey'],st['hidx'],st['hsz'],st['hside'],st['meta'],st['acc'],st['prev'],st['lasttrade'],ots,opx,of,R,np.zeros(4,np.int64))
    return st,n,ots[:n],opx[:n],of[:n]
L=128;G=book.GRID
rows=[(G//2,'A','B',5000,10,1,L),(G//2+1,'A','A',5001,7,2,L),(G+5,'A','B',4999,3,3,L),
      (G+6,'T','B',5001,2,9,L),(G+7,'C','A',5001,2,2,L),
      (2*G+1,'M','B',5000,4,1,L),
      (3*G+1,'C','B',5000,4,1,L),
      (4*G+1,'A','A',5001,1,4,L)]
st,n,ts,px,f=go(rows)
print('rows',n);print(ts//G);print(px);print(f[:,:14])
assert n==4,n
assert f[0,0]==10 and f[0,5]==7, f[0]
assert f[1,0]==10 and f[1,1]==3 and f[1,5]==5 and f[1,11]==2 and f[1,13]==1, f[1]
assert f[2,0]==4, f[2]
assert f[3,0]==3 and f[3,5]==5, f[3]
print('book test passed')
