import numpy as np,numba as nb
@nb.njit(cache=True)
def race(i0,side,target,stop,H,L,C,sess,tmax):
    s0=sess[i0];j=i0+1;n=len(C)
    while j<n and j<=i0+tmax and sess[j]==s0:
        if side>0:
            hs=L[j]<=stop;ht=H[j]>=target
        else:
            hs=H[j]>=stop;ht=L[j]<=target
        if hs:return -1,j
        if ht:return 1,j
        j+=1
    return 0,min(j-1,n-1)
@nb.njit(cache=True)
def run_line(sig_a,sig_b,line,H,L,C,sess,tmax,mode,out):
    """mode 0: revert to line (target=line, stop=d beyond). mode 1: trend away (target=d further, stop=line). sig_a = price above line."""
    i=0;n=len(C);k=0
    while i<n-1:
        if sig_a[i] or sig_b[i]:
            above=sig_a[i];e=C[i];dd=abs(e-line[i])
            if mode==0:
                side=-1 if above else 1;tg=line[i];st=e-side*dd
            else:
                side=1 if above else -1;tg=e+side*dd;st=line[i]
            r,j=race(i,side,tg,st,H,L,C,sess,tmax);ex=tg if r==1 else (st if r==-1 else C[j])
            out[k,0]=i;out[k,1]=1e4*side*(ex-e)/e;k+=1;i=j+1
        else:i+=1
    return k
@nb.njit(cache=True)
def nonoverlap(idx,h):
    keep=np.zeros(len(idx),np.bool_);last=-10**12
    for k in range(len(idx)):
        if idx[k]>=last+h:keep[k]=True;last=idx[k]
    return keep
