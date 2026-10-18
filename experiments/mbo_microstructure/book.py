"""Order book reconstruction from Databento MBO (one instrument) + 100 ms feature grid (numba, no typed dict).
Grid rows use only messages with ts_recv < grid_ts; snapshots are taken only at packet boundaries (previous msg had F_LAST).
Order table: open-addressing hash (linear probing, tombstones reused)."""
import numpy as np,numba as nb

TICK=100_000_000      # 0.1 USD, 1e-9 fixed point
NL=40000              # price levels centered on first price
GRID=100_000_000      # 100 ms
NF=19                 # 0-4 bid sz L1..5, 5-9 ask sz L1..5, 10 OFI, 11 buyvol, 12 sellvol, 13 ntrades, 14 add_b, 15 add_a, 16 cxl_b, 17 cxl_a, 18 last trade px
FLAST=128
HB=25;HS=1<<HB;HM=HS-1

def new_state(first_px):
    base=(first_px//TICK)*TICK-(NL//2)*TICK
    return dict(base=np.int64(base),bsz=np.zeros(NL,np.int64),asz=np.zeros(NL,np.int64),
                hkey=np.full(HS,-1,np.int64),hidx=np.zeros(HS,np.int32),hsz=np.zeros(HS,np.int32),hside=np.zeros(HS,np.int8),
                meta=np.array([-1,-1,0,0,0],np.int64),   # 0 best_bid_idx, 1 best_ask_idx, 2 next_grid, 3 prev_last, 4 n_orders
                acc=np.zeros(9,np.float64),prev=np.array([0,0,-9,-9],np.int64),lasttrade=np.zeros(1,np.float64))

@nb.njit(cache=True)
def _h(o):
    return ((o*np.int64(-7046029254386353131))>>np.int64(20))&np.int64(HM)

@nb.njit(cache=True)
def _find(hkey,o):
    i=_h(o)
    while hkey[i]!=-1:
        if hkey[i]==o:return i
        i=(i+1)&HM
    return -1

@nb.njit(cache=True)
def _insert(hkey,o):
    i=_h(o)
    while hkey[i]>=0:
        i=(i+1)&HM
    return i

@nb.njit(cache=True)
def _scan(sz,start,step):
    i=start;n=len(sz)
    while 0<=i<n and sz[i]==0:i+=step
    if 0<=i<n:return i
    return -1

@nb.njit(cache=True)
def run(ts_recv,action,side,price,size,oid,flags,
        base,bsz,asz,hkey,hidx,hsz,hside,meta,acc,prev,lasttrade,
        out_ts,out_px,out_f,max_rows,chk):
    n=len(ts_recv);rows=0
    bb=meta[0];ba=meta[1];nxt=meta[2];prev_last=meta[3];norders=meta[4]
    pbq=prev[0];paq=prev[1];pbp=prev[2];pap=prev[3]
    for k in range(n):
        t=ts_recv[k]
        if nxt==0:nxt=(t//GRID+1)*GRID
        if prev_last==1:
            while nxt<=t and rows<max_rows:
                out_ts[rows]=nxt
                out_px[rows,0]=(base+bb*TICK)/1e9 if bb>=0 else np.nan
                out_px[rows,1]=(base+ba*TICK)/1e9 if ba>=0 else np.nan
                c=0;i=bb
                while i>=0 and c<5:
                    if bsz[i]>0:
                        out_f[rows,c]=bsz[i];c+=1
                    i-=1
                while c<5:
                    out_f[rows,c]=0;c+=1
                c=0;i=ba
                while 0<=i<len(asz) and c<5:
                    if asz[i]>0:
                        out_f[rows,5+c]=asz[i];c+=1
                    i+=1
                while c<5:
                    out_f[rows,5+c]=0;c+=1
                out_f[rows,10]=acc[0];out_f[rows,11]=acc[1];out_f[rows,12]=acc[2];out_f[rows,13]=acc[3]
                out_f[rows,14]=acc[4];out_f[rows,15]=acc[5];out_f[rows,16]=acc[6];out_f[rows,17]=acc[7]
                out_f[rows,18]=lasttrade[0]
                for j in range(9):acc[j]=0.0
                rows+=1;nxt+=GRID
        a=action[k];s=side[k];p=price[k];q=np.int64(size[k]);o=np.int64(oid[k])
        if a==82:  # R
            bsz[:]=0;asz[:]=0;bb=-1;ba=-1
            hkey[:]=-1;norders=0
        elif a==65: # A
            if s==66 or s==65:
                idx=(p-base)//TICK
                if 0<=idx<len(bsz):
                    j=_insert(hkey,o);hkey[j]=o;hidx[j]=idx;hsz[j]=q;hside[j]=1 if s==66 else 0;norders+=1
                    if s==66:
                        bsz[idx]+=q
                        if bb<0 or idx>bb:bb=idx
                        acc[4]+=1
                    else:
                        asz[idx]+=q
                        if ba<0 or idx<ba:ba=idx
                        acc[5]+=1
        elif a==67 or a==77:
            j=_find(hkey,o)
            if j>=0:
                idx=np.int64(hidx[j]);isb=hside[j]==1;old=np.int64(hsz[j])
                if a==67:
                    rem=min(old,q);new=old-rem;newidx=idx
                    if isb:acc[6]+=1
                    else:acc[7]+=1
                else:
                    newidx=(p-base)//TICK;new=q
                if isb:bsz[idx]-=old
                else:asz[idx]-=old
                if new>0 and 0<=newidx<len(bsz):
                    if isb:bsz[newidx]+=new
                    else:asz[newidx]+=new
                    hidx[j]=newidx;hsz[j]=new
                else:
                    hkey[j]=-2;norders-=1
                if isb:
                    if new>0 and (bb<0 or newidx>bb):bb=newidx
                    if bb>=0 and bsz[bb]==0:bb=_scan(bsz,bb,-1)
                else:
                    if new>0 and (ba<0 or newidx<ba):ba=newidx
                    if ba>=0 and asz[ba]==0:ba=_scan(asz,ba,1)
        elif a==84:
            tidx=(p-base)//TICK
            chk[0]+=1
            if (s==66 and tidx==ba) or (s==65 and tidx==bb):chk[1]+=1
            elif (s==66 and ba>=0 and tidx>ba) or (s==65 and bb>=0 and tidx<bb):chk[2]+=1   # trade through best
            else:chk[3]+=1                                                                   # trade inside/not at best
            acc[3]+=1
            if s==66:acc[1]+=q
            elif s==65:acc[2]+=q
            lasttrade[0]=p/1e9
        bq=bsz[bb] if bb>=0 else 0;aq=asz[ba] if ba>=0 else 0
        bp=bb;ap=ba
        if pbp!=-9 and (bp!=pbp or bq!=pbq or ap!=pap or aq!=paq):
            e=0.0
            if bp>pbp:e+=bq
            elif bp==pbp:e+=bq-pbq
            else:e-=pbq
            if ap<pap or pap<0:e-=aq
            elif ap==pap:e-=aq-paq
            else:e+=paq
            acc[0]+=e
        pbq=bq;paq=aq;pbp=bp;pap=ap
        prev_last=1 if (flags[k]&FLAST)!=0 else 0
    meta[0]=bb;meta[1]=ba;meta[2]=nxt;meta[3]=prev_last;meta[4]=norders
    prev[0]=pbq;prev[1]=paq;prev[2]=pbp;prev[3]=pap
    return rows
