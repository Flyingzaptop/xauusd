"""Event-level best bid/ask series. (1) CME from MBO (emitted at packet end when BBO changes). (2) broker from cTrader depth quotes
(emitted after each timestamp batch when BBO changes)."""
import numpy as np,numba as nb
from book import TICK,NL,HS,HM,_h,_find,_insert,_scan,FLAST,new_state

@nb.njit(cache=True)
def run_bbo(ts_recv,action,side,price,size,oid,flags,base,bsz,asz,hkey,hidx,hsz,hside,meta,out_ts,out_b,out_a,out_bq,out_aq):
    n=len(ts_recv);no=0;bb=meta[0];ba=meta[1];lb=meta[2];la=meta[3]
    for k in range(n):
        a=action[k];s=side[k];p=price[k];q=np.int64(size[k]);o=np.int64(oid[k])
        if a==82:
            bsz[:]=0;asz[:]=0;bb=-1;ba=-1;hkey[:]=-1
        elif a==65:
            if s==66 or s==65:
                idx=(p-base)//TICK
                if 0<=idx<len(bsz):
                    j=_insert(hkey,o);hkey[j]=o;hidx[j]=idx;hsz[j]=q;hside[j]=1 if s==66 else 0
                    if s==66:
                        bsz[idx]+=q
                        if bb<0 or idx>bb:bb=idx
                    else:
                        asz[idx]+=q
                        if ba<0 or idx<ba:ba=idx
        elif a==67 or a==77:
            j=_find(hkey,o)
            if j>=0:
                idx=np.int64(hidx[j]);isb=hside[j]==1;old=np.int64(hsz[j])
                if a==67:
                    new=old-min(old,q);newidx=idx
                else:
                    newidx=(p-base)//TICK;new=q
                if isb:bsz[idx]-=old
                else:asz[idx]-=old
                if new>0 and 0<=newidx<len(bsz):
                    if isb:bsz[newidx]+=new
                    else:asz[newidx]+=new
                    hidx[j]=newidx;hsz[j]=new
                else:hkey[j]=-2
                if isb:
                    if new>0 and (bb<0 or newidx>bb):bb=newidx
                    if bb>=0 and bsz[bb]==0:bb=_scan(bsz,bb,-1)
                else:
                    if new>0 and (ba<0 or newidx<ba):ba=newidx
                    if ba>=0 and asz[ba]==0:ba=_scan(asz,ba,1)
        if (flags[k]&FLAST)!=0 and (bb!=lb or ba!=la):
            out_ts[no]=ts_recv[k];out_b[no]=bb;out_a[no]=ba;out_bq[no]=bsz[bb] if bb>=0 else 0;out_aq[no]=asz[ba] if ba>=0 else 0;no+=1;lb=bb;la=ba
    meta[0]=bb;meta[1]=ba;meta[2]=lb;meta[3]=la
    return no

BTICK=1000  # broker price unit 1e-5 USD; 0.01 USD = 1000
@nb.njit(cache=True)
def run_broker(ts,qid,bid,ask,size,isnew,base,bl,al,hkey,hpx,hside,hsz,meta,out_ts,out_b,out_a):
    # bl/al: summed size per 0.01 level; quotes keyed by quote_id
    n=len(ts);no=0;bb=meta[0];ba=meta[1];lb=meta[2];la=meta[3]
    for k in range(n):
        o=qid[k];isb=bid[k]>0;p=bid[k] if isb else ask[k];idx=(p-base)//BTICK
        if isnew[k]:
            if 0<=idx<len(bl):
                j=_insert(hkey,o);hkey[j]=o;hpx[j]=idx;hside[j]=1 if isb else 0;hsz[j]=abs(size[k])
                if isb:
                    bl[idx]+=abs(size[k])
                    if bb<0 or idx>bb:bb=idx
                else:
                    al[idx]+=abs(size[k])
                    if ba<0 or idx<ba:ba=idx
        else:
            j=_find(hkey,o)
            if j>=0:
                i2=np.int64(hpx[j]);sz=np.int64(hsz[j])
                if hside[j]==1:
                    bl[i2]-=sz
                    if bb==i2 and bl[i2]<=0:bb=_scan(bl,i2,-1)
                else:
                    al[i2]-=sz
                    if ba==i2 and al[i2]<=0:ba=_scan(al,i2,1)
                hkey[j]=-2
        last=(k==n-1) or (ts[k+1]!=ts[k])
        if last and bb>=0 and ba>=0 and bb>=ba:
            # crossed after a full batch: stale quotes left over (e.g. feed switch) -> reset, the stream refills within ~1 s
            bl[:]=0;al[:]=0;hkey[:]=-1;bb=-1;ba=-1;meta[4]+=1;meta[5]=ts[k]
            continue
        if last and (bb!=lb or ba!=la) and bb>=0 and ba>=0 and ts[k]-meta[5]>2_000_000_000:
            out_ts[no]=ts[k];out_b[no]=bb;out_a[no]=ba;no+=1;lb=bb;la=ba
    meta[0]=bb;meta[1]=ba;meta[2]=lb;meta[3]=la
    return no
