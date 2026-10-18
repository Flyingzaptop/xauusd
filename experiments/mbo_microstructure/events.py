"""Large-player event detector on one GC outright (numba). Event rows:
 0 ts_recv, 1 type, 2 side(+1 bid/buy, -1 ask/sell), 3 size, 4 distance_ticks_from_same_side_best, 5 extra
 types: 1 ICEBERG (resting order executed beyond its max displayed size; first detection), 2 BIG_ADD (limit add >= big_add within 3 ticks of best),
        3 BIG_PULL (cancel removing >= big_add lots resting within 3 ticks of best, not after a fill), 4 BIG_AGGR (aggressor volume in one packet >= big_aggr, side=aggressor; extra = levels swept)"""
import numpy as np,numba as nb
from book import TICK,NL,HS,HM,_h,_find,_insert,_scan,FLAST

def new_ev_state(first_px):
    base=(first_px//TICK)*TICK-(NL//2)*TICK;z=lambda n,dt:np.zeros(n,dt)
    return dict(base=np.int64(base),bsz=z(NL,np.int64),asz=z(NL,np.int64),hkey=np.full(HS,-1,np.int64),hidx=z(HS,np.int32),hsz=z(HS,np.int32),hside=z(HS,np.int8),
                hdisp=z(HS,np.int32),hfill=z(HS,np.int32),hice=z(HS,np.int8),meta=np.array([-1,-1,0,0,0,0,0],np.int64))

@nb.njit(cache=True)
def run_ev(ts,action,side,price,size,oid,flags,base,bsz,asz,hkey,hidx,hsz,hside,hdisp,hfill,hice,meta,big_add,big_aggr,out):
    n=len(ts);no=0;bb=meta[0];ba=meta[1];pb=meta[2];ps=meta[3];pmin=meta[4];pmax=meta[5];justfilled=meta[6]
    for k in range(n):
        a=action[k];s=side[k];p=price[k];q=np.int64(size[k]);o=np.int64(oid[k]);t=ts[k]
        if a==82:
            bsz[:]=0;asz[:]=0;bb=-1;ba=-1;hkey[:]=-1
        elif a==65:
            if s==66 or s==65:
                idx=(p-base)//TICK
                if 0<=idx<len(bsz):
                    j=_insert(hkey,o);hkey[j]=o;hidx[j]=idx;hsz[j]=q;hside[j]=1 if s==66 else 0;hdisp[j]=q;hfill[j]=0;hice[j]=0
                    if q>=big_add:
                        dist=(bb-idx) if s==66 else (idx-ba)
                        if (s==66 and bb>=0 and dist<=3) or (s==65 and ba>=0 and dist<=3):
                            out[no,0]=t;out[no,1]=2;out[no,2]=1 if s==66 else -1;out[no,3]=q;out[no,4]=dist;out[no,5]=0;no+=1
                    if s==66:
                        bsz[idx]+=q
                        if bb<0 or idx>bb:bb=idx
                    else:
                        asz[idx]+=q
                        if ba<0 or idx<ba:ba=idx
        elif a==70:
            j=_find(hkey,o)
            if j>=0:
                hfill[j]+=q
                if hice[j]==0 and hfill[j]>hdisp[j]:
                    hice[j]=1;isb=hside[j]==1;idx=np.int64(hidx[j])
                    out[no,0]=t;out[no,1]=1;out[no,2]=1 if isb else -1;out[no,3]=hdisp[j];out[no,4]=(bb-idx) if isb else (idx-ba);out[no,5]=hfill[j];no+=1
                justfilled=o
        elif a==84:
            if s==66:pb+=q
            elif s==65:ps+=q
            ti=(p-base)//TICK
            if pmin<0 or ti<pmin:pmin=ti
            if ti>pmax:pmax=ti
        elif a==67 or a==77:
            j=_find(hkey,o)
            if j>=0:
                idx=np.int64(hidx[j]);isb=hside[j]==1;old=np.int64(hsz[j])
                if a==67:
                    new=old-min(old,q);newidx=idx
                    if q>=big_add and o!=justfilled:
                        dist=(bb-idx) if isb else (idx-ba)
                        if dist<=3:
                            out[no,0]=t;out[no,1]=3;out[no,2]=1 if isb else -1;out[no,3]=q;out[no,4]=dist;out[no,5]=0;no+=1
                else:
                    newidx=(p-base)//TICK;new=q
                    if new>hdisp[j]:hdisp[j]=new
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
        if (flags[k]&FLAST)!=0:
            if pb>=big_aggr or ps>=big_aggr:
                sd=1 if pb>=ps else -1
                out[no,0]=t;out[no,1]=4;out[no,2]=sd;out[no,3]=pb if sd>0 else ps;out[no,4]=0;out[no,5]=(pmax-pmin+1) if pmin>=0 else 0;no+=1
            pb=0;ps=0;pmin=-1;pmax=-1;justfilled=-1
    meta[0]=bb;meta[1]=ba;meta[2]=pb;meta[3]=ps;meta[4]=pmin;meta[5]=pmax;meta[6]=justfilled
    return no
