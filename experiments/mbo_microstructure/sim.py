"""Passive-order (probe) simulator on the MBO stream with exact queue position (price-time priority).
A probe is a virtual 1-lot LIMIT order that joins the best bid (buy) / best ask (sell) at decision_ts+latency, and is cancelled after ttl.
Queue: ahead = volume at the level at arrival; reduced by cancels/size-decreases/price-moves of orders that arrived earlier (hseq<probe_seq).
Fill: when an order that arrived AFTER the probe at the same price is executed (F), or ahead volume reaches 0 and a fill happens at the level.
Probes do not move the market (1 lot). Fills in the order table use F (fill) messages; the quantity bookkeeping uses the following C/M messages."""
import numpy as np,numba as nb
from book import TICK,NL,HS,HM,_h,_find,_insert,_scan,FLAST

MAXP=64
def new_sim_state(first_px):
    base=(first_px//TICK)*TICK-(NL//2)*TICK
    z=lambda n,dt:np.zeros(n,dt)
    return dict(base=np.int64(base),bsz=z(NL,np.int64),asz=z(NL,np.int64),hkey=np.full(HS,-1,np.int64),hidx=z(HS,np.int32),hsz=z(HS,np.int32),
                hside=z(HS,np.int8),hseq=z(HS,np.int64),
                meta=np.array([-1,-1,0,0,0],np.int64),   # 0 bb,1 ba,2 seq counter,3 prev_last,4 decision pointer
                pa=z(MAXP,np.int8),pside=z(MAXP,np.int8),pidx=z(MAXP,np.int64),pseq=z(MAXP,np.int64),pahead=z(MAXP,np.int64),
                ptp=z(MAXP,np.int64),pdl=z(MAXP,np.int64),pid=z(MAXP,np.int64),pahead0=z(MAXP,np.int64))

@nb.njit(cache=True)
def run_sim(ts_recv,action,side,price,size,oid,flags,
            base,bsz,asz,hkey,hidx,hsz,hside,hseq,meta,pa,pside,pidx,pseq,pahead,ptp,pdl,pid,pahead0,
            dts,dside,latency,ttl,out,max_out,diag):
    # out columns: 0 id, 1 place_ts, 2 fill_ts (-1 none), 3 side, 4 price_idx, 5 ahead0, 6 ahead_at_fill, 7 outcome(1 fill,0 ttl,-1 empty book)
    n=len(ts_recv);no=0
    bb=meta[0];ba=meta[1];seq=meta[2];prev_last=meta[3];dp=meta[4]
    nd=len(dts)
    for k in range(n):
        t=ts_recv[k]
        if prev_last==1:
            # placements due
            while dp<nd and dts[dp]+latency<=t:
                sd=dside[dp]
                if (sd>0 and bb<0) or (sd<0 and ba<0):
                    if no<max_out:
                        out[no,0]=dp;out[no,1]=t;out[no,2]=-1;out[no,3]=sd;out[no,4]=-1;out[no,5]=0;out[no,6]=0;out[no,7]=-1;no+=1
                elif dts[dp]+latency+ttl<=t:
                    # no message arrived between placement and expiry: cannot be filled
                    if no<max_out:
                        out[no,0]=dp;out[no,1]=dts[dp]+latency;out[no,2]=-1;out[no,3]=sd;out[no,4]=bb if sd>0 else ba;out[no,5]=bsz[bb] if sd>0 else asz[ba];out[no,6]=out[no,5];out[no,7]=0;no+=1
                else:
                    slot=-1
                    for i in range(MAXP):
                        if pa[i]==0:
                            slot=i;break
                    if slot>=0:
                        pa[slot]=1;pside[slot]=sd;pid[slot]=dp;ptp[slot]=dts[dp]+latency;pdl[slot]=dts[dp]+latency+ttl;pseq[slot]=seq
                        if sd>0:
                            pidx[slot]=bb;pahead[slot]=bsz[bb]
                        else:
                            pidx[slot]=ba;pahead[slot]=asz[ba]
                        pahead0[slot]=pahead[slot]
                    else:
                        diag[0]+=1
                dp+=1
            # ttl expiry
            for i in range(MAXP):
                if pa[i]==1 and pdl[i]<=t:
                    if no<max_out:
                        out[no,0]=pid[i];out[no,1]=ptp[i];out[no,2]=-1;out[no,3]=pside[i];out[no,4]=pidx[i];out[no,5]=pahead0[i];out[no,6]=pahead[i];out[no,7]=0;no+=1
                    pa[i]=0
        a=action[k];s=side[k];p=price[k];q=np.int64(size[k]);o=np.int64(oid[k])
        seq+=1
        if a==82:
            bsz[:]=0;asz[:]=0;bb=-1;ba=-1;hkey[:]=-1
            for i in range(MAXP):pa[i]=0
        elif a==65:
            if s==66 or s==65:
                idx=(p-base)//TICK
                if 0<=idx<len(bsz):
                    j=_insert(hkey,o);hkey[j]=o;hidx[j]=idx;hsz[j]=q;hside[j]=1 if s==66 else 0;hseq[j]=seq
                    if s==66:
                        bsz[idx]+=q
                        if bb<0 or idx>bb:bb=idx
                    else:
                        asz[idx]+=q
                        if ba<0 or idx<ba:ba=idx
        elif a==67 or a==77:
            j=_find(hkey,o)
            if j>=0:
                idx=np.int64(hidx[j]);isb=hside[j]==1;old=np.int64(hsz[j]);oseq=hseq[j]
                if a==67:
                    rem=min(old,q);new=old-rem;newidx=idx
                else:
                    newidx=(p-base)//TICK;new=q
                lost=old-new if (a==77 and newidx==idx and new<=old) else old   # volume leaving the original queue spot
                keep_prio=(a==67) or (a==77 and newidx==idx and new<=old)
                # probes affected: orders that were ahead of the probe at this price/side
                for i in range(MAXP):
                    if pa[i]==1 and pidx[i]==idx and ((pside[i]>0)==isb) and oseq<pseq[i]:
                        pahead[i]-=(rem if a==67 else lost)
                        if pahead[i]<0:pahead[i]=0
                if isb:bsz[idx]-=old
                else:asz[idx]-=old
                if new>0 and 0<=newidx<len(bsz):
                    if isb:bsz[newidx]+=new
                    else:asz[newidx]+=new
                    hidx[j]=newidx;hsz[j]=new
                    if not keep_prio:hseq[j]=seq
                else:
                    hkey[j]=-2
                if isb:
                    if new>0 and (bb<0 or newidx>bb):bb=newidx
                    if bb>=0 and bsz[bb]==0:bb=_scan(bsz,bb,-1)
                else:
                    if new>0 and (ba<0 or newidx<ba):ba=newidx
                    if ba>=0 and asz[ba]==0:ba=_scan(asz,ba,1)
        elif a==70:   # F: resting order executed
            j=_find(hkey,o)
            if j>=0:
                idx=np.int64(hidx[j]);isb=hside[j]==1;oseq=hseq[j]
                for i in range(MAXP):
                    if pa[i]==1 and pidx[i]==idx and ((pside[i]>0)==isb) and oseq>pseq[i]:
                        # an order behind the probe traded -> the probe (in front of it) would have been filled
                        if pahead[i]>0:diag[1]+=1
                        if no<max_out:
                            out[no,0]=pid[i];out[no,1]=ptp[i];out[no,2]=t;out[no,3]=pside[i];out[no,4]=pidx[i];out[no,5]=pahead0[i];out[no,6]=pahead[i];out[no,7]=1;no+=1
                        pa[i]=0
        prev_last=1 if (flags[k]&FLAST)!=0 else 0
    meta[0]=bb;meta[1]=ba;meta[2]=seq;meta[3]=prev_last;meta[4]=dp
    return no
