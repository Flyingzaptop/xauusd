"""Aligned BTC/ETH tensors (device-resident) and batch construction for SeqModel."""
import numpy as np,torch
HZ=[10,60,300,1800,3600]   # seconds
class Market:
    def __init__(s,files_btc,files_eth,device='cpu'):
        s.dev=device;sy=[s._load(fb) for fb in files_btc],[s._load(fe) for fe in files_eth]
        s.sym=[s._concat(x) for x in sy]
        s0=max(m['s0'] for m in s.sym);s1=min(m['s0']+len(m['s_cnt']) for m in s.sym);s.s0=s0;s.T=s1-s0
        s.t=[]
        for m in s.sym:
            a=s0-m['s0'];sl=slice(a,a+s.T);
            s.t.append(dict(ev=torch.tensor(m['ev'],dtype=torch.float16,device=device),off=torch.tensor(m['s_off'][sl],device=device),cnt=torch.tensor(m['s_cnt'][sl].astype(np.int64),device=device),
                            ssum=torch.tensor(m['s_sum'][sl],device=device),last=torch.tensor(np.log(m['s_last'][sl]),dtype=torch.float64,device=device),
                            msum=torch.tensor(m['m_sum'],device=device),m0=int(m['m0'])))
        print(f'Market: {s.T} aligned seconds; events BTC {len(s.sym[0]["ev"])} ETH {len(s.sym[1]["ev"])}',flush=True)
    @staticmethod
    def _load(f):
        z=np.load(f,allow_pickle=False);return {k:z[k] for k in z.files}
    @staticmethod
    def _concat(parts):
        parts=sorted(parts,key=lambda p:int(p['s0']));out={'s0':int(parts[0]['s0']),'m0':int(parts[0]['m0'])}
        ev=[];off=[];cnt=[];last=[];ss=[];ms=[];base=0;cur_s=out['s0'];cur_m=out['m0']
        for p in parts:
            gap=int(p['s0'])-cur_s                            # fill missing seconds (if any) with empty rows
            if gap>0:off.append(np.full(gap,base));cnt.append(np.zeros(gap,np.int32));last.append(np.full(gap,last[-1][-1] if last else p['s_last'][0]));ss.append(np.zeros((gap,6),np.float32))
            gm=int(p['m0'])-cur_m
            if gm>0:ms.append(np.zeros((gm,6),np.float32))
            ev.append(p['ev']);off.append(p['s_off']+base);cnt.append(p['s_cnt']);last.append(p['s_last']);ss.append(p['s_sum']);ms.append(p['m_sum'])
            base+=len(p['ev']);cur_s=int(p['s0'])+len(p['s_cnt']);cur_m=int(p['m0'])+len(p['m_sum'])
        out.update(ev=np.concatenate(ev),s_off=np.concatenate(off),s_cnt=np.concatenate(cnt),s_last=np.concatenate(last),s_sum=np.concatenate(ss),m_sum=np.concatenate(ms))
        return out
    def batch(s,g,Lf,Lc,K):
        """g: LongTensor [B] anchor second index (aligned); features use seconds <= g (bar g closed), labels from g+1."""
        B=len(g);dev=s.dev;secs=g.view(B,1)-Lf+1+torch.arange(Lf,device=dev).view(1,Lf)
        EV=[];MK=[];SS=[];MS=[];Y=[];PT=[];VOL=[]
        for t in s.t:
            off=t['off'][secs];cnt=t['cnt'][secs];j=(off+cnt).unsqueeze(-1)-K+torch.arange(K,device=dev)
            mk=j>=off.unsqueeze(-1);j=j.clamp(min=0);EV.append(t['ev'][j].float()*mk.unsqueeze(-1));MK.append(mk)
            SS.append(t['ssum'][secs])
            mrel=(s.s0+g)//60-1-t['m0'];mi=mrel.view(B,1)-Lc+1+torch.arange(Lc,device=dev).view(1,Lc);MS.append(t['msum'][mi.clamp(0,len(t['msum'])-1)])  # last closed minute
            PT.append(t['ssum'][(secs+1).clamp(max=s.T-1)])           # next-second summary targets (self-supervised)
            p0=t['last'][g+1];r=torch.stack([(t['last'][g+1+h]-p0)*1e4 for h in HZ],-1).float();Y.append(r)
            VOL.append(t['ssum'][secs][...,3].std(1).clamp(min=0.5))   # 1-s return std in bp over the fine window
        st=lambda x:torch.stack(x,1)
        return dict(ev=st(EV),mask=st(MK),ssum=st(SS),msum=st(MS),pre_t=st(PT),y=st(Y),vol=st(VOL))
