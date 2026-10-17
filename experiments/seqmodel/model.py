"""Hierarchical event-sequence model: event encoder -> 1-s patch tokens (+ 1-min coarse tokens) -> long causal backbone
(Mamba-2 + attention hybrid, or pure attention fallback) over interleaved BTC/ETH tokens -> pretraining + multi-horizon heads."""
import math,torch,torch.nn as nn,torch.nn.functional as F
class EventEncoder(nn.Module):
    def __init__(s,fe=6,fs=6,de=64,d=256):
        super().__init__();s.mlp=nn.Sequential(nn.Linear(fe+1,de),nn.GELU(),nn.Linear(de,de),nn.GELU())
        s.att=nn.Linear(de,1);s.out=nn.Sequential(nn.Linear(3*de+fs,d),nn.GELU(),nn.Linear(d,d))
    def forward(s,ev,mask,summ):                       # ev [B,L,K,6], mask [B,L,K], summ [B,L,6]
        K=ev.shape[2];pos=torch.linspace(0,1,K,device=ev.device).expand(*ev.shape[:3]).unsqueeze(-1)
        h=s.mlp(torch.cat([ev,pos],-1));m=mask.unsqueeze(-1)
        mean=(h*m).sum(2)/m.sum(2).clamp(min=1);mx=h.masked_fill(~m,-1e4).amax(2).masked_fill(m.sum(2)==0,0)
        w=s.att(h).masked_fill(~m,-1e4).softmax(2);ap=(w*h*m).sum(2)
        return s.out(torch.cat([mean,mx,ap,summ],-1))
class AttnBlock(nn.Module):
    def __init__(s,d,heads,drop):
        super().__init__();s.n1=nn.LayerNorm(d);s.qkv=nn.Linear(d,3*d);s.o=nn.Linear(d,d);s.n2=nn.LayerNorm(d);s.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d));s.h=heads;s.drop=nn.Dropout(drop)
    def forward(s,x):
        B,L,D=x.shape;q,k,v=s.qkv(s.n1(x)).view(B,L,3,s.h,D//s.h).permute(2,0,3,1,4)
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,L,D);x=x+s.drop(s.o(a))
        return x+s.drop(s.ff(s.n2(x)))
class MambaBlock(nn.Module):
    def __init__(s,d,drop):
        super().__init__();from mamba_ssm import Mamba2;s.n=nn.LayerNorm(d);s.m=Mamba2(d_model=d,d_state=64,d_conv=4,expand=2,headdim=64);s.drop=nn.Dropout(drop)
        s.n2=nn.LayerNorm(d);s.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x):
        x=x+s.drop(s.m(s.n(x)));return x+s.drop(s.ff(s.n2(x)))
class SeqModel(nn.Module):
    def __init__(s,d=256,depth=8,heads=8,backend='attn',attn_every=4,drop=0.1,n_h=5,Lc=1440,Lf=3600):
        super().__init__();s.enc=EventEncoder(d=d);s.coarse=nn.Sequential(nn.Linear(6,d),nn.GELU(),nn.Linear(d,d))
        s.sym=nn.Embedding(2,d);s.kind=nn.Embedding(2,d);s.pos_c=nn.Embedding(Lc,d);s.pos_f=nn.Embedding(Lf,d)
        blocks=[]
        for i in range(depth):
            if backend=='mamba' and (i+1)%attn_every!=0:blocks.append(MambaBlock(d,drop))
            else:blocks.append(AttnBlock(d,heads,drop))
        s.blocks=nn.ModuleList(blocks);s.norm=nn.LayerNorm(d)
        s.pre=nn.Linear(d,6)            # next-second summary of the token's own symbol (self-supervised)
        s.head=nn.Linear(2*d,2*n_h)     # [BTC horizons, ETH horizons] from the last BTC and ETH fine tokens
        s.n_h=n_h
    def forward(s,batch):
        ev,mask,ssum,msum=batch['ev'],batch['mask'],batch['ssum'],batch['msum']   # ev [B,2,Lf,K,6] ...; msum [B,2,Lc,6]
        B,_,Lf,K,_=ev.shape;Lc=msum.shape[2];dev=ev.device
        f=s.enc(ev.flatten(0,1),mask.flatten(0,1),ssum.flatten(0,1)).view(B,2,Lf,-1)
        c=s.coarse(msum)
        sy=s.sym(torch.arange(2,device=dev)).view(1,2,1,-1)
        f=f+sy+s.kind.weight[1]+s.pos_f(torch.arange(Lf,device=dev))
        c=c+sy+s.kind.weight[0]+s.pos_c(torch.arange(Lc,device=dev))
        x=torch.cat([c.transpose(1,2).reshape(B,2*Lc,-1),f.transpose(1,2).reshape(B,2*Lf,-1)],1)  # interleave BTC/ETH in time
        for b in s.blocks:x=b(x)
        x=s.norm(x);fx=x[:,2*Lc:].view(B,Lf,2,-1)
        pre=s.pre(fx)                                       # [B,Lf,2,6]
        y=s.head(torch.cat([fx[:,-1,0],fx[:,-1,1]],-1)).view(B,2,s.n_h)
        return pre,y
