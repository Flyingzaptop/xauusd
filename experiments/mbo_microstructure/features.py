import glob,numpy as np
TICK=0.1
def build(gd):
    Z=[np.load(f) for f in sorted(glob.glob(gd+'/chunk_*.npz'))]
    ts=np.concatenate([z['ts'] for z in Z]);px=np.concatenate([z['px'] for z in Z]);F=np.concatenate([z['f'] for z in Z]);del Z
    N=len(ts);bid=px[:,0];ask=px[:,1];mid=(bid+ask)/2;sp=(ask-bid)/TICK
    hour=((ts//10**9)%86400)/3600.0
    sess=~((hour>=22)&(hour<23))&np.isfinite(mid)&(sp>0)&(sp<=100)
    day=ts//(86400*10**9)
    cs=lambda x:np.concatenate([[0.],np.cumsum(x.astype(np.float64))])
    def win(c,w):
        o=np.zeros(N);o[w:]=c[w:N]-c[:N-w];return o
    bsz1,asz1=F[:,0].astype(np.float64),F[:,5].astype(np.float64);b5=F[:,:5].sum(1).astype(np.float64);a5=F[:,5:10].sum(1).astype(np.float64)
    micro=(bid*asz1+ask*bsz1)/(bsz1+asz1+1e-9)
    feat={}
    def add(k,v):feat[k]=v.astype(np.float32)
    for k,v in {'spread':sp,'imb1':(bsz1-asz1)/(bsz1+asz1+1e-9),'imb5':(b5-a5)/(b5+a5+1e-9),'micro_mid':(micro-mid)/TICK,'ldepth1':np.log1p(bsz1+asz1),'ldepth5':np.log1p(b5+a5)}.items():add(k,v)
    ofi=cs(F[:,10]);bv=cs(F[:,11]);sv=cs(F[:,12]);nt=cs(F[:,13]);adb=cs(F[:,14]);ada=cs(F[:,15]);cxb=cs(F[:,16]);cxa=cs(F[:,17])
    for w in [1,10,50,300]:
        add(f'ofi{w}',win(ofi,w));add(f'tv{w}',win(bv,w)-win(sv,w));add(f'nt{w}',win(nt,w))
        add(f'addimb{w}',win(adb,w)-win(ada,w));add(f'cxlimb{w}',win(cxb,w)-win(cxa,w))
        m=np.full(N,np.nan);m[w:]=(mid[w:]-mid[:-w])/TICK;add(f'mom{w}',m)
    names=list(feat);X=np.empty((N,len(names)),np.float32)
    for j,k in enumerate(names):X[:,j]=feat[k]
    return dict(ts=ts,bid=bid,ask=ask,mid=mid,sp=sp,sess=sess,day=day,X=X,names=names)
