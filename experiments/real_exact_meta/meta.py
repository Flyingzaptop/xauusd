"""Exact meta-gradient learned reward on real XAUUSD sessions. See PROTOCOL.md."""
import sys,json,time,copy,argparse
from pathlib import Path
import numpy as np,torch
from torch.func import functional_call
from torch.nn.attention import sdpa_kernel,SDPBackend
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sequential_teacher_v4'))
from model import Actor,Teacher
ap=argparse.ArgumentParser()
ap.add_argument('--data',default='/home/user/data_prep/xau_sessions.npz');ap.add_argument('--out',required=True)
ap.add_argument('--K',type=int,default=10);ap.add_argument('--n-inner',type=int,default=16);ap.add_argument('--n-outer',type=int,default=32)
ap.add_argument('--cycles',type=int,default=400);ap.add_argument('--n-test',type=int,default=48);ap.add_argument('--seed',type=int,default=1)
ap.add_argument('--width',type=int,default=32);ap.add_argument('--run-test',action='store_true');ap.add_argument('--threads',type=int,default=4)
a=ap.parse_args();torch.set_num_threads(a.threads);torch.backends.mha.set_fastpath_enabled(False);torch.manual_seed(a.seed)
Z=np.load(a.data);X=torch.tensor(Z['x']);M=torch.tensor(Z['money']);FU=torch.tensor(Z['future']);FU[...,0]/=30
MK=torch.tensor(Z['mask']);DAY=Z['day'];N=X.shape[1];H=FU.shape[2];F=X.shape[-1]
def split(lo,hi):return np.where((DAY>=lo)&(DAY<=hi))[0]
TRAIN=split('2017-01-01','2022-12-31');SEL=split('2023-01-01','2023-12-31');TEST=split('2025-07-01','2026-09-04')
rng=np.random.default_rng(a.seed+999)
# train days are split once into inner-pool / outer-pool so outer evaluation never reuses inner days of the same cycle
perm=rng.permutation(TRAIN);INNER_POOL=perm[:len(perm)//2];OUTER_POOL=perm[len(perm)//2:]
WAIT=torch.tensor([1.,0,0])
def expected_return(p,r,mask):
    p=torch.where(mask[...,None],p,WAIT);b=p.shape[0];v=[None]*(N+1);v[N]=torch.zeros(b)
    for i in range(N-1,-1,-1):
        j=min(i+H,N);v[i]=(p[:,i]*torch.stack([v[i+1],r[:,i,1]+v[j],r[:,i,2]+v[j]],-1)).sum(-1)
    return v[0]
def probs(actor,params,x):
    s=x.shape;d,_=functional_call(actor,params,(x.reshape(-1,*s[-2:]),));return d.probs.reshape(s[0],s[1],3)
def rms(r,mask):return r/(r[...,1:][mask].pow(2).mean().sqrt()+1e-6)
# supervised critic: ridge regression of money on flattened student features, all inner-pool train days
def fit_critic():
    xs=X[INNER_POOL].reshape(-1,X.shape[2]*F);ys=M[INNER_POOL].reshape(-1,3)[:,1:];k=MK[INNER_POOL].reshape(-1)
    xs=torch.cat([xs[k],torch.ones(int(k.sum()),1)],1).double();ys=ys[k].double();lam=1e3
    W=torch.linalg.solve(xs.T@xs+lam*torch.eye(xs.shape[1],dtype=torch.double),xs.T@ys);return W.float()
CRITIC=fit_critic()
def reward(mode,teacher,idx):
    x,m,f,k=X[idx],M[idx],FU[idx],MK[idx];b=len(idx)
    if mode in('money','money_big'):r=m
    elif mode=='constant':r=torch.tensor([0.,.5,.5]).expand(b,N,3).clone()
    elif mode=='critic':
        xx=torch.cat([x.reshape(b,N,-1),torch.ones(b,N,1)],-1);r=torch.cat([torch.zeros(b,N,1),xx@CRITIC],-1)
    else:
        ff=torch.zeros_like(f) if mode=='zero_future' else f
        s=teacher(x.reshape(-1,*x.shape[-2:]),ff.reshape(-1,H,2)).mean.reshape(b,N,2);r=torch.cat([torch.zeros(b,N,1),2*s-1],-1)
        if mode=='shuffled':r=r[torch.randperm(b,generator=torch.Generator().manual_seed(7))]
    return rms(r,k)
def evaluate(actor,params,idx,chunk=64):
    tot=0.
    with torch.no_grad():
        for c in range(0,len(idx),chunk):
            j=idx[c:c+chunk];tot+=float(expected_return(probs(actor,params,X[j]),M[j],MK[j]).sum())
    return tot/len(idx)  # expected bp per day
def adapt(mode,teacher,seed,lr,eval_idx,track=False):
    g=np.random.default_rng(seed);n=a.n_inner*(8 if mode=='money_big' else 1)
    inner=g.choice(INNER_POOL,n,replace=False)
    with torch.random.fork_rng():
        torch.manual_seed(seed);actor=Actor(F,width=a.width,depth=1)
    params=dict(actor.named_parameters());r=reward(mode,teacher,inner)
    if not track:r=r.detach()
    x,k=X[inner],MK[inner]
    for _ in range(a.K):
        loss=-expected_return(probs(actor,params,x),r,k).mean()
        gr=torch.autograd.grad(loss,tuple(params.values()),create_graph=track,allow_unused=True)
        params={kk:v if gg is None else v-lr*gg for (kk,v),gg in zip(params.items(),gr)}
    if track:
        o=eval_idx;return expected_return(probs(actor,params,X[o]),M[o],MK[o]).mean()
    return evaluate(actor,{k_:v.detach() for k_,v in params.items()},eval_idx)
def fixed(policy,idx):
    p=torch.zeros(len(idx),N,3);p[...,{'flat':0,'always_long':1,'always_short':2}[policy]]=1
    return float(expected_return(p,M[idx],MK[idx]).mean())
def paired(x,y):
    d=np.asarray(x)-np.asarray(y);g=np.random.default_rng(0);bs=[g.choice(d,len(d)).mean() for _ in range(4000)]
    return {'mean_diff':float(d.mean()),'ci95':[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))],'wins':int((d>0).sum()),'n':len(d)}
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);t0=time.time();rep={'args':vars(a),'days':{'train':len(TRAIN),'sel':len(SEL),'test':len(TEST)}}
CAL_EVAL=OUTER_POOL[:200]
with sdpa_kernel(SDPBackend.MATH):
    cal={lr:float(np.mean([adapt('money',None,s,lr,CAL_EVAL) for s in range(10000,10006)])) for lr in [.01,.03,.1,.3,1.]}
    lr=max(cal,key=cal.get);rep['lr_calibration']=cal;rep['lr']=lr;print('lr',cal,'->',lr,flush=True)
    teacher=Teacher(F,width=a.width,depth=1,future_steps=H);initial=copy.deepcopy(teacher)
    opt=torch.optim.Adam(teacher.parameters(),lr=1e-3);hist=[];best=-1e9;best_state=copy.deepcopy(teacher.state_dict());best_cycle=0
    for c in range(a.cycles):
        g=np.random.default_rng(50000+c);o=g.choice(OUTER_POOL,a.n_outer,replace=False)
        loss=-adapt('learned',teacher,20000+c,lr,o,track=True)
        opt.zero_grad();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.));opt.step()
        row={'cycle':c+1,'outer_bp_day':float(-loss),'grad_norm':gn,'t':round(time.time()-t0)}
        if (c+1)%25==0:
            sel=float(np.mean([adapt('learned',teacher,s,lr,SEL) for s in range(30000,30008)]));row['selection_bp_day']=sel
            if sel>best:best=sel;best_state=copy.deepcopy(teacher.state_dict());best_cycle=c+1
            print(json.dumps(row),flush=True)
        hist.append(row)
    teacher.load_state_dict(best_state);rep['best_cycle']=best_cycle;rep['selection_best']=best
    torch.save({'model':teacher.state_dict(),'width':a.width,'depth':1,'features':F,'future_steps':H,'feature_names':list(Z['names'])},out/'teacher.pt')
    # selection-period comparison of all arms (allowed: selection data), test only with --run-test
    def arms(idx,seeds):
        res={}
        for name,t,mode in [('learned',teacher,'learned'),('learned_zero_future',teacher,'zero_future'),('initial',initial,'learned'),('constant',None,'constant'),
                            ('shuffled',teacher,'shuffled'),('money',None,'money'),('money_big',None,'money_big'),('critic',None,'critic')]:
            res[name]=[adapt(mode,t,s,lr,idx) for s in seeds];print(name,np.mean(res[name]),flush=True)
        r={'means':{k:float(np.mean(v)) for k,v in res.items()},'fixed':{p:fixed(p,idx) for p in ['flat','always_long','always_short']},
           'learned_minus':{k:paired(res['learned'],v) for k,v in res.items() if k!='learned'},'per_student':res}
        return r
    rep['selection_eval']=arms(SEL,range(40000,40000+a.n_test))
    if a.run_test:rep['test_eval']=arms(TEST,range(900000,900000+a.n_test))
    rep['elapsed_s']=time.time()-t0
(out/'history.json').write_text(json.dumps(hist,indent=1));(out/'result.json').write_text(json.dumps(rep,indent=1))
print(json.dumps({k:rep[k] for k in ['lr','best_cycle']}),json.dumps(rep['selection_eval']['means']),json.dumps(rep['selection_eval']['fixed']))
