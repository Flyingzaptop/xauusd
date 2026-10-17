"""Fair teacher benchmark (v5).
Fixes found in the v4 exact-meta gate:
  * inner budget/data large enough that money reward actually learns the task;
  * reward scale removed: every reward (money or teacher) is divided by its batch RMS
    over OPEN actions, and the inner LR is tuned once for money and reused for teacher;
  * meta-training uses a fresh student init and fresh episodes every cycle;
  * evaluation on fresh seeds never used for training/selection, paired statistics.
Learner: exact expected-return SGD over the finite fixed-exit SMDP (as v4 meta_gradient).
"""
import sys,json,time,copy,argparse
from pathlib import Path
import numpy as np,torch
from torch.func import functional_call
from torch.nn.attention import sdpa_kernel,SDPBackend
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sequential_teacher_v4'))
from model import Actor,Teacher
from synthetic import make_episode,future_view,FEATURES,CONTEXT
H=4;N=12;COST=.05

def episode(task,seed):
    if task.startswith('noisy_'):
        base=task[6:];e=make_episode(base,seed,length=N);rng=np.random.default_rng(10**7+seed)
        # same causal signal and drift; much larger unpredictable noise on the price path
        ch=np.diff(e.opens)+rng.normal(0,1.0,N);e.opens[:]=e.opens[0]+np.r_[0,np.cumsum(ch)];return e
    return make_episode(task,seed,length=N)

def batch(task,seed,n):
    eps=[episode(task,seed*1000+j) for j in range(n)]
    x=torch.tensor(np.stack([e.observations[:-1] for e in eps]))
    f=torch.tensor(np.stack([[future_view(e,i,H) for i in range(N)] for e in eps]))
    p=torch.tensor(np.stack([e.opens for e in eps]),dtype=torch.float32)
    end=torch.minimum(torch.arange(N)+H,torch.tensor(N));d=p[:,end]-p[:,:-1]
    return x,f,torch.stack([torch.zeros_like(d),d-COST,-d-COST],-1)

def expected_return(probs,r):
    b,n,_=probs.shape;v=[None]*(n+1);v[n]=torch.zeros(b)
    for i in range(n-1,-1,-1):
        j=min(i+H,n);v[i]=(probs[:,i]*torch.stack([v[i+1],r[:,i,1]+v[j],r[:,i,2]+v[j]],-1)).sum(-1)
    return v[0]
def oracle(r):
    b,n,_=r.shape;v=[None]*(n+1);v[n]=torch.zeros(b)
    for i in range(n-1,-1,-1):
        j=min(i+H,n);v[i]=torch.stack([v[i+1],r[:,i,1]+v[j],r[:,i,2]+v[j]],-1).max(-1).values
    return v[0]
def probs(actor,params,x):
    s=x.shape;d,_=functional_call(actor,params,(x.reshape(-1,*s[-2:]),));return d.probs.reshape(s[0],s[1],3)
def rms_norm(r):
    return r/(r[...,1:].pow(2).mean().sqrt()+1e-6)

def teacher_reward(teacher,x,f,mode):
    b,n=x.shape[:2];ff=f
    if mode=='zero_future':ff=torch.zeros_like(f)
    s=teacher(x.reshape(-1,*x.shape[-2:]),ff.reshape(-1,*ff.shape[-2:])).mean.reshape(b,n,2)
    r=torch.cat([torch.zeros(b,n,1),2*s-1],-1)
    if mode=='shuffled':r=r[torch.randperm(b,generator=torch.Generator().manual_seed(7))]
    return r

def adapt(task,teacher,seed,mode,K,lr,n_inner,n_eval=64,track=False):
    """Fresh student (init=seed) trained K exact steps on fresh inner episodes; returns
    expected money on fresh held-out episodes and the oracle on those episodes."""
    with torch.random.fork_rng():
        torch.manual_seed(seed);actor=Actor(FEATURES,width=16,depth=1)
    params=dict(actor.named_parameters());x,f,m=batch(task,1+seed,n_inner)
    if mode=='money':r=m
    elif mode=='constant':r=torch.tensor([0.,.5,.5]).expand(len(x),N,3)+0*m
    else:r=teacher_reward(teacher,x,f,mode)
    r=rms_norm(r)
    if not track:r=r.detach()
    for _ in range(K):
        loss=-expected_return(probs(actor,params,x),r).mean()
        g=torch.autograd.grad(loss,tuple(params.values()),create_graph=track,allow_unused=True)
        params={k:v if gg is None else v-lr*gg for (k,v),gg in zip(params.items(),g)}
    vx,vf,vm=batch(task,5*10**6+seed,n_eval)
    return expected_return(probs(actor,params,vx),vm).mean(),float(oracle(vm).mean())

def paired(a,b):
    d=np.asarray(a)-np.asarray(b);rng=np.random.default_rng(0)
    bs=[rng.choice(d,len(d)).mean() for _ in range(4000)]
    return {'mean_diff':float(d.mean()),'ci95':[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))],'wins':int((d>0).sum()),'n':len(d)}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--K',type=int,default=10)
    ap.add_argument('--n-inner',type=int,default=32);ap.add_argument('--cycles',type=int,default=150);ap.add_argument('--n-test',type=int,default=16);ap.add_argument('--out',default=None)
    a=ap.parse_args();torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False);torch.manual_seed(20261017)
    out=Path(a.out or Path(__file__).resolve().parents[2]/'reports/teacher_bench_v5'/a.task);out.mkdir(parents=True,exist_ok=True)
    T=a.task;K=a.K;NI=a.n_inner;t0=time.time();rep={'task':T,'K':K,'n_inner':NI,'H':H,'N':N,'cost':COST}
    with sdpa_kernel(SDPBackend.MATH):
        # 1. tune inner LR for money on calibration seeds 10_000..10_007 (rms-normalized reward)
        grid=[.03,.1,.3,1.,3.];cal={}
        for lr in grid:cal[lr]=float(np.mean([float(adapt(T,None,s,'money',K,lr,NI)[0]) for s in range(10000,10008)]))
        lr=max(cal,key=cal.get);rep['lr_calibration']=cal;rep['lr']=lr;print('lr',cal,'->',lr,flush=True)
        # 2. meta-train teacher: each cycle a fresh student seed and fresh episodes
        teacher=Teacher(FEATURES,width=16,depth=1,future_steps=H);initial=copy.deepcopy(teacher)
        opt=torch.optim.Adam(teacher.parameters(),lr=1e-3);hist=[];best=-1e9;best_state=None
        for c in range(a.cycles):
            seeds=[20000+2*c,20001+2*c]
            loss=-torch.stack([adapt(T,teacher,s,'learned',K,lr,NI,track=True)[0] for s in seeds]).mean()
            opt.zero_grad();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.));opt.step()
            row={'cycle':c+1,'outer':float(-loss),'grad_norm':gn}
            if (c+1)%15==0:
                sel=float(np.mean([float(adapt(T,teacher,s,'learned',K,lr,NI)[0]) for s in range(30000,30006)]));row['selection']=sel
                if sel>best:best=sel;best_state=copy.deepcopy(teacher.state_dict())
                print(json.dumps(row),flush=True)
            hist.append(row)
        if best_state is not None:teacher.load_state_dict(best_state)
        # 3. fresh test seeds, paired against controls
        test=range(900000,900000+a.n_test);res={};orc=[]
        for name,t,mode in [('learned',teacher,'learned'),('learned_zero_future',teacher,'zero_future'),('initial',initial,'learned'),
                            ('constant',teacher,'constant'),('shuffled',teacher,'shuffled'),('money',teacher,'money')]:
            vals=[adapt(T,t,s,mode,K,lr,NI) for s in test];res[name]=[float(v[0]) for v in vals];orc=[v[1] for v in vals]
        rep['oracle_mean']=float(np.mean(orc));rep['test_means']={k:float(np.mean(v)) for k,v in res.items()}
        rep['test_ratio_to_oracle']={k:float(np.mean(v)/np.mean(orc)) for k,v in res.items()}
        rep['learned_minus']={k:paired(res['learned'],v) for k,v in res.items() if k!='learned'}
        rep['learned_beats_money']=rep['learned_minus']['money']['ci95'][0]>0
        rep['results']=res;rep['elapsed_s']=time.time()-t0;rep['selection_best']=best
    torch.save({'model':teacher.state_dict(),'width':16,'depth':1,'future_steps':H,'features':FEATURES},out/'teacher.pt')
    (out/'history.json').write_text(json.dumps(hist,indent=1));(out/'result.json').write_text(json.dumps(rep,indent=1))
    print(json.dumps({k:rep[k] for k in ['task','lr','oracle_mean','test_means','learned_beats_money','elapsed_s']},indent=1))
