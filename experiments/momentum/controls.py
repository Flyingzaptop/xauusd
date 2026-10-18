"""M6 controls for the candidate (see PROTOCOL.md). usage: controls.py OUT.json"""
import sys,json,numpy as np,pandas as pd
src=open('candidate.py').read();pre_src=src.split("sl={};bh={}")[0];exec(pre_src.replace("O=sys.argv[1]","O='/tmp'"))
P={};F={}
for s in ('BTCUSDT','ETHUSDT'):
    p=pd.concat([spot(s),perp(s)]);p=p[~p.index.duplicated(keep='last')].sort_index();P[s]=p
    f=pd.read_parquet(f'/home/user/crypto/{s}_funding.parquet');f.index=pd.to_datetime(f.calc_time,unit='ms');F[s]=f.last_funding_rate.astype(float).resample('1D').sum().reindex(p.index).fillna(0)
P=pd.DataFrame(P);F=pd.DataFrame(F).fillna(0);R=np.log(P).diff();W=(0.5/(R.rolling(20).std()*np.sqrt(365))).clip(upper=2)
def run(S,cap=1.0):
    S=S*W;raw=(S.shift(1)*R).dropna(how='all');sc=1/raw.rolling(60,min_periods=30).std().shift(1)
    pre=(S.shift(1)*sc/2);pr=(pre*R).sum(1,min_count=1);ps=(0.20/np.sqrt(365))/pr.rolling(60,min_periods=30).std().shift(1)
    N=pre.mul(ps,axis=0);g=N.abs().sum(1);N=N.div((g/cap).clip(lower=1),axis=0)
    x=((N*R).sum(1,min_count=1)-(N*F).sum(1)-N.diff().abs().sum(1)*10e-4);return x[x.index>='2017-12-15'].dropna()
sh=lambda x:float(x.mean()/x.std()*np.sqrt(365))
SIG=sum(np.sign(P-P.rolling(n).mean()) for n in (20,50,100))/3
cand=run(SIG);longonly=run(SIG*0+1)
rng=np.random.default_rng(0);shifted=[]
for k in rng.integers(180,2000,50):
    S=SIG.copy();S[:]=np.roll(SIG.values,k,axis=0);shifted.append(sh(run(S)))
x=cand.values;n=len(x);B=[]
for _ in range(5000):
    idx=[];i=rng.integers(n)
    while len(idx)<n:
        idx.append(i);i=i+1 if rng.random()>1/30 else rng.integers(n);i%=n
    y=x[idx];B.append(y.mean()/y.std()*np.sqrt(365))
out={'candidate_sharpe':sh(cand),'longonly_voltarget_sharpe':sh(longonly),'shifted_signal_sharpe_mean':float(np.mean(shifted)),'shifted_p95':float(np.quantile(shifted,.95)),
     'share_shifted_ge_candidate':float(np.mean(np.array(shifted)>=sh(cand))),'bootstrap_sharpe_ci90':[float(np.quantile(B,.05)),float(np.quantile(B,.95))],'bootstrap_P_sharpe_le_0':float(np.mean(np.array(B)<=0))}
print(json.dumps(out,indent=1));json.dump(out,open(sys.argv[1],'w'),indent=1)
