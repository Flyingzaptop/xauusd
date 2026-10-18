"""Today's target positions for the trend candidate (BTC+ETH, SMA 20/50/100 ensemble, 20% vol target, gross <= 1x).
Uses only completed daily candles (00:00 UTC) from Binance public market data. Prints fraction of equity per asset
(+ long / - short) and appends to a CSV log for forward tracking. usage: signal.py [LOG.csv]"""
import sys,json,urllib.request,numpy as np,pandas as pd
def kl(sym,n=400):
    u=f'https://data-api.binance.vision/api/v3/klines?symbol={sym}&interval=1d&limit={n}'
    k=json.load(urllib.request.urlopen(u,timeout=30));now=pd.Timestamp.now(tz='UTC').floor('D').tz_localize(None)
    s=pd.Series([float(x[4]) for x in k],index=pd.to_datetime([x[0] for x in k],unit='ms'));return s[s.index<now]   # completed days only
def targets(P):
    R=np.log(P).diff();W=(0.5/(R.rolling(20).std()*np.sqrt(365))).clip(upper=2)
    S=(sum(np.sign(P-P.rolling(n).mean()) for n in (20,50,100))/3)*W
    raw=S.shift(1)*R;sc=1/raw.rolling(60,min_periods=30).std()          # through today's close -> position for the next day
    pre=S*sc/2;pr=(pre.shift(1)*R).sum(1,min_count=1);ps=(0.20/np.sqrt(365))/pr.rolling(60,min_periods=30).std()
    N=pre.mul(ps,axis=0);g=N.abs().sum(1);return N.div(g.clip(lower=1),axis=0)
if __name__=='__main__':
    P=pd.DataFrame({s:kl(s) for s in ('BTCUSDT','ETHUSDT')}).dropna();N=targets(P);last=N.iloc[-1]
    print(f'as of close {P.index[-1].date()} (hold until next 00:00 UTC):',{k:round(float(v),3) for k,v in last.items()},'gross',round(float(last.abs().sum()),3))
    if len(sys.argv)>1:
        row=pd.DataFrame([{'date':P.index[-1].date(),**{f'{k}_pos':float(v) for k,v in last.items()},**{f'{k}_close':float(P[k].iloc[-1]) for k in P}}])
        import os;row.to_csv(sys.argv[1],mode='a',header=not os.path.exists(sys.argv[1]),index=False)
