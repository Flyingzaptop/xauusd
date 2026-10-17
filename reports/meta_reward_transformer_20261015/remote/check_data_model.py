import tempfile,json,torch,numpy as np
from pathlib import Path
from data import Data
from model import MarketTransformer
torch.set_num_threads(1);rng=np.random.default_rng(5);n=2200;I=2;F=48
with tempfile.TemporaryDirectory() as tmp:
 r=Path(tmp);t=1577836800+np.arange(n)*60;x=rng.normal(size=(I,n,F)).astype('float16');mm=np.memmap(r/'features.f16',mode='w+',dtype='float16',shape=x.shape);mm[:]=x;mm.flush();del mm
 c=100+np.cumsum(rng.normal(0,.1,n));o=np.r_[c[0],c[:-1]];np.save(r/'anchors.npy',t);np.save(r/'gold_ohlcv.npy',np.c_[o,np.maximum(o,c)+.01,np.minimum(o,c)-.01,c,np.ones(n)]);np.save(r/'clock.npy',np.zeros((n,8),dtype='float16'));(r/'prepared.json').write_text(json.dumps({'shape':x.shape,'instruments':I}))
 d=Data(r,device='cpu');ids=np.array([400,500]);b,clock,f=d.batch(ids,teacher=True);assert b.shape==(2,I,120,96);assert f.shape==(2,120,8)
 a=MarketTransformer(d=24,depth=1,heads=4,instruments=I,dropout=.25).eval();teacher=MarketTransformer(teacher=True,d=24,depth=1,heads=4,instruments=I).eval()
 with torch.no_grad():pred=a(b,clock)[0].probs.clone();scores=teacher(b,clock,f).mean
 assert ((scores>0)&(scores<1)).all()
 mm=np.memmap(r/'features.f16',mode='r+',dtype='float16',shape=x.shape);mm[:,500:]*=3;mm.flush()
 altered=np.load(r/'gold_ohlcv.npy');altered[500:,:4]+=10;np.save(r/'gold_ohlcv.npy',altered)
 d2=Data(r,device='cpu');b2,c2,f2=d2.batch(np.array([400]),teacher=True)
 torch.testing.assert_close(b[:1],b2)
 with torch.no_grad():torch.testing.assert_close(pred[:1],a(b2,c2)[0].probs)
 # Independently test the recorded time offset: entry at t_i uses [i-120,i).
 np.testing.assert_allclose(b[0,0,:,0].numpy(),np.arcsinh(x[0,280:400,0].astype('float32')),rtol=1e-5)
 # Dropout replay must produce the same probability law for outer policy gradient.
 teacher.train();state=torch.get_rng_state();p=teacher(b,clock,f);v=p.sample();lp=p.log_prob(v).detach();torch.set_rng_state(state);p2=teacher(b,clock,f);torch.testing.assert_close(lp,p2.log_prob(v).detach())
 print(json.dumps({'passed':True,'checks':['continuous bounded teacher scores','actor shape','past-only model inputs and outputs','exact entry offset','dropout replay likelihood']}))
