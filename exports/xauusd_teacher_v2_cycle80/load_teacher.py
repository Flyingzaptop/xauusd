"""Reference teacher inference. Scores need future data; not live probabilities."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from model import MarketTransformer

ROOT=Path(__file__).resolve().parent

def verify_package():
 manifest=json.loads((ROOT/'MANIFEST.json').read_text())
 for name,info in manifest['files'].items():
  p=ROOT/name
  if not p.exists() or p.stat().st_size!=info['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=info['sha256']:
   raise RuntimeError('Package integrity failure: '+name)
 return manifest

class TeacherScorer:
 def __init__(self,device='cpu',verify=True):
  if verify:verify_package()
  self.device=device;checkpoint=torch.load(ROOT/'best_teacher.pt',map_location='cpu',weights_only=True);cfg=checkpoint['config']
  self.model=MarketTransformer(teacher=True,d=cfg['hidden'],depth=cfg['depth_temporal'],heads=6,instruments=222,features=48,dropout=cfg['dropout'],patch=cfg['patch']).to(device).eval()
  self.model.load_state_dict(checkpoint['model'],strict=True);self.cycle=checkpoint['cycle']
 @torch.inference_mode()
 def score(self,past,clock,future):
  x=torch.as_tensor(past,dtype=torch.float32,device=self.device);c=torch.as_tensor(clock,dtype=torch.float32,device=self.device);f=torch.as_tensor(future,dtype=torch.float32,device=self.device)
  if x.ndim!=4 or tuple(x.shape[1:])!=(222,120,96):raise ValueError('past must be B,222,120,96 in exact saved symbol/feature order')
  if c.shape!=(x.shape[0],8) or f.shape!=(x.shape[0],120,8):raise ValueError('clock B,8; future B,120,8 required')
  if not all(torch.isfinite(v).all() for v in [x,c,f]):raise ValueError('Nonfinite model input: use supplied masking/preprocessing')
  return self.model(x,c,f).mean.cpu().numpy() # columns LONG, SHORT, not calibrated probabilities

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');ap.add_argument('--prepared-root');ap.add_argument('--anchor-index',type=int);ap.add_argument('--device',default='cpu');args=ap.parse_args();torch.set_num_threads(4)
 scorer=TeacherScorer(args.device)
 if args.self_test:
  z=np.load(ROOT/'inference_fixture.npz');pred=scorer.score(z['past'],z['clock'],z['future']);np.testing.assert_allclose(pred,z['expected'],atol=2e-5,rtol=2e-5);print(json.dumps({'passed':True,'cycle':scorer.cycle,'LONG_SHORT':pred.tolist()}))
 elif args.prepared_root is not None and args.anchor_index is not None:
  from data import Data
  d=Data(args.prepared_root,device=args.device,cache=False);schema=json.loads((ROOT/'prepared.json').read_text())
  if d.meta['symbols']!=schema['symbols'] or d.meta['names']!=schema['names']:raise ValueError('Prepared schema/order mismatch')
  x,c,f=d.batch(np.array([args.anchor_index]),teacher=True);print(scorer.score(x,c,f).tolist())
 else:ap.error('Use --self-test or --prepared-root ROOT --anchor-index N')
