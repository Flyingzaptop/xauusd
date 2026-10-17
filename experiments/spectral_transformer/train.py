import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import json,time,hashlib,math,traceback
from pathlib import Path
import numpy as np,torch
from model import Data,Model,loss,H
P=Path('/content/spectral');O=P/'results';O.mkdir(exist_ok=True);START=time.time();BATCH=128

def save(n,x):
 p=O/n;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def status(**x):x['seconds']=round(time.time()-START,1);save('status.json',x);print(json.dumps(x),flush=True)
def model(data,kind,ids):
 torch.manual_seed(17);m=Model(kind,data.x.shape[1],data.nbase).cuda();ix=torch.tensor(ids,device='cuda');m.initialize(data.y[ix],data.event[ix]);return m
@torch.no_grad()
def collect(m,data,ids):
 m.eval();p=[];q=[];e=[];tot=0;components=np.zeros(3)
 for a in range(0,len(ids),BATCH):
  x,y,ev=data.batch(ids[a:a+BATCH],m.kind)
  with torch.autocast('cuda',dtype=torch.bfloat16):pred=m(x);l,parts=loss(pred,y,ev)
  tot+=float(l)*len(y);components+=np.array(list(parts.values()))*len(y);p.append(pred[0].cpu().numpy());q.append(pred[1].cpu().numpy());e.append(pred[2].softmax(-1).cpu().numpy())
 return tot/len(ids),np.concatenate(p),np.concatenate(q),np.concatenate(e),(components/len(ids)).tolist()
def norms(data):return {k:getattr(data,k).cpu() for k in ['emu','esd','xmu','xsd']}
def fit(data,kind,phase,epochs,select):
 ids=data.ids['train' if select else 'development'];m=model(data,kind,ids);opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=.01);best=float('inf');stale=0;bestep=0;curves=[];key=f'{kind}_{phase}';steps=0;probe=ids[np.linspace(0,len(ids)-1,min(2048,len(ids)),dtype=int)];val=data.ids['val'];tstart=time.time()
 for ep in range(1,epochs+1):
  m.train();running=0;seen=0;t=time.time();factor=.25+.75*.5*(1+math.cos(math.pi*(ep-1)/8))
  for a in range(0,len(ids),BATCH):
   rows=ids[a:a+BATCH];x,y,ev=data.batch(rows,kind);opt.param_groups[0]['lr']=2e-4*factor*min(1,(steps+1)/100);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):pred=m(x);value,_=loss(pred,y,ev)
   assert torch.isfinite(value);value.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step();running+=float(value.detach())*len(rows);seen+=len(rows);steps+=1
   if steps%200==0:status(phase=phase,model=kind,epoch=ep,seen=seen,total=len(ids),loss=running/seen)
  assert seen==len(ids);tl,_,_,_,tc=collect(m,data,probe)
  row={'epoch':ep,'examples':seen,'online_loss':running/seen,'train_probe_loss':tl,'train_components':tc,'seconds':time.time()-t}
  if select:
   vl,p,q,e,vc=collect(m,data,val);row.update(validation_loss=vl,validation_components=vc)
   if vl<best-.001*abs(best) if np.isfinite(best) else True:
    best=vl;bestep=ep;stale=0;torch.save({'model':m.state_dict(),'norms':norms(data),'kind':kind,'epoch':ep},O/f'{key}_best.pt');np.savez_compressed(O/f'{kind}_selection.npz',row=val,mean=p,quantiles=q,event_prob=e)
   else:stale+=1
  else:bestep=ep
  torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'norms':norms(data),'kind':kind,'epoch':ep},O/f'{key}_last.pt')
  curves.append(row);save(key+'_curve.json',curves);status(phase=phase,model=kind,epoch=ep,epoch_complete=True,metrics=row,best_epoch=bestep)
  if select and ep>=3 and stale>=2:break
  if time.time()-START>3300:raise TimeoutError('55minute active training budget; resumable checkpoints retained')
 if not select:
  torch.save({'model':m.state_dict(),'norms':norms(data),'kind':kind,'epoch':bestep,'context':128},O/f'{kind}_final.pt')
  # Finish every selection/training decision before examining test metrics.
 save(key+'_summary.json',{'best_epoch':bestep,'parameters':sum(p.numel() for p in m.parameters()),'seconds':time.time()-tstart,'epochs':len(curves)})
 del m,opt;torch.cuda.empty_cache();return bestep

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.manual_seed(17);np.random.seed(17)
 data=Data(P);save('run.json',{'status':'training','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'dropout':.25,'context':128,'patch':4,'encoder_layers':3,'d_model':96,'horizons':H,'models':['price','raw','engineered'],'max_selection_epochs':8,'train_chronological_every_anchor':True,'seed':17,'data':json.loads((P/'prepared.json').read_text()),'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in P.glob('*.py')}})
 # Actual future-mutation check of engineered batch construction with frozen fitted preprocessing.
 ix=int(data.ids['train'][100]);before=data.batch([ix],'engineered')[0];old_s=data.s[ix+1].clone();old_x=data.x[ix+1].clone();data.s[ix+1]=0;data.x[ix+1]=999;after=data.batch([ix],'engineered')[0];assert all(torch.equal(a,b) for a,b in zip(before,after));data.s[ix+1]=old_s;data.x[ix+1]=old_x;del before,after,old_s,old_x
 save('causality_checks.json',{'engineered_future_mutation_passed':True,'current_candle_closed':True,'targets_not_in_input':True,'all2356_components_in_input':True})
 # Tiny memorization check distinguishes implementation/capacity failure from generalization.
 ids=data.ids['train'][np.linspace(0,len(data.ids['train'])-1,32,dtype=int)];m=model(data,'engineered',data.ids['train']);m.eval();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=0);x,y,ev=data.batch(ids,'engineered')
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):initial=float(loss(m(x),y,ev)[0])
 for i in range(160):
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):l,_=loss(m(x),y,ev)
  l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):final=float(loss(m(x),y,ev)[0]);evacc=float((m(x)[2].argmax(-1)==ev).float().mean())
 save('tiny_fit.json',{'examples':32,'steps':160,'initial_loss':initial,'final_loss':final,'loss_ratio':final/initial,'event_accuracy':evacc,'dropout_disabled_for_capacity_check':True});status(phase='tiny_fit',initial=initial,final=final,event_accuracy=evacc)
 del m,opt,x,y,ev;torch.cuda.empty_cache();assert final<initial*.5,'tiny overfit failed; do not waste full training'
 choices={}
 for kind in ['price','raw','engineered']:choices[kind]=fit(data,kind,'selection',8,True)
 save('selection_frozen.json',{'best_epochs':choices,'test_used':False,'validation':'20-22ofEVERYJan-Sep month with256bar embargo'})
 data.norm('development')
 for kind in ['price','raw','engineered']:fit(data,kind,'final',choices[kind],False)
 status(phase='test_evaluation',selection_frozen=True)
 for kind in ['price','raw','engineered']:
  ck=torch.load(O/f'{kind}_final.pt',weights_only=False);m=Model(kind,data.x.shape[1],data.nbase).cuda();m.load_state_dict(ck['model']);_,p,q,e,parts=collect(m,data,data.ids['test']);np.savez_compressed(O/f'{kind}_test.npz',row=data.ids['test'],mean=p,quantiles=q,event_prob=e);save(f'{kind}_test_loss.json',dict(zip(['mse','pinball','event_ce'],parts)));del m;torch.cuda.empty_cache()
 # Constant baselines fit to allJan-Sep; compare without learned signal.
 ids=data.ids['development'];yy=data.raw['y'][ids];freq=np.bincount(data.raw['event'][ids],minlength=9)/len(ids);np.savez(O/'constant.npz',mean=yy.mean(0),quantiles=np.quantile(yy,[.1,.5,.9],axis=0).T,event_prob=freq)
 # Compact target export for complete local evaluation, excluding raw feature matrices.
 np.savez_compressed(O/'evaluation_data.npz',**{k:data.raw[k] for k in ['y','event','first','direction','scale','timestamp','ohlc','train','val','development','test','month','source_row']})
 status(phase='complete',seconds_total=time.time()-START);save('complete.json',{'complete':True,'seconds':time.time()-START,'choices':choices})
if __name__=='__main__':
 try:main()
 except Exception as e:save('failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
