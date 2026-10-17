"""Outer score-function RL reward teacher; inner contextual-bandit actor training.
No differentiation through the student. Genuine external feedback from executed
student policies on disjoint dates. See RUN_PROTOCOL.json for actual coverage.
"""
from pathlib import Path
import os,time,json,hashlib,random,traceback,copy,math
import numpy as np,torch,pandas as pd
from model import MarketTransformer
from data import Data
R=Path('/content/meta_reward');O=R/'results';O.mkdir(exist_ok=True);T=time.monotonic();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
CFG={'seed':177,'batch':256,'inner_steps':32,'students':3,'student_lr':3e-4,'entropy_coefficient':.01,'warmup_updates':50,'teacher_lr':1e-4,'dropout':.25,'patch':10,'depth_temporal':6,'depth_market':6,'hidden':192,'cost_per_ounce':.25,'primary_wall_seconds':14400,'comparison_wall_seconds':5400,'meta_days_per_update':2,'evaluation_batch':256,'final_day_cap':None}
def write(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));os.replace(tmp,p)
def status(**kw):kw.update(seconds=round(time.monotonic()-T,1),allocated_GB=round(torch.cuda.memory_allocated()/1e9,2));write(R/'training_status.json',kw);print(json.dumps(kw),flush=True)
def save(name,**kw):
 p=O/(name+'.pt');torch.save(kw,p.with_suffix('.tmp'));os.replace(p.with_suffix('.tmp'),p)
def seeds(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def new_student(s):
 seeds(s);m=MarketTransformer(patch=CFG['patch']).cuda();opt=torch.optim.AdamW(m.parameters(),lr=CFG['student_lr'],weight_decay=.01);return m,opt
@torch.no_grad()
def evaluate(model,ids,seed=0,full=False):
 model.eval();acts=[];rng=np.random.default_rng(seed)
 for start in range(0,len(ids),CFG['evaluation_batch']):
  ix=ids[start:start+CFG['evaluation_batch']];x,c=D.batch(ix)
  with torch.autocast('cuda',dtype=torch.bfloat16):p,_=model(x,c)
  probs=p.probs.float().cpu().numpy();u=rng.random(len(ix));a=(u[:,None]>np.cumsum(probs,axis=1)).sum(1).clip(0,2);acts.append(a)
  del x,c,p
 actions=np.concatenate(acts) if acts else np.array([],dtype=int);mean,n,records=D.execute(ids,actions,CFG['cost_per_ounce']);model.train()
 result={'mean_net_dollars_per_day':mean,'trades':n,'days':int(len(np.unique(D.ts[ids]//86400))),'rows':len(ids),'seed':seed}
 if full:result['records']=records;result['actions']=actions
 return result

def student_step(student,opt,ids,rewards,actions,policy,value):
 # Action sampling and this gradient share one forward/dropout realization.
 advantage=(rewards-value.detach());norm=advantage.std(unbiased=False).clamp(min=.1);loss=-(policy.log_prob(actions)*advantage/norm).mean()+.5*(value-rewards).square().mean()-CFG['entropy_coefficient']*policy.entropy().mean()
 if not torch.isfinite(loss):raise RuntimeError('Non-finite student loss')
 updates=opt.param_groups[0].get('updates',0)+1
 for group in opt.param_groups:group['updates']=updates;group['lr']=CFG['student_lr']*min(1,updates/CFG['warmup_updates'])
 opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(student.parameters(),1.);opt.step()
 return float(loss.detach()),float(grad),float(policy.entropy().mean().detach())
try:
 seeds(CFG['seed']);status(phase='loading_dataset');D=Data();write(O/'dataset_protocol.json',{'counts':{k:len(v) for k,v in D.splits.items()},'source':D.meta,'GPU_cache_rows':D.cache_end,'GPU_cache_bytes':D.x.numel()*D.x.element_size(),'reward_scale_for_direct_control':D.reward_scale,'split_boundaries':D.secs,'config':CFG})
 # Snapshot exact sources before any training. Never depend on a later local copy.
 (O/'source').mkdir(exist_ok=True)
 for f in R.glob('*.py'):(O/'source'/f.name).write_bytes(f.read_bytes())
 write(O/'source_hashes.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (O/'source').glob('*.py')})
 write(O/'run_config.json',CFG)
 teacher=MarketTransformer(teacher=True,patch=CFG['patch']).cuda();topt=torch.optim.AdamW(teacher.parameters(),lr=CFG['teacher_lr'],weight_decay=.01);save('initial_teacher',model=teacher.state_dict(),config=CFG)
 students=[new_student(177+100*i) for i in range(CFG['students'])];teacher.train();rng=np.random.default_rng(177);order=rng.permutation(D.splits['train']);cursor=0;outer=0;baseline=0.;curves=[];best=-np.inf;best_cycle=0;counts=[0]*len(students)
 total_cycles=math.ceil(len(order)/(CFG['batch']*CFG['inner_steps']));write(O/'planned_coverage.json',{'inner_train_rows':len(order),'outer_cycles_for_one_combined_pass':total_cycles,'per_student_examples_if_equal':len(order)/3,'combined_pass_is_not_one_pass_per_student':True})
 status(phase='meta_training',cycles=total_cycles,cache_GB=D.x.numel()*2/1e9)
 start_primary=time.monotonic()
 for outer in range(total_cycles):
  if outer and time.monotonic()-start_primary>CFG['primary_wall_seconds']:break
  if outer>=96 and outer-best_cycle>=64:break
  si=outer%len(students);student,opt=students[si];meta_ids=D.eval_ids('meta',CFG['meta_days_per_update'],seed=7000+outer);before=evaluate(student,meta_ids,seed=9000+outer)
  logs=[];losses=[];entropies=[]
  for k in range(CFG['inner_steps']):
   if cursor>=len(order):break
   ids=order[cursor:cursor+CFG['batch']];cursor+=len(ids);counts[si]+=len(ids);x,c,f=D.batch(ids,teacher=True);state=torch.cuda.get_rng_state()
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):dist=teacher(x,c,f);scores=dist.sample();old_lp=dist.log_prob(scores).detach()
   logs.append({'ids':ids.copy(),'scores':scores.cpu(),'rng':state.cpu(),'old_lp':old_lp.cpu()})
   with torch.autocast('cuda',dtype=torch.bfloat16):policy,value=student(x,c)
   actions=policy.sample();selected=scores.gather(1,(actions-1).clamp(min=0)[:,None]).squeeze(-1);rewards=torch.where(actions==0,torch.zeros_like(selected),2*selected-1).detach()
   loss,grad,entropy=student_step(student,opt,ids,rewards,actions,policy,value);losses.append(loss);entropies.append(entropy)
   del x,c,f,policy,value,actions,scores,dist
  after=evaluate(student,meta_ids,seed=9000+outer);gain=after['mean_net_dollars_per_day']-before['mean_net_dollars_per_day'];advantage=gain-baseline;baseline=.9*baseline+.1*gain
  # Outer policy gradient, exact replay of sampled teacher actions and dropout.
  topt.zero_grad(set_to_none=True);replay_error=0.;normalizer=sum(len(v['ids']) for v in logs)
  for record in logs:
   x,c,f=D.batch(record['ids'],teacher=True);current_rng=torch.cuda.get_rng_state();torch.cuda.set_rng_state(record['rng'])
   with torch.autocast('cuda',dtype=torch.bfloat16):dist=teacher(x,c,f)
   torch.cuda.set_rng_state(current_rng);lp=dist.log_prob(record['scores'].cuda());replay_error=max(replay_error,float((lp.detach()-record['old_lp'].cuda()).abs().max()))
   if replay_error>.03:raise RuntimeError('Teacher policy likelihood replay mismatch: '+str(replay_error))
   tl=-lp.sum()*float(advantage)/max(normalizer,1);tl.backward();del x,c,f,dist,lp,tl
  tgrad=float(torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.));topt.step()
  row={'cycle':outer+1,'student':si,'examples':cursor,'student_examples':counts.copy(),'meta_before':before,'meta_after':after,'teacher_external_reward':gain,'teacher_advantage':advantage,'teacher_gradient_norm':tgrad,'teacher_likelihood_replay_max_error':replay_error,'student_loss_mean':float(np.mean(losses)),'student_entropy':float(np.mean(entropies)),'seconds':time.monotonic()-T}
  curves.append(row);write(O/'learning_curves.json',curves)
  if (outer+1)%8==0 or outer==0 or cursor==len(order):
   dev_ids=D.eval_ids('dev',days=12,seed=4242);dev=evaluate(student,dev_ids,seed=2424);row['development']=dev
   if dev['mean_net_dollars_per_day']>best:
    best=dev['mean_net_dollars_per_day'];best_cycle=outer+1;save('best_teacher',model=teacher.state_dict(),optimizer=topt.state_dict(),cycle=best_cycle,development=dev,config=CFG)
   save('latest_teacher',model=teacher.state_dict(),optimizer=topt.state_dict(),cycle=outer+1,config=CFG)
   save('resume_state',teacher=teacher.state_dict(),teacher_optimizer=topt.state_dict(),students=[{'model':m.state_dict(),'optimizer':o.state_dict()} for m,o in students],cursor=cursor,order=order,outer=outer+1,baseline=baseline,counts=counts,curves=curves,best=best,best_cycle=best_cycle,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_generator=rng.bit_generator.state,config=CFG)
   for j,(m,o) in enumerate(students):save(f'student_{j}',model=m.state_dict(),optimizer=o.state_dict(),examples=counts[j],config=CFG)
  write(O/'learning_curves.json',curves);status(phase='meta_training',cycle=outer+1,planned_cycles=total_cycles,examples=cursor,external_gain=gain,teacher_gradient=tgrad,best_dev=best)
 # Save/resume artifacts even when the allotted primary wall-clock budget ends.
 write(O/'primary_complete.json',{'complete':True,'combined_examples':cursor,'unique_train_rows':len(order),'combined_pass_fraction':cursor/len(order),'student_counts':counts,'cycles':len(curves),'best_cycle':best_cycle,'best_dev':best,'elapsed_seconds':time.monotonic()-T,'ended_due_to_wall_budget':cursor<len(order)})
 teacher.load_state_dict(torch.load(O/'best_teacher.pt',weights_only=False)['model']);teacher.eval();del topt
 del students;torch.cuda.empty_cache();target_steps=max(1,min(math.ceil(min(counts)/CFG['batch']),int(CFG['comparison_wall_seconds']//(3*.9))));fresh_results={};start_comparison=time.monotonic()
 initial=MarketTransformer(teacher=True,patch=CFG['patch']).cuda();initial.load_state_dict(torch.load(O/'initial_teacher.pt',weights_only=False)['model']);initial.eval()
 # Matched fresh students: identical initial weights and identical training anchors.
 for mode in ['learned_teacher','direct_money','untrained_teacher']:
  student,opt=new_student(888);local=np.random.default_rng(888);sequence=local.permutation(D.splits['train']);examples=0;train_curve=[]
  for step in range(target_steps):
   # Equal completed updates across arms; wall limit checked between complete arms.
   ids=sequence[step*CFG['batch']:(step+1)*CFG['batch']]
   if not len(ids):break
   x,c,f=D.batch(ids,teacher=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):policy,value=student(x,c)
   action=policy.sample()
   if mode=='direct_money':
    v=torch.tensor(D.pnl[ids],device='cuda');r=torch.where(action==0,0.,torch.where(action==1,v,-v)-CFG['cost_per_ounce'])/D.reward_scale
   else:
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):scores=(teacher if mode=='learned_teacher' else initial)(x,c,f).mean
    r=torch.where(action==0,0.,2*scores.gather(1,(action-1).clamp(min=0)[:,None]).squeeze(-1)-1)
   loss,grad,entropy=student_step(student,opt,ids,r.detach(),action,policy,value);examples+=len(ids);del x,c,f,policy,value,action
   if step%100==0:status(phase='fresh_student',mode=mode,step=step,target_steps=target_steps,examples=examples,loss=loss,entropy=entropy)
  save('fresh_'+mode,model=student.state_dict(),optimizer=opt.state_dict(),examples=examples,config=CFG)
  # All development rows, then final test. Nothing updated based on test outcomes.
  metrics={}
  for split in ['dev','test']:
   status(phase='final_evaluation',mode=mode,split=split,rows=len(D.splits[split]));res=evaluate(student,D.splits[split],seed=5150,full=True);records=np.asarray(res.pop('records'),dtype=float).reshape(-1,4);actions=res.pop('actions');np.savez_compressed(O/f'{mode}_{split}.npz',rows=D.splits[split],actions=actions,trades=records)
   if len(records):
    unique,ix=np.unique(records[:,0].astype('int64')//86400,return_inverse=True);daily=np.bincount(ix,weights=records[:,3]);all_days=np.unique(D.ts[D.splits[split]]//86400);pnl_day=np.zeros(len(all_days));pnl_day[np.searchsorted(all_days,unique)]=daily
    res.update(net_total=float(records[:,3].sum()),mean_net_per_trade=float(records[:,3].mean()),profitable_day_fraction=float((pnl_day>0).mean()),daily_pnl_std=float(pnl_day.std()),worst_day=float(pnl_day.min()),max_trade_holding_minutes=float(((records[:,1]-records[:,0])/60).max()),holding_over_120min=int(((records[:,1]-records[:,0])>7200).sum()),gross_mean_per_trade=float(records[:,3].mean()+CFG['cost_per_ounce']),cost_scenarios_mean_per_trade={str(c):float(records[:,3].mean()+CFG['cost_per_ounce']-c) for c in [0,.1,.25,.5,1]})
    np.savez_compressed(O/f'{mode}_{split}_daily.npz',day=all_days,pnl=pnl_day)
   metrics[split]=res
  fresh_results[mode]={'train_examples':examples,'train_steps':target_steps,'metrics':metrics};write(O/'fresh_results.json',fresh_results);del student,opt;torch.cuda.empty_cache()
 write(O/'complete.json',{'complete':True,'seconds':time.monotonic()-T,'config':CFG,'fresh_results':fresh_results,'note':'One fresh transfer seed; full dataset preparation does not mean a full pass per meta student. See exact counts.'});status(phase='complete')
except Exception:
 (O/'error.txt').write_text(traceback.format_exc());status(phase='failed',error=traceback.format_exc());raise
