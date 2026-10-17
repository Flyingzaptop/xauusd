"""V2: outer RL reward teacher, adaptive inner exploration, contiguous evaluation."""
from pathlib import Path
import os,time,json,hashlib,random,traceback,math
import numpy as np,pandas as pd,torch
from model import MarketTransformer
from data import Data
from objective import trade_metrics,contiguous_ids
R=Path('/content/meta_reward');O=R/'results';O.mkdir(exist_ok=True);T=time.monotonic()
CFG={'version':2,'seed':177,'batch':256,'inner_steps':32,'students':3,'student_lr':3e-4,'teacher_lr':1e-4,'entropy_initial':.05,'entropy_target':.5*math.log(3),'entropy_dual_lr':.01,'entropy_min':.01,'entropy_max':2.,'warmup_updates':50,'dropout':.25,'patch':10,'depth_temporal':6,'depth_market':6,'hidden':192,'cost_per_ounce':.25,'concentration_coefficient':1.,'primary_wall_seconds':14400,'max_cycles':96,'min_cycles':48,'patience_cycles':48,'meta_calendar_days':14,'development_start':'2024-03-04','development_calendar_days':28,'test_start':'2025-01-06','test_calendar_days':84,'evaluation_batch':256,'fresh_max_steps':768,'final_day_cap':None}
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True

def write(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));os.replace(tmp,p)
def status(**kw):
 kw.update(seconds=round(time.monotonic()-T,1),allocated_GB=round(torch.cuda.memory_allocated()/1e9,2));write(R/'training_status.json',kw);print(json.dumps(kw),flush=True)
def save(name,**kw):
 p=O/(name+'.pt');torch.save(kw,p.with_suffix('.tmp'));os.replace(p.with_suffix('.tmp'),p)
def seeds(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def new_student(seed):
 seeds(seed);m=MarketTransformer(patch=CFG['patch']).cuda();opt=torch.optim.AdamW(m.parameters(),lr=CFG['student_lr'],weight_decay=.01)
 for g in opt.param_groups:g['entropy_alpha']=CFG['entropy_initial'];g['updates']=0
 return m,opt

def student_step(m,opt,rewards,policy,value,action):
 alpha=opt.param_groups[0]['entropy_alpha'];advantage=rewards-value.detach();scale=advantage.std(unbiased=False).clamp(min=.1)
 policy_loss=-(policy.log_prob(action)*advantage/scale).mean();value_loss=.5*(value-rewards).square().mean();ent=policy.entropy().mean();loss=policy_loss+value_loss-alpha*ent
 if not torch.isfinite(loss):raise RuntimeError('Nonfinite student loss')
 n=opt.param_groups[0]['updates']+1
 for g in opt.param_groups:g['updates']=n;g['lr']=CFG['student_lr']*min(1,n/CFG['warmup_updates'])
 opt.zero_grad(set_to_none=True);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.));opt.step()
 new_alpha=float(np.clip(alpha*np.exp(CFG['entropy_dual_lr']*(CFG['entropy_target']-float(ent.detach()))),CFG['entropy_min'],CFG['entropy_max']))
 for g in opt.param_groups:g['entropy_alpha']=new_alpha
 return {'total':float(loss.detach()),'policy':float(policy_loss.detach()),'value':float(value_loss.detach()),'entropy':float(ent.detach()),'entropy_alpha':new_alpha,'gradient_norm':grad}

@torch.no_grad()
def evaluate(model,ids,start,days,seed,save_prefix=None):
 model.eval();rng=np.random.default_rng(seed);actions=[];prob_sum=np.zeros(3);n=0
 for j in range(0,len(ids),CFG['evaluation_batch']):
  ix=ids[j:j+CFG['evaluation_batch']];x,c=D.batch(ix)
  with torch.autocast('cuda',dtype=torch.bfloat16):policy,_=model(x,c)
  probs=policy.probs.float().cpu().numpy();u=rng.random(len(ix));a=(u[:,None]>np.cumsum(probs,axis=1)).sum(1).clip(0,2);actions.append(a);prob_sum+=probs.sum(0);n+=len(ix)
  if j%(CFG['evaluation_batch']*32)==0:status(phase='evaluation',label=save_prefix or 'meta',done=j,total=len(ids))
 actions=np.concatenate(actions) if actions else np.array([],dtype=int);_,_,records=D.execute(ids,actions,CFG['cost_per_ounce']);calendar=np.arange(start//86400,start//86400+days);res=trade_metrics(records,calendar);res.update(rows=len(ids),seed=seed,mean_action_probability=(prob_sum/max(n,1)).tolist(),validation_objective=-res['utility_per_day'])
 if save_prefix:
  write(O/(save_prefix+'.json'),res);np.savez_compressed(O/(save_prefix+'.npz'),rows=ids,actions=actions,trades=np.asarray(records,dtype=float).reshape(-1,4),day=calendar,daily_net=res['daily_net'])
 model.train();return res

def timestamp(s):return int(pd.Timestamp(s,tz='UTC').timestamp())
try:
 seeds(CFG['seed']);write(O/'run_config.json',CFG);(O/'source').mkdir(exist_ok=True)
 hashes={}
 for p in R.glob('*.py'):(O/'source'/p.name).write_bytes(p.read_bytes());hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
 write(O/'source_hashes.json',hashes);status(phase='loading_dataset');D=Data();write(O/'dataset_protocol.json',{'counts':{k:len(v) for k,v in D.splits.items()},'config':CFG,'prepared':D.meta})
 teacher=MarketTransformer(teacher=True,patch=CFG['patch']).cuda();topt=torch.optim.AdamW(teacher.parameters(),lr=CFG['teacher_lr'],weight_decay=.01);save('initial_teacher',model=teacher.state_dict(),config=CFG)
 students=[new_student(177+100*i) for i in range(CFG['students'])];teacher.train();rng=np.random.default_rng(177);order=rng.permutation(D.splits['train']);cursor=0;baseline=0.;gain_second=1.;counts=[0]*3;curves=[];best=-np.inf;best_cycle=0
 dev_start=timestamp(CFG['development_start']);dev_ids=contiguous_ids(D.ts,D.splits['dev'],dev_start,CFG['development_calendar_days']);test_start=timestamp(CFG['test_start']);test_ids=contiguous_ids(D.ts,D.splits['test'],test_start,CFG['test_calendar_days'])
 meta_starts=list(range(timestamp('2023-01-09'),timestamp('2023-12-01'),14*86400));write(O/'evaluation_windows.json',{'meta_starts':meta_starts,'dev_start':dev_start,'dev_rows':len(dev_ids),'test_start':test_start,'test_rows':len(test_ids),'calendar_includes_zero_trade_days':True,'outer_reward':'delta of (net PnL - L2 norm of individual net trade PnL) / calendar days','test_used_for_selection':False})
 resume_path=O/'resume_state.pt';start_outer=0;previous_seconds=0.
 if resume_path.exists():
  state=torch.load(resume_path,map_location='cpu',weights_only=False)
  if state['config']!=CFG:raise RuntimeError('Refusing resume with changed scientific config')
  teacher.load_state_dict(state['teacher']);topt.load_state_dict(state['teacher_optimizer'])
  for (m,opt),saved in zip(students,state['students']):m.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
  cursor=state['cursor'];order=state['order'];start_outer=state['outer'];baseline=state['baseline'];gain_second=state['gain_second'];counts=state['counts'];curves=state['curves'];best=state['best'];best_cycle=state['best_cycle'];rng.bit_generator.state=state['numpy_generator'];torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);previous_seconds=float(curves[-1]['seconds']) if curves else 0.
  write(O/'resume_loaded.json',{'outer':start_outer,'cursor':cursor,'previous_elapsed_seconds':previous_seconds})
  del state
 start_primary=time.monotonic()-previous_seconds;stop_reason='max_cycles'
 for outer in range(start_outer,CFG['max_cycles']):
  if outer and time.monotonic()-start_primary>CFG['primary_wall_seconds']:stop_reason='wall_budget';break
  if outer>=CFG['min_cycles'] and outer-best_cycle>=CFG['patience_cycles']:stop_reason='validation_patience';break
  if cursor>=len(order):stop_reason='combined_data_pass';break
  si=outer%3;student,opt=students[si];meta_start=int(rng.choice(meta_starts));meta_ids=contiguous_ids(D.ts,D.splits['meta'],meta_start,CFG['meta_calendar_days']);before=evaluate(student,meta_ids,meta_start,CFG['meta_calendar_days'],9000+outer)
  logs=[];sl=[]
  for k in range(CFG['inner_steps']):
   ids=order[cursor:cursor+CFG['batch']];cursor+=len(ids);counts[si]+=len(ids)
   if not len(ids):break
   x,c,f=D.batch(ids,teacher=True);state=torch.cuda.get_rng_state()
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):dist=teacher(x,c,f);scores=dist.sample();old_lp=dist.log_prob(scores).detach()
   logs.append({'ids':ids.copy(),'scores':scores.cpu(),'rng':state.cpu(),'old_lp':old_lp.cpu()})
   with torch.autocast('cuda',dtype=torch.bfloat16):policy,value=student(x,c)
   action=policy.sample();selected=scores.gather(1,(action-1).clamp(min=0)[:,None]).squeeze(-1);reward=torch.where(action==0,torch.zeros_like(selected),2*selected-1).detach();sl.append(student_step(student,opt,reward,policy,value,action));del x,c,f,policy,value,action,scores,dist
  after=evaluate(student,meta_ids,meta_start,CFG['meta_calendar_days'],9000+outer);gain=after['utility_per_day']-before['utility_per_day'];raw_adv=gain-baseline;gain_second=.95*gain_second+.05*raw_adv**2;adv=raw_adv/max(math.sqrt(gain_second),.1);baseline=.9*baseline+.1*gain
  topt.zero_grad(set_to_none=True);replay=0.;teacher_loss=0.;den=sum(len(v['ids']) for v in logs)
  for rec in logs:
   x,c,f=D.batch(rec['ids'],teacher=True);state=torch.cuda.get_rng_state();torch.cuda.set_rng_state(rec['rng'])
   with torch.autocast('cuda',dtype=torch.bfloat16):dist=teacher(x,c,f)
   torch.cuda.set_rng_state(state);lp=dist.log_prob(rec['scores'].cuda());replay=max(replay,float((lp.detach()-rec['old_lp'].cuda()).abs().max()))
   if replay>.03:raise RuntimeError('Teacher replay mismatch')
   tl=-lp.sum()*float(adv)/max(den,1);teacher_loss+=float(tl.detach());tl.backward();del x,c,f,dist,lp,tl
  tgrad=float(torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.));topt.step()
  row={'cycle':outer+1,'student':si,'examples':cursor,'student_examples':counts.copy(),'meta_start':meta_start,'meta_before':before,'meta_after':after,'teacher_external_reward':gain,'teacher_advantage':adv,'teacher_policy_train_loss':teacher_loss,'teacher_validation_objective':after['validation_objective'],'teacher_gradient_norm':tgrad,'teacher_likelihood_replay_max_error':replay,'student_loss_components':{key:float(np.mean([v[key] for v in sl])) for key in sl[0]},'seconds':time.monotonic()-T};curves.append(row)
  if (outer+1)%8==0 or outer==0:
   dev=evaluate(student,dev_ids,dev_start,CFG['development_calendar_days'],2424,save_prefix=f'dev_cycle_{outer+1:03}');row['development']=dev
   if dev['utility_per_day']>best:best=dev['utility_per_day'];best_cycle=outer+1;save('best_teacher',model=teacher.state_dict(),optimizer=topt.state_dict(),cycle=best_cycle,development=dev,config=CFG);save('best_student',model=student.state_dict(),optimizer=opt.state_dict(),student_index=si,cycle=best_cycle,config=CFG)
  # Recoverable every completed cycle, independent of validation frequency.
  write(O/'learning_curves.json',curves);save('resume_state',teacher=teacher.state_dict(),teacher_optimizer=topt.state_dict(),students=[{'model':m.state_dict(),'optimizer':o.state_dict()} for m,o in students],cursor=cursor,order=order,outer=outer+1,baseline=baseline,gain_second=gain_second,counts=counts,curves=curves,best=best,best_cycle=best_cycle,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_generator=rng.bit_generator.state,config=CFG)
  status(phase='meta_training',cycle=outer+1,examples=cursor,best_dev_utility=best,entropy=row['student_loss_components']['entropy'],entropy_alpha=row['student_loss_components']['entropy_alpha'])
 write(O/'primary_complete.json',{'cycles':len(curves),'combined_examples':cursor,'combined_pass_fraction':cursor/len(order),'student_counts':counts,'stop_reason':stop_reason,'best_cycle':best_cycle,'best_dev_utility':best,'elapsed_seconds':time.monotonic()-T})
 teacher.load_state_dict(torch.load(O/'best_teacher.pt',weights_only=False)['model']);teacher.eval();del students,topt;torch.cuda.empty_cache();initial=MarketTransformer(teacher=True,patch=CFG['patch']).cuda();initial.load_state_dict(torch.load(O/'initial_teacher.pt',weights_only=False)['model']);initial.eval();steps=max(1,min(CFG['fresh_max_steps'],min(counts)//CFG['batch']));fresh={}
 for mode in ['learned_teacher','direct_money','untrained_teacher']:
  student,opt=new_student(888);sequence=np.random.default_rng(888).permutation(D.splits['train']);examples=0;logs=[]
  for step in range(steps):
   ids=sequence[step*CFG['batch']:(step+1)*CFG['batch']];x,c,f=D.batch(ids,teacher=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):policy,value=student(x,c)
   action=policy.sample()
   if mode=='direct_money':
    v=torch.tensor(D.pnl[ids],device='cuda');reward=torch.where(action==0,0.,torch.where(action==1,v,-v)-CFG['cost_per_ounce'])/D.reward_scale
   else:
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):scores=(teacher if mode=='learned_teacher' else initial)(x,c,f).mean
    reward=torch.where(action==0,0.,2*scores.gather(1,(action-1).clamp(min=0)[:,None]).squeeze(-1)-1)
   logs.append(student_step(student,opt,reward.detach(),policy,value,action));examples+=len(ids);del x,c,f,policy,value,action
   if step%64==0:status(phase='fresh_student',mode=mode,step=step,total_steps=steps)
  save('fresh_'+mode,model=student.state_dict(),optimizer=opt.state_dict(),examples=examples,config=CFG);write(O/('fresh_'+mode+'_loss.json'),logs)
  metrics={}
  for split,ids,start,days in [('dev',dev_ids,dev_start,CFG['development_calendar_days']),('test',test_ids,test_start,CFG['test_calendar_days'])]:metrics[split]=evaluate(student,ids,start,days,5150,save_prefix=f'{mode}_{split}')
  fresh[mode]={'train_steps':steps,'train_examples':examples,'metrics':metrics};write(O/'fresh_results.json',fresh);del student,opt;torch.cuda.empty_cache()
 write(O/'complete.json',{'complete':True,'seconds':time.monotonic()-T,'config':CFG,'fresh_results':fresh});status(phase='complete')
except Exception:
 (O/'error.txt').write_text(traceback.format_exc());status(phase='failed',error=traceback.format_exc());raise
