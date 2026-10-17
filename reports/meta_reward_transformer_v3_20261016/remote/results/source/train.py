"""V3: warm-start teacher, centered direction rewards, balanced regimes, accumulated outer RL."""
from pathlib import Path
import os,time,json,hashlib,random,traceback,math
import numpy as np,pandas as pd,torch
from model import MarketTransformer
from data import Data
from objective import trade_metrics,contiguous_ids,centered_teacher_reward,entropy_coefficient,robust_excess
R=Path('/content/meta_reward');O=R/'results';O.mkdir(exist_ok=True);T=time.monotonic()
CFG={'version':3,'seed':177,'batch':256,'inner_steps':32,'students':3,'student_lr':1e-4,'teacher_lr':3e-5,'teacher_accumulation':4,'advantage_clip':3.,'calibration_rows':4096,'entropy_start':.10,'entropy_end':.03,'entropy_schedule_steps':1500,'warmup_updates':50,'dropout':.25,'patch':10,'depth_temporal':6,'depth_market':6,'hidden':192,'cost_per_ounce':.25,'concentration_coefficient':1.,'primary_wall_seconds':10800,'max_cycles':96,'min_cycles':48,'patience_cycles':48,'meta_calendar_days':14,'development_starts':['2024-03-04','2024-06-03','2024-09-02','2024-12-02'],'development_calendar_days':28,'test_start':'2025-04-07','test_calendar_days':84,'evaluation_batch':256,'fresh_max_steps':768,'fresh_seeds':[888,889]}
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
 for g in opt.param_groups:g['entropy_alpha']=CFG['entropy_start'];g['updates']=0
 return m,opt

def student_step(m,opt,rewards,policy,value,action):
 alpha=entropy_coefficient(opt.param_groups[0]['updates'],CFG['entropy_start'],CFG['entropy_end'],CFG['entropy_schedule_steps']);advantage=rewards-value.detach();scale=advantage.std(unbiased=False).clamp(min=.1)
 policy_loss=-(policy.log_prob(action)*advantage/scale).mean();value_loss=.5*(value-rewards).square().mean();ent=policy.entropy().mean();loss=policy_loss+value_loss-alpha*ent
 if not torch.isfinite(loss):raise RuntimeError('Nonfinite student loss')
 n=opt.param_groups[0]['updates']+1
 for g in opt.param_groups:g['updates']=n;g['lr']=CFG['student_lr']*min(1,n/CFG['warmup_updates'])
 opt.zero_grad(set_to_none=True);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.));opt.step()
 new_alpha=entropy_coefficient(n,CFG['entropy_start'],CFG['entropy_end'],CFG['entropy_schedule_steps'])
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
 actions=np.concatenate(actions) if actions else np.array([],dtype=int);_,_,records=D.execute(ids,actions,CFG['cost_per_ounce']);calendar=np.arange(start//86400,start//86400+days);res=trade_metrics(records,calendar);res.update(long_trades=sum(v[2]==1 for v in records),short_trades=sum(v[2]==-1 for v in records),rows=len(ids),seed=seed,mean_action_probability=(prob_sum/max(n,1)).tolist(),validation_objective=-res['utility_per_day'])
 if save_prefix:
  write(O/(save_prefix+'.json'),res);np.savez_compressed(O/(save_prefix+'.npz'),rows=ids,actions=actions,trades=np.asarray(records,dtype=float).reshape(-1,4),day=calendar,daily_net=res['daily_net'])
 model.train();return res

def timestamp(s):return int(pd.Timestamp(s,tz='UTC').timestamp())
@torch.no_grad()
def calibrate(model):
 was_training=model.training;model.eval();outputs=[]
 for j in range(0,len(calibration_ids),CFG['evaluation_batch']):
  x,c,f=D.batch(calibration_ids[j:j+CFG['evaluation_batch']],teacher=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):dist=model(x,c,f)
  outputs.append(dist.mean.float());del x,c,f,dist
 values=torch.cat(outputs);center=values.mean(0).detach();diagnostic={'mean_LONG_SHORT':center.cpu().tolist(),'std_of_conditional_means':values.std(0).cpu().tolist(),'rows':len(values)}
 model.train(was_training);return center,diagnostic

def control_metrics(ids,start,days):
 result={};calendar=np.arange(start//86400,start//86400+days)
 for name,a in [('long',1),('short',2),('flat',0)]:
  _,_,records=D.execute(ids,np.full(len(ids),a),CFG['cost_per_ounce']);m=trade_metrics(records,calendar);m.update(long_trades=len(records) if a==1 else 0,short_trades=len(records) if a==2 else 0);result[name]=m
 return result

def evaluate_development(model,seed,prefix):
 windows=[]
 for j,(start,ids,controls) in enumerate(dev_windows):
  windows.append(evaluate(model,ids,start,CFG['development_calendar_days'],seed+j,save_prefix=f'{prefix}_window{j+1}'))
 score,excesses=robust_excess(windows,[w[2] for w in dev_windows]);days=sum(w['days'] for w in windows);net=sum(w['net_total'] for w in windows)
 return {'utility_per_day':score,'selection_criterion':'worst excess utility against best static control across 4 dev windows','mean_net_dollars_per_day':net/days,'net_total':net,'days':days,'trades':sum(w['trades'] for w in windows),'long_trades':sum(w['long_trades'] for w in windows),'short_trades':sum(w['short_trades'] for w in windows),'window_excess':excesses,'windows':windows,'validation_objective':-score}

try:
 seeds(CFG['seed']);write(O/'run_config.json',CFG);(O/'source').mkdir(exist_ok=True)
 hashes={}
 for p in R.glob('*.py'):(O/'source'/p.name).write_bytes(p.read_bytes());hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
 write(O/'source_hashes.json',hashes);status(phase='loading_dataset');D=Data();write(O/'dataset_protocol.json',{'counts':{k:len(v) for k,v in D.splits.items()},'config':CFG,'prepared':D.meta})
 teacher=MarketTransformer(teacher=True,patch=CFG['patch']).cuda();parent=torch.load(R/'teacher_v2_cycle80.pt',map_location='cpu',weights_only=True);teacher.load_state_dict(parent['model']);write(O/'warm_start.json',{'parent_cycle':parent['cycle'],'parent_sha256':hashlib.sha256((R/'teacher_v2_cycle80.pt').read_bytes()).hexdigest(),'optimizer_reset_for_new_objective':True});del parent;topt=torch.optim.AdamW(teacher.parameters(),lr=CFG['teacher_lr'],weight_decay=.01);save('initial_teacher',model=teacher.state_dict(),config=CFG)
 students=[new_student(177+100*i) for i in range(CFG['students'])];teacher.train();rng=np.random.default_rng(177);order=rng.permutation(D.splits['train']);cursor=0;baseline=0.;gain_second=1.;counts=[0]*3;curves=[];best=-np.inf;best_cycle=0
 calibration_ids=np.random.default_rng(1701).choice(D.splits['train'],min(CFG['calibration_rows'],len(D.splits['train'])),replace=False);np.save(O/'calibration_anchor_indices.npy',calibration_ids);reward_center,calibration=calibrate(teacher);write(O/'initial_reward_calibration.json',calibration)
 dev_windows=[]
 for date in CFG['development_starts']:
  start=timestamp(date);ids=contiguous_ids(D.ts,D.splits['dev'],start,CFG['development_calendar_days']);dev_windows.append((start,ids,control_metrics(ids,start,CFG['development_calendar_days'])))
 test_start=timestamp(CFG['test_start']);test_ids=contiguous_ids(D.ts,D.splits['test'],test_start,CFG['test_calendar_days'])
 meta_starts=list(range(timestamp('2023-01-09'),timestamp('2023-12-01'),14*86400));meta_windows={};pools={'up':[],'down':[]}
 for start in meta_starts:
  ids=contiguous_ids(D.ts,D.splits['meta'],start,CFG['meta_calendar_days']);controls=control_metrics(ids,start,CFG['meta_calendar_days']);meta_windows[start]=(ids,controls);pools['up' if controls['long']['net_total']>controls['short']['net_total'] else 'down'].append(start)
 assert all(pools.values()),'Need both historical rising and falling meta windows'
 write(O/'evaluation_windows.json',{'meta_pools':pools,'development_starts':CFG['development_starts'],'dev_controls':[w[2] for w in dev_windows],'test_start':test_start,'test_rows':len(test_ids),'calendar_includes_zero_trade_days':True,'outer_reward':'improvement in worst of up/down-window excess utility vs best fixed-direction control','test_used_for_selection':False})
 topt.zero_grad(set_to_none=True);teacher_updates=0
 start_primary=time.monotonic();stop_reason='max_cycles'
 for outer in range(CFG['max_cycles']):
  if outer and outer%CFG['teacher_accumulation']==0 and time.monotonic()-start_primary>CFG['primary_wall_seconds']:stop_reason='wall_budget';break
  if outer%CFG['teacher_accumulation']==0 and outer>=CFG['min_cycles'] and outer-best_cycle>=CFG['patience_cycles']:stop_reason='validation_patience';break
  if cursor>=len(order):stop_reason='combined_data_pass';break
  si=outer%3;student,opt=students[si]
  if outer%CFG['teacher_accumulation']==0:
   pair=[int(rng.choice(pools['up'])),int(rng.choice(pools['down']))];topt.zero_grad(set_to_none=True)
  before=[evaluate(student,meta_windows[start][0],start,CFG['meta_calendar_days'],9000+outer*2+j) for j,start in enumerate(pair)]
  logs=[];sl=[]
  for k in range(CFG['inner_steps']):
   ids=order[cursor:cursor+CFG['batch']];cursor+=len(ids);counts[si]+=len(ids)
   if not len(ids):break
   x,c,f=D.batch(ids,teacher=True);state=torch.cuda.get_rng_state()
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):dist=teacher(x,c,f);scores=dist.sample();old_lp=dist.log_prob(scores).detach()
   logs.append({'ids':ids.copy(),'scores':scores.cpu(),'rng':state.cpu(),'old_lp':old_lp.cpu()})
   with torch.autocast('cuda',dtype=torch.bfloat16):policy,value=student(x,c)
   action=policy.sample();reward=centered_teacher_reward(scores,reward_center,action).detach();sl.append(student_step(student,opt,reward,policy,value,action));del x,c,f,policy,value,action,scores,dist
  after=[evaluate(student,meta_windows[start][0],start,CFG['meta_calendar_days'],9000+outer*2+j) for j,start in enumerate(pair)];before_score,before_excess=robust_excess(before,[meta_windows[start][1] for start in pair]);after_score,after_excess=robust_excess(after,[meta_windows[start][1] for start in pair]);gain=after_score-before_score;raw_adv=gain-baseline;gain_second=.95*gain_second+.05*raw_adv**2;adv=float(np.clip(raw_adv/max(math.sqrt(gain_second),.1),-CFG['advantage_clip'],CFG['advantage_clip']));baseline=.9*baseline+.1*gain
  replay=0.;teacher_loss=0.;den=sum(len(v['ids']) for v in logs)
  for rec in logs:
   x,c,f=D.batch(rec['ids'],teacher=True);state=torch.cuda.get_rng_state();torch.cuda.set_rng_state(rec['rng'])
   with torch.autocast('cuda',dtype=torch.bfloat16):dist=teacher(x,c,f)
   torch.cuda.set_rng_state(state);lp=dist.log_prob(rec['scores'].cuda());replay=max(replay,float((lp.detach()-rec['old_lp'].cuda()).abs().max()))
   if replay>.03:raise RuntimeError('Teacher replay mismatch')
   tl=-lp.sum()*float(adv)/(max(den,1)*CFG['teacher_accumulation']);teacher_loss+=float(tl.detach());tl.backward();del x,c,f,dist,lp,tl
  tgrad=float(torch.sqrt(sum(p.grad.detach().float().square().sum() for p in teacher.parameters() if p.grad is not None)))
  if (outer+1)%CFG['teacher_accumulation']==0:
   torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.);topt.step();teacher_updates+=1;reward_center,calibration=calibrate(teacher)
  row={'cycle':outer+1,'student':si,'examples':cursor,'student_examples':counts.copy(),'meta_starts':pair.copy(),'meta_before_excess':before_excess,'meta_after_excess':after_excess,'teacher_optimizer_steps':teacher_updates,'reward_calibration':calibration,'meta_before':before,'meta_after':after,'teacher_external_reward':gain,'teacher_advantage':adv,'teacher_policy_train_loss':teacher_loss,'teacher_validation_objective':-after_score,'teacher_gradient_norm':tgrad,'teacher_likelihood_replay_max_error':replay,'student_loss_components':{key:float(np.mean([v[key] for v in sl])) for key in sl[0]},'seconds':time.monotonic()-T};curves.append(row)
  if (outer+1)%8==0 or outer==0:
   dev=evaluate_development(student,2424,prefix=f'dev_cycle_{outer+1:03}');row['development']=dev
   if dev['utility_per_day']>best:best=dev['utility_per_day'];best_cycle=outer+1;save('best_teacher',model=teacher.state_dict(),optimizer=topt.state_dict(),cycle=best_cycle,development=dev,reward_center=reward_center.cpu(),reward_calibration=calibration,config=CFG);save('best_student',model=student.state_dict(),optimizer=opt.state_dict(),student_index=si,cycle=best_cycle,config=CFG)
  # Recoverable every completed cycle, independent of validation frequency.
  write(O/'learning_curves.json',curves);save('resume_state',teacher=teacher.state_dict(),teacher_optimizer=topt.state_dict(),teacher_gradients=[None if p.grad is None else p.grad.detach().cpu() for p in teacher.parameters()],teacher_updates=teacher_updates,meta_pair=pair.copy(),reward_center=reward_center.cpu(),reward_calibration=calibration,students=[{'model':m.state_dict(),'optimizer':o.state_dict()} for m,o in students],cursor=cursor,order=order,outer=outer+1,baseline=baseline,gain_second=gain_second,counts=counts,curves=curves,best=best,best_cycle=best_cycle,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_generator=rng.bit_generator.state,config=CFG)
  status(phase='meta_training',cycle=outer+1,examples=cursor,best_dev_utility=best,entropy=row['student_loss_components']['entropy'],entropy_alpha=row['student_loss_components']['entropy_alpha'])
 write(O/'primary_complete.json',{'cycles':len(curves),'combined_examples':cursor,'combined_pass_fraction':cursor/len(order),'student_counts':counts,'stop_reason':stop_reason,'best_cycle':best_cycle,'best_dev_utility':best,'teacher_optimizer_steps':teacher_updates,'elapsed_seconds':time.monotonic()-T})
 chosen=torch.load(O/'best_teacher.pt',weights_only=False);teacher.load_state_dict(chosen['model']);teacher.eval();learned_center=chosen['reward_center'].cuda();del chosen,students,topt;torch.cuda.empty_cache()
 initial=MarketTransformer(teacher=True,patch=CFG['patch']).cuda();initial.load_state_dict(torch.load(O/'initial_teacher.pt',weights_only=False)['model']);initial.eval();old_center,old_calibration=calibrate(initial);write(O/'frozen_reward_calibration.json',{'learned':learned_center.cpu().tolist(),'previous_teacher':old_calibration});steps=max(1,min(CFG['fresh_max_steps'],min(counts)//CFG['batch']));fresh={}
 # A new, unobserved 84-day final period; no tuning or selection based on it.
 write(O/'test_static_controls.json',control_metrics(test_ids,test_start,CFG['test_calendar_days']))
 for fresh_seed in CFG['fresh_seeds']:
  for mode in ['learned_teacher','previous_teacher','direct_money']:
   key=f'{mode}_seed{fresh_seed}';student,opt=new_student(fresh_seed);sequence=np.random.default_rng(fresh_seed).permutation(D.splits['train']);examples=0;logs=[]
   for step in range(steps):
    ids=sequence[step*CFG['batch']:(step+1)*CFG['batch']];x,c,f=D.batch(ids,teacher=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):policy,value=student(x,c)
    action=policy.sample()
    if mode=='direct_money':
     v=torch.tensor(D.pnl[ids],device='cuda');reward=torch.where(action==0,0.,torch.where(action==1,v,-v)-CFG['cost_per_ounce'])/D.reward_scale
    else:
     with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):scores=(teacher if mode=='learned_teacher' else initial)(x,c,f).mean
     reward=centered_teacher_reward(scores,learned_center if mode=='learned_teacher' else old_center,action)
    logs.append(student_step(student,opt,reward.detach(),policy,value,action));examples+=len(ids);del x,c,f,policy,value,action
    if step%64==0:status(phase='fresh_student',mode=key,step=step,total_steps=steps)
   save('fresh_'+key,model=student.state_dict(),examples=examples,seed=fresh_seed,config=CFG);write(O/('fresh_'+key+'_loss.json'),logs)
   dev=evaluate_development(student,5150,prefix=key+'_dev');test=evaluate(student,test_ids,test_start,CFG['test_calendar_days'],5150,save_prefix=key+'_test');fresh[key]={'mode':mode,'seed':fresh_seed,'train_steps':steps,'train_examples':examples,'metrics':{'dev':dev,'test':test}};write(O/'fresh_results.json',fresh);del student,opt;torch.cuda.empty_cache()
 write(O/'complete.json',{'complete':True,'seconds':time.monotonic()-T,'config':CFG,'fresh_results':fresh});status(phase='complete')
except Exception:
 (O/'error.txt').write_text(traceback.format_exc());status(phase='failed',error=traceback.format_exc());raise
