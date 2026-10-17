"""Sequential PPO, elapsed-minute GAE and a shared rollout for all evaluations."""
import numpy as np
import torch
from environment import TradingEnv,LONG,WAIT
from synthetic import future_view

def rollout(actor,episodes,execution,seed,teacher=None,teacher_mode='sample',greedy=False,track_teacher=False,reward_unit=10.):
    rng=np.random.default_rng(seed);teacher_rng=np.random.default_rng(seed+100000000);envs=[TradingEnv(e,execution) for e in episodes];paths=[[] for _ in envs];teacher_logs=[]
    while True:
        active=[i for i,e in enumerate(envs) if not e.done]
        if not active:break
        x=torch.tensor(np.stack([envs[i].observation() for i in active]),dtype=torch.float32)
        masks=torch.tensor(np.stack([envs[i].mask() for i in active]))
        with torch.no_grad():dist,values=actor(x,masks)
        probs=dist.probs.numpy();actions=probs.argmax(1) if greedy else (rng.random(len(active))[:,None]>probs.cumsum(1)).sum(1).clip(0,2)
        shaped={}
        chosen=[j for j,a in enumerate(actions) if a!=WAIT]
        if teacher is not None and chosen:
            future=torch.tensor(np.stack([future_view(episodes[active[j]],envs[active[j]].i,execution.horizon_minutes) for j in chosen]))
            with torch.set_grad_enabled(track_teacher):
                td=teacher(x[chosen],future)
                idx=torch.tensor(actions[chosen]-1);row=torch.arange(len(chosen));selected=torch.distributions.Beta(td.concentration1[row,idx],td.concentration0[row,idx])
                # Explicit RNG fork keeps student/random-action stream independent.
                with torch.random.fork_rng():
                    torch.manual_seed(int(teacher_rng.integers(2**31-1)))
                    score=selected.sample() if teacher_mode=='sample' else selected.mean
                if track_teacher:teacher_logs.append(selected.log_prob(score).sum())
            shaped={j:2*float(score[k].detach())-1 for k,j in enumerate(chosen)}
        for j,i in enumerate(active):
            env=envs[i];start=env.i;_,money,done,tr=env.step(int(actions[j]));reward=(money/reward_unit if teacher is None else shaped.get(j,0.))*execution.gamma_per_minute**max(tr['minutes']-1,0)
            paths[i].append({'observation':x[j].numpy(),'mask':masks[j].numpy(),'action':int(actions[j]),'old_logp':float(np.log(max(probs[j,actions[j]],1e-30))),
                             'value':float(values[j]),'reward':reward,'money':money,'discount':tr['discount'],'minutes':tr['minutes'],'done':done,'start':start,'probabilities':probs[j].tolist()})
    return paths,[e.metrics() for e in envs],teacher_logs

def advantages(paths,lam_per_minute=.97):
    flat=[]
    for path in paths:
        gae=0.;nextv=0.
        for r in reversed(path):
            delta=r['reward']+r['discount']*nextv-r['value'];gae=delta+r['discount']*lam_per_minute**r['minutes']*gae
            r['advantage']=gae;r['target_value']=gae+r['value'];nextv=r['value']
        flat.extend(path)
    return flat

def update(actor,optimizer,paths,seed,epochs=3,entropy_alpha=.01,minibatch=256):
    rows=advantages(paths);x=torch.tensor(np.stack([r['observation'] for r in rows]));mask=torch.tensor(np.stack([r['mask'] for r in rows]))
    action=torch.tensor([r['action'] for r in rows]);old=torch.tensor([r['old_logp'] for r in rows]);adv=torch.tensor([r['advantage'] for r in rows]);target=torch.tensor([r['target_value'] for r in rows])
    adv=(adv-adv.mean())/adv.std(unbiased=False).clamp(min=1e-6);rng=np.random.default_rng(seed);logs=[]
    for _ in range(epochs):
        order=rng.permutation(len(rows))
        for s in range(0,len(rows),minibatch):
            ix=order[s:s+minibatch];d,v=actor(x[ix],mask[ix]);ratio=(d.log_prob(action[ix])-old[ix]).exp();pl=-torch.minimum(ratio*adv[ix],ratio.clamp(.8,1.2)*adv[ix]).mean()
            vl=.5*(v-target[ix]).square().mean();en=d.entropy().mean();loss=pl+.5*vl-entropy_alpha*en
            optimizer.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);optimizer.step()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite PPO loss')
            logs.append({'policy':float(pl.detach()),'value':float(vl.detach()),'entropy':float(en.detach()),'gradient_norm':float(gn),'approx_kl':float((old[ix]-d.log_prob(action[ix])).mean().detach())})
    return {k:float(np.mean([r[k] for r in logs])) for k in logs[0]}

def train(actor,episodes_factory,execution,seed,updates=100,batch_episodes=16,teacher=None,teacher_mode='sample',track_teacher=False,optimizer=None):
    optimizer=optimizer or torch.optim.AdamW(actor.parameters(),lr=3e-4,weight_decay=.001);logs=[];credit=[]
    for u in range(updates):
        episodes=episodes_factory(u,batch_episodes)
        paths,metrics,tlog=rollout(actor,episodes,execution,seed+u,teacher,teacher_mode,track_teacher=track_teacher);credit.extend(tlog)
        log=update(actor,optimizer,paths,seed+u);log.update(update=u+1,net_mean=float(np.mean([m['net'] for m in metrics])),transitions=sum(map(len,paths)));logs.append(log)
    return logs,credit
