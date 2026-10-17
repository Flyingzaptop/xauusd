from pathlib import Path
import subprocess,tarfile,json,torch,psutil,os
r=Path('/content/meta_reward');r.mkdir(exist_ok=True)
with tarfile.open('/content/meta-reward-v2-code.tar.gz') as t:t.extractall(r,filter='data')
subprocess.check_call(['python','-m','pip','install','-q','pyarrow','psutil','requests'])
env={'gpu':torch.cuda.get_device_name(0),'vram_bytes':torch.cuda.get_device_properties(0).total_memory,'RAM_bytes':psutil.virtual_memory().total,'CPU':os.cpu_count(),'torch':torch.__version__}
(r/'environment.json').write_text(json.dumps(env,indent=2));assert env['vram_bytes']>78_000_000_000,'Requires 80GB GPU for unchanged full cache'
for name in ['check_objective.py','check_features.py','check_data_model.py','check_gpu_meta.py']:
 p=subprocess.run(['python',str(r/name)],cwd=r,capture_output=True,text=True,timeout=180)
 (r/(name+'.log')).write_text(p.stdout+'\n'+p.stderr)
 if p.returncode:raise RuntimeError(name+' failed: '+p.stderr[-2000:])
p=subprocess.Popen(['python','-u',str(r/'supervise.py')],stdout=open(r/'supervisor.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'supervisor_pid':p.pid,'environment':env,'checks_passed':True}))
