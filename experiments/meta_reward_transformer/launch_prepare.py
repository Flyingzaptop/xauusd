from pathlib import Path
import subprocess,tarfile,json,torch,psutil,os
r=Path('/content/meta_reward');r.mkdir(exist_ok=True)
with tarfile.open('/content/meta-reward-code.tar.gz') as t:t.extractall(r,filter='data')
subprocess.check_call(['python','-m','pip','install','-q','pyarrow','psutil','requests'])
env={'gpu':torch.cuda.get_device_name(0),'vram_bytes':torch.cuda.get_device_properties(0).total_memory,'RAM_bytes':psutil.virtual_memory().total,'CPU':os.cpu_count(),'torch':torch.__version__}
(r/'environment.json').write_text(json.dumps(env,indent=2))
p=subprocess.Popen(['python','-u',str(r/'prepare.py')],stdout=open(r/'prepare.log','w'),stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({'pid':p.pid,'environment':env}))
