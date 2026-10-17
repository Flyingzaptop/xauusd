from pathlib import Path
import subprocess,tarfile,json,torch,psutil,os,io,base64
r=Path('/content/meta_reward');r.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE_B64))) as t:t.extractall(r,filter='data')
import shutil,hashlib
shutil.copyfile('/content/teacher_v2_cycle80.pt',r/'teacher_v2_cycle80.pt')
assert hashlib.sha256((r/'teacher_v2_cycle80.pt').read_bytes()).hexdigest()==WARM_START_SHA256,'Warm-start upload integrity failure'
subprocess.check_call(['python','-m','pip','install','-q','pyarrow','psutil','requests'])
env={'gpu':torch.cuda.get_device_name(0),'vram_bytes':torch.cuda.get_device_properties(0).total_memory,'RAM_bytes':psutil.virtual_memory().total,'CPU':os.cpu_count(),'torch':torch.__version__,'execution_mode':'foreground kernel waits for full supervisor job'}
(r/'environment.json').write_text(json.dumps(env,indent=2));assert env['vram_bytes']>78_000_000_000,'Requires 80GB GPU for full cache';print(json.dumps(env),flush=True)
for name in ['check_objective.py','check_training_math.py','check_features.py','check_data_model.py','check_gpu_meta.py']:
 p=subprocess.run(['python',str(r/name)],cwd=r,capture_output=True,text=True,timeout=180);(r/(name+'.log')).write_text(p.stdout+'\n'+p.stderr)
 if p.returncode:raise RuntimeError(name+' failed: '+p.stderr[-2000:])
 print(name+': PASSED',flush=True)
# Blocking execution: kernel remains genuinely busy while the job runs.
subprocess.run(['python','-u',str(r/'supervise.py')],cwd=r,check=True)
print((r/'training_result.json').read_text(),flush=True)
