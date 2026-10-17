from pathlib import Path
import tarfile,subprocess,json
r=Path('/content/meta_reward')
with tarfile.open('/content/meta-reward-training-code.tar.gz') as t:t.extractall(r,filter='data')
p=subprocess.run(['python',str(r/'check_gpu_meta.py')],cwd=r,capture_output=True,text=True,timeout=120)
(r/'gpu_meta_checks.log').write_text(p.stdout+'\n'+p.stderr);print(p.stdout);print(p.stderr[-2000:]);assert p.returncode==0
(r/'gpu_meta_checks.json').write_text(p.stdout.strip().splitlines()[-1])
