from pathlib import Path
import subprocess,json,tarfile
r=Path('/content/intermarket');r.mkdir(exist_ok=True)
with tarfile.open('/content/intermarket-prep-code.tar.gz') as archive:archive.extractall(r,filter='data')
print(subprocess.check_output(['free','-h'],text=True))
p=subprocess.Popen(['bash','-c','python -m pip install -q pandas pyarrow lightgbm psutil requests && python /content/intermarket/prepare.py'],stdout=open(r/'prepare.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':p.pid,'root':str(r)}))
