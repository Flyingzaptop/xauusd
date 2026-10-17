import tarfile
with tarfile.open('/content/stage2-inputs.tar.gz') as tar: tar.extractall('/content',filter='data')
import json,os,subprocess,tarfile,time
from pathlib import Path
out=Path('/content/stage2-training');out.mkdir(exist_ok=True)
started=time.time();last='';archived=0

def archive():
    with tarfile.open('/content/stage2-results.tmp.tar.gz','w:gz',compresslevel=1) as tar:
        tar.add(out,arcname='remote')
    os.replace('/content/stage2-results.tmp.tar.gz','/content/stage2-results.tar.gz')
with (out/'console.log').open('w') as log:
    p=subprocess.Popen(['python','-u','/content/stage2-src/train.py','--data','/content/stage2-dataset.npz','--pretrained','/content/pretrained','--out',str(out)],stdout=log,stderr=subprocess.STDOUT)
    while p.poll() is None:
        if time.time()-started>1650:p.kill();p.wait();break
        f=out/'status.json'
        if f.exists():
            current=f.read_text()
            if current!=last:print(current,flush=True);last=current
        if time.time()-archived>90 and (out/'run.json').exists():archive();archived=time.time()
        time.sleep(5)
(out/'worker_result.json').write_text(json.dumps(dict(returncode=p.returncode,seconds=time.time()-started)))
archive();print('Worker exit:',p.returncode,flush=True)
if p.returncode:print((out/'console.log').read_text()[-6000:],flush=True)
