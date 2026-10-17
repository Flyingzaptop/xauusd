import tarfile,subprocess
from pathlib import Path
p=Path('/content/spectral')
with tarfile.open('/content/spectral-code.tar.gz') as t:t.extractall(p,filter='data')
log=open(p/'supervisor.log','w');worker=subprocess.Popen(['python','-u',str(p/'supervise.py')],stdout=log,stderr=subprocess.STDOUT,cwd=p,start_new_session=True)
print({'supervisor_pid':worker.pid},flush=True)
