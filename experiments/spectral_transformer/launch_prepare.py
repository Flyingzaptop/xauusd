import tarfile,subprocess,shutil
from pathlib import Path
with tarfile.open('/content/spectral-inputs.tar.gz') as t:t.extractall('/content',filter='data')
shutil.copy('/content/prepare_remote.py','/content/spectral/prepare_remote.py')
log=open('/content/spectral/prepare.log','w');p=subprocess.Popen(['python','-u','/content/spectral/prepare_remote.py'],stdout=log,stderr=subprocess.STDOUT,cwd='/content/spectral',start_new_session=True)
print({'prepare_pid':p.pid})
