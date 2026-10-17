from pathlib import Path
import subprocess
r=Path('/content/intermarket');p=subprocess.Popen(['python','/content/intermarket/prepare.py'],stdout=open(r/'prepare.log','a'),stderr=subprocess.STDOUT,start_new_session=True);print('prep pid',p.pid)
