import psutil
from pathlib import Path
for p in psutil.process_iter(['cmdline']):
 cmd=p.info['cmdline'] or []
 if len(cmd)>1 and cmd[1]=='/content/intermarket/prepare.py':
  p.terminate();p.wait(timeout=10);print('Stopped prep worker',p.pid)
p=Path('/content/intermarket/status.json');print(p.read_text() if p.exists() else 'not started')
