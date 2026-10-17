from pathlib import Path
import subprocess,json
r=Path('/content/intermarket')
worker=r/'training_supervisor.py'
worker.write_text('''from pathlib import Path
import json,time,subprocess
r=Path('/content/intermarket')
while not (r/'prepared.json').exists():
 s=json.loads((r/'status.json').read_text())
 if s.get('phase')=='failed':raise RuntimeError('Preparation failed; training not started')
 time.sleep(5)
p=subprocess.run(['python','/content/intermarket/train.py'],stdout=open(r/'training.log','w'),stderr=subprocess.STDOUT,timeout=7200)
(r/'training_result.json').write_text(json.dumps({'returncode':p.returncode}))
''')
p=subprocess.Popen(['python',str(worker)],stdout=open(r/'supervisor.log','w'),stderr=subprocess.STDOUT,start_new_session=True);print('Training supervisor',p.pid)
