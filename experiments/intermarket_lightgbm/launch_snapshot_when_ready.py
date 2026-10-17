from pathlib import Path
import subprocess
r=Path('/content/intermarket');w=r/'snapshot_supervisor.py'
w.write_text('''from pathlib import Path
import subprocess,time,json,tarfile,hashlib
r=Path('/content/intermarket')
while not (r/'training_result.json').exists():time.sleep(5)
assert json.loads((r/'training_result.json').read_text())['returncode']==0
assert json.loads((r/'results/complete.json').read_text())['complete']
p=subprocess.run(['python',str(r/'patterns.py')],stdout=open(r/'patterns.log','w'),stderr=subprocess.STDOUT)
assert p.returncode==0, 'Pattern extraction failed; inspect patterns.log'
files=[p for p in (r/'results').iterdir() if p.is_file()]
files += [r/f for f in ['prepared.json','audit.json','inventory.json','environment.json','training.log','prepare.log','train_status.json','training_result.json','status.json']]
rows=[{'file':str(p.relative_to(r)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
(r/'snapshot_manifest.json').write_text(json.dumps(rows,indent=2))
archive=Path('/content/intermarket-snapshot.tar.gz')
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
 for p in files+[r/'snapshot_manifest.json']:tar.add(p,arcname=str(p.relative_to(r)),recursive=False)
(r/'snapshot_ready.json').write_text(json.dumps({'complete':True,'bytes':archive.stat().st_size,'files':len(files)}))
print((r/'snapshot_ready.json').read_text())
''')
p=subprocess.Popen(['python',str(w)],stdout=open(r/'snapshot_supervisor.log','w'),stderr=subprocess.STDOUT,start_new_session=True);print('Snapshot supervisor',p.pid)
