"""Nonblocking local snapshot: Colab worker keeps running after tool returns."""
from pathlib import Path
import tarfile,os,json
p=Path('/content/spectral');r=p/'results'
if (p/'prepared.json').exists():print((p/'prepared.json').read_text()[:500])
print((r/'status.json').read_text() if (r/'status.json').exists() else (p/'prepare.log').read_text()[-1800:])
if (r/'failure.json').exists():print((r/'failure.json').read_text())
with tarfile.open('/content/spectral-snapshot.tmp.tar.gz','w:gz',compresslevel=1) as t:
 if r.exists():
  for f in r.iterdir():
   if f.is_file() and f.suffix!='.tmp':t.add(f,arcname='remote/'+f.name)
 for n in ['prepared.json','prepare.log','training.log']:
  if (p/n).exists():t.add(p/n,arcname='remote/'+n)
os.replace('/content/spectral-snapshot.tmp.tar.gz','/content/spectral-snapshot.tar.gz')
