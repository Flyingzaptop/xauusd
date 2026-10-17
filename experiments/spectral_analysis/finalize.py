import json,hashlib,tarfile,re
from pathlib import Path
ROOT=Path('/workspace/xauusd');OUT=ROOT/'reports/spectral_analysis_20261012';CODE=ROOT/'experiments/spectral_analysis'
for link in re.findall(r'\]\((/workspace/[^)]+)\)',(OUT/'REPORT_RU.md').read_text()):assert Path(link).exists(),link
files=[p for p in OUT.iterdir() if p.is_file() and p.suffix!='.npy' and p.name!='MANIFEST.json']+[p for p in CODE.iterdir() if p.is_file()]
manifest=[]
for p in sorted(files):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 manifest.append({'file':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
(OUT/'MANIFEST.json').write_text(json.dumps({'files':manifest,'excluded_large_derived_files':['base.npy','spectral.npy'],'original_HDF5_in_separate_dataset':True},indent=2))
archive=Path('/workspace/xauusd-spectral-analysis-2021.tar.gz')
with tarfile.open(archive,'w:gz',compresslevel=3) as t:
 for p in sorted(files+[OUT/'MANIFEST.json']):t.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
with tarfile.open(archive) as t:
 for row in manifest:assert hashlib.sha256(t.extractfile(row['file']).read()).hexdigest()==row['sha256']
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'verified_files':len(manifest)}))
