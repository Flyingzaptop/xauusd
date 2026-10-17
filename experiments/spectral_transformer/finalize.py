import json,hashlib,tarfile,re,sys,importlib.metadata
from pathlib import Path
ROOT=Path('/workspace/xauusd');P=ROOT/'reports/spectral_transformer_20261013';CODE=ROOT/'experiments/spectral_transformer';R=P/'remote'
assert json.loads((R/'complete.json').read_text())['complete'];assert json.loads((R/'worker_result.json').read_text())['returncode']==0;assert json.loads((P/'evaluation_checks.json').read_text())['passed'];assert json.loads((P/'local_checks.json').read_text())['passed']
for link in re.findall(r'\]\((/workspace/[^)]+)\)',(P/'REPORT_RU.md').read_text()):assert Path(link).exists(),link
for kind in ['price','raw','engineered']:
 curve=json.loads((R/f'{kind}_final_curve.json').read_text());assert all(r['examples']==265021 for r in curve)
run=json.loads((R/'run.json').read_text());assert run['source_sha256']['train.py']==hashlib.sha256((CODE/'train.py').read_bytes()).hexdigest()
(P/'READY.json').write_text(json.dumps({'complete':True,'year':2021,'all_year_candles':354340,'components':2356,'engineered_features':23072,'final_train_rows':265021,'test_rows':89079,'development':'Jan-Sep search/selection/full refit','final_evaluation':'Oct-Dec, already previously inspected in prior research','all_training_epochs_full_chronological_pass':True,'checkpoints_saved':True,'no_future_features':True,'gpu':'A10080GB','gpu_stopped_after_sync':True},indent=2))
files=[p for p in P.rglob('*') if p.is_file() and p.name!='MANIFEST.json']+[p for p in CODE.iterdir() if p.is_file()];rows=[]
for p in sorted(files):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 rows.append({'file':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
(P/'MANIFEST.json').write_text(json.dumps({'files':rows,'excluded_transient_gpu_feature_matrix':'reproducible_from_included_input_reference_code','original_HDF5_preserved_separately':True},indent=2))
archive=Path('/workspace/xauusd-spectral-transformer-study.tar.gz')
with tarfile.open(archive,'w:gz',compresslevel=1) as t:
 for p in sorted(files+[P/'MANIFEST.json']):t.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
with tarfile.open(archive) as t:
 for row in rows:assert hashlib.sha256(t.extractfile(row['file']).read()).hexdigest()==row['sha256']
print({'archive':str(archive),'bytes':archive.stat().st_size,'verified_files':len(rows)})
