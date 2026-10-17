"""Merge two completed segments, preserving their original provenance reports."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

root = Path(__file__).resolve().parents[2]
dest = root / 'reports/pcgrad_capacity_20261009/remote'
src = root / 'reports/pcgrad_capacity_smaller_20261009/remote'
original = dest / 'run_initial.json'
if not original.exists():
    shutil.copyfile(dest / 'run.json', original)
a = json.loads(original.read_text())
b = json.loads((src / 'run.json').read_text())
assert a['status'] == b['status'] == 'complete'
assert a['dataset_sha256'] == b['dataset_sha256']
assert a['settings'] == b['settings']
assert not (a['trials'].keys() & b['trials'].keys())
for source, filename in [(a, 'train.py'), (b, 'train_smaller.py')]:
    assert source['source_sha256'] == hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()
for key in a['metrics']['bridge']:
    assert np.isclose(a['metrics']['bridge'][key], b['metrics']['bridge'][key])
for path in src.iterdir():
    if path.name.startswith('w') and path.name != 'worker_result.json':
        shutil.copyfile(path, dest / path.name)
shutil.copyfile(src / 'run.json', dest / 'run_smaller.json')
merged = dict(a)
for key in ['stages', 'trials', 'metrics']:
    merged[key] = {**a[key], **b[key]}
merged['seconds'] = a['seconds'] + b['seconds']
merged['segments'] = [dict(report='run_initial.json', source='train.py'),
                      dict(report='run_smaller.json', source='train_smaller.py')]
merged['derived_merged_report'] = True
merged['decisions_frozen_before_test'] = False
merged['all_segments_frozen_before_own_test'] = True
merged['selection_used_test'] = False
merged['stop_reason'] = b['stop_reason']
merged['selected_width'] = {
    mode: min(merged['stages'], key=lambda w: np.mean([
        merged['trials'][f'w{w}_{mode}_{s}']['best_validation'] for s in [17, 29]]))
    for mode in ['adam', 'pcgrad']}
(dest / 'run.json').write_text(json.dumps(merged, indent=2))
print('Merged', len(merged['trials']), 'trials; selected by validation:', merged['selected_width'])
