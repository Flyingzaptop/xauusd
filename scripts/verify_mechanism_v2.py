"""Verify saved reports/provenance and preservation, without fitting models."""
import csv
import hashlib
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

from probe_memory_v2 import verify_source_reports


def main():
    cfg=json.loads(Path('configs/mechanism-v2.json').read_text())
    provenance=verify_source_reports(cfg,Path('reports/v2'),Path('artifacts/mechanism-v2'))
    runs=[]
    for task in cfg['tasks']:
        report=json.loads(Path(f'reports/v2/{task}.json').read_text())
        for run in report['runs']:
            assert len(run['curve'])==run['epochs']
            assert run['finite_gradients']
            for epoch in run['curve']:
                assert all(math.isfinite(v) for split in ['train','validation'] for v in epoch[split].values())
            for split in ['final_train','final_test','selected_test']:
                assert all(math.isfinite(v) for v in run[split].values())
            if run['variant']=='tree':
                for group in ['memory.observe.','memory.merge.','memory.readout.']:
                    assert sum(v for k,v in run['parameter_change_l2'].items() if k.startswith(group))>0
        runs.extend(report['runs'])
    assert len(runs)==54
    assert sum(r['optimizer_updates'] for r in runs)==6300
    for name,expected in [('seed_metrics',54),('aggregate_summary',18)]:
        with open(f'reports/v2/plots/{name}.csv',newline='',encoding='utf-8') as f:
            assert len(list(csv.DictReader(f)))==expected
    for task in cfg['tasks']:
        for suffix in ['png','svg']:
            assert Path(f'reports/v2/plots/{task}.{suffix}').stat().st_size>1000
    protected=['src/crossing_memory.py','scripts/run_experiment.py','scripts/synthetic_checks.py',
               'configs/experiment.json','reports/market_metrics.json','reports/market_metrics.csv',
               'reports/synthetic_metrics.json','reports/crossing-memory-v1.md','docs/operator.md','docs/experiment-protocol.md']
    subprocess.run(['git','diff','--exit-code','b160b4e523064c6c90990c22c60a5715c6a093aa','--',*protected],check=True)
    old_manifest=json.loads(Path('reports/artifact-manifest.json').read_text())
    for artifact in old_manifest['artifacts']:
        assert hashlib.sha256(Path(artifact['path']).read_bytes()).hexdigest()==artifact['sha256']
    junit=Path('artifacts/mechanism-v2/unit-tests.xml')
    subprocess.run([sys.executable,'-m','pytest','-q',f'--junitxml={junit}'],check=True)
    suites=list(ET.parse(junit).getroot().iter('testsuite'))
    test_count=sum(int(s.attrib['tests']) for s in suites)
    assert all(int(s.attrib.get('failures',0))==0 and int(s.attrib.get('errors',0))==0 for s in suites)
    result=dict(training_commit=provenance['commit'],source_and_checkpoint_provenance_verified=True,
                completed_fits=54,optimizer_updates=6300,all_recorded_metrics_and_gradients_finite=True,
                full_tree_observe_merge_readout_parameters_changed=True,unit_tests_passed=test_count,
                unit_test_command='python -m pytest -q',v1_protected_files_unchanged=protected,
                v1_local_artifacts_unchanged=len(old_manifest['artifacts']),market_training_run=False,
                plot_version=matplotlib.__version__,plot_files=12,per_seed_csv_rows=54,aggregate_csv_rows=18,
                execution_note='Sequential all-task launcher intentionally stopped after nine completed tiny fits; five independent --task processes completed remaining fits. No training source or configuration changed.')
    Path('reports/v2/verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
