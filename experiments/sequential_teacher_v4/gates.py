"""Fail closed: expensive runs require actual evidence, matching code, untouched test."""
from pathlib import Path
import json,hashlib
from provenance import current

def source_hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}

def assess(report_dir,stage='real_pilot'):
    p=Path(report_dir);required=['mechanics','synthetic','meta_gradient'] if stage=='exact_pilot' else ['mechanics','synthetic','meta'] if stage=='real_pilot' else ['mechanics','synthetic','meta','real_pilot']
    issues=[];manifest=p/'gate_manifest.json'
    if not manifest.exists():return {'allowed':False,'reasons':['No content-hashed evidence manifest'],'stage':stage}
    m=json.loads(manifest.read_text())
    if m.get('source_hashes')!=source_hashes():issues.append('Code changed after evidence was recorded; revalidate affected gates')
    for k in required:
        item=m.get(k,{})
        if item.get('passed') is not True:issues.append(k+' gate has not passed')
        f=p/item.get('file','__missing__')
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=item.get('sha256'):issues.append(k+' evidence missing or changed')
        elif not current(json.loads(f.read_text())):issues.append(k+' results do not match current training code')
    if stage=='scale_up' and not m.get('untouched_test_protocol'):issues.append('No untouched test protocol')
    return {'allowed':not issues,'reasons':issues,'stage':stage}

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['real_pilot','exact_pilot','scale_up'],default='real_pilot');a.add_argument('--reports',default='/workspace/xauusd/reports/sequential_teacher_v4_20261016');args=a.parse_args()
    r=assess(args.reports,args.stage);print(json.dumps(r,indent=2));raise SystemExit(0 if r['allowed'] else 2)
