"""One CPU validation command; sequential to respect the two-core workspace quota."""
from pathlib import Path
import subprocess,sys,json,time
B=Path(__file__).parent;P=Path('/workspace/xauusd/reports/sequential_teacher_v4_20261016');P.mkdir(parents=True,exist_ok=True)
steps=[('unit',[sys.executable,'-m','unittest','discover','-s',str(B),'-p','test_*.py','-v']),('synthetic',[sys.executable,str(B/'run_gates.py')]),('meta',[sys.executable,str(B/'meta.py')])]+[(task,[sys.executable,str(B/'meta_gradient.py'),'--task',task]) for task in ['direction','wait','cross_market']]+[('execution',[sys.executable,str(B/'check_market_execution.py')]),('report',[sys.executable,str(B/'finalize_report.py')])]
if __name__=='__main__':
    start=time.time()
    for name,cmd in steps:
        with (P/(name+'_validation.log')).open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        if result.returncode:raise SystemExit(f'{name} failed; inspect {P}')
    print((P/'launch_decision.json').read_text());print('CPU validation completed in',round(time.time()-start),'seconds. No GPU allocated.')
