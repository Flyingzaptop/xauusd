"""Run the bounded comparison, independent verification and HTML report automatically."""
import subprocess,sys,os,json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'reports/angular_reconstruction_v2_20261008'

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    programs=[]
    if not (OUT/'diagnostic/results.json').exists():programs.append('diagnose.py')
    programs+=['train.py','verify.py','report.py']
    for program in programs:
        result=subprocess.run([sys.executable,'-u',str(HERE/program)],cwd=ROOT,creationflags=flags)
        if result.returncode:
            OUT.mkdir(parents=True,exist_ok=True)
            (OUT/'pipeline_failure.json').write_text(json.dumps(dict(program=program,exit_code=result.returncode),indent=2),encoding='utf-8')
            raise SystemExit(result.returncode)

if __name__=='__main__':main()
