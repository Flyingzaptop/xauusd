import hashlib,json,subprocess,tarfile,time
from pathlib import Path
cli='/workspace/colab-tools/bin/colab';out=Path('/workspace/xauusd/reports/transformer_stage2_20261012');archive=Path('/workspace/scratch/stage2-results.tar.gz')
expected=hashlib.sha256(Path('/workspace/xauusd/experiments/transformer_stage2/train.py').read_bytes()).hexdigest();last=None
while True:
    r=subprocess.run([cli,'download','-s','xauusd-stage2','/content/stage2-results.tar.gz',str(archive)],capture_output=True,text=True)
    if r.returncode==0:
        try:
            with tarfile.open(archive) as tar:
                run=json.load(tar.extractfile('remote/run.json'))
                assert run['source_sha256']['train.py']==expected
                final='remote/worker_result.json' in tar.getnames()
                tar.extractall(out,filter='data')
            status=json.loads((out/'remote/status.json').read_text())
            msg=dict(status=status,stages=run.get('stages',{}),trials=len(run.get('trials',{})))
            if msg!=last:print(json.dumps(msg),flush=True);last=msg
            if final:
                result=json.loads((out/'remote/worker_result.json').read_text());print('WORKER:',result,flush=True)
                if result['returncode']==0 and run['status']=='complete':
                    r=subprocess.run([cli,'stop','-s','xauusd-stage2'],capture_output=True,text=True);print(r.stdout,r.stderr,flush=True)
                else:print('Worker failed; GPU retained for immediate recovery.',flush=True)
                break
        except Exception as e:print('RETRY',type(e).__name__,str(e),flush=True)
    else:print('Snapshot not ready',flush=True)
    time.sleep(40)
