"""Run the fixed experiment on a CUDA host; use an empty output directory."""
import argparse,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=root/'reports/multiscale_curriculum_20261010/dataset.npz');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.out.exists() and any(a.out.iterdir()):p.error('--out must be empty')
subprocess.run([sys.executable,str(Path(__file__).with_name('train.py')),'--data',str(a.data.resolve()),'--out',str(a.out.resolve())],check=True)
(a.out/'worker_result.json').write_text('{"returncode":0}\n')
