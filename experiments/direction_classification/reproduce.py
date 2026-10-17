"""Stage recovered imports and run the month classification pilot on CUDA."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--data',type=Path,default=root/'reports/pcgrad_mse_month_20261009/dataset.npz')
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
if args.out.exists() and any(args.out.iterdir()):parser.error('--out must be a fresh empty directory')
with tempfile.TemporaryDirectory(prefix='xau-classification-') as tmp:
    tmp=Path(tmp)
    for filename,source in {
        'capacity_train.py':Path(__file__).with_name('train.py'),
        'capacity_model.py':root/'experiments/pcgrad_capacity/model.py',
        'full_train.py':root/'experiments/loss_energy_full/train.py',
        'projection.py':root/'experiments/pcgrad_mse/projection.py',
    }.items():shutil.copyfile(source,tmp/filename)
    subprocess.run([sys.executable,str(tmp/'capacity_train.py'),'--data',str(args.data.resolve()),'--out',str(args.out.resolve())],check=True)
