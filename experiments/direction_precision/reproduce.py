"""Run the fixed precision study on a CUDA machine."""
import argparse
from pathlib import Path
import shutil,subprocess,sys,tempfile
root=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser()
ap.add_argument('--data',type=Path,default=root/'reports/pcgrad_mse_month_20261009/dataset.npz')
ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args()
if args.out.exists() and any(args.out.iterdir()):ap.error('--out must be empty')
with tempfile.TemporaryDirectory(prefix='xau-precision-') as tmp:
    tmp=Path(tmp)
    files={'capacity_train.py':'experiments/direction_precision/train.py',
           'policy.py':'experiments/direction_precision/policy.py',
           'classification_base.py':'experiments/direction_classification/train.py',
           'capacity_model.py':'experiments/pcgrad_capacity/model.py',
           'full_train.py':'experiments/loss_energy_full/train.py',
           'projection.py':'experiments/pcgrad_mse/projection.py'}
    for name,source in files.items():shutil.copyfile(root/source,tmp/name)
    subprocess.run([sys.executable,str(tmp/'capacity_train.py'),'--data',str(args.data.resolve()),'--out',str(args.out.resolve())],check=True)
