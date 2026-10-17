"""Run the recovered experiment on a CUDA host without import-path surgery."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smaller', action='store_true',
                        help='Run the requested extension: widths 13, 8, 5, 3, 2')
    parser.add_argument('--reference', type=Path,
                        default=root / 'reports/pcgrad_mse_month_20261009/remote/run.json')
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('--out must be empty: use a fresh directory for this experiment')
    with tempfile.TemporaryDirectory(prefix='xau-capacity-') as staging:
        staging = Path(staging)
        sources = {
            'full_train.py': root / 'experiments/loss_energy_full/train.py',
            'pcgrad_train.py': root / 'experiments/pcgrad_mse/train.py',
            'projection.py': root / 'experiments/pcgrad_mse/projection.py',
            'capacity_model.py': Path(__file__).with_name('model.py'),
            'capacity_train.py': Path(__file__).with_name('train_smaller.py' if args.smaller else 'train.py'),
        }
        for target, source in sources.items():
            shutil.copyfile(source, staging / target)
        subprocess.run([sys.executable, str(staging / 'capacity_train.py'),
                        '--data', str(args.data.resolve()),
                        '--out', str(args.out.resolve()),
                        '--reference', str(args.reference.resolve())], check=True)


if __name__ == '__main__':
    main()
