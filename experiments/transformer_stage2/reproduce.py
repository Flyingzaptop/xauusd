"""CUDA reproduction of stage two only, using existing stage-one checkpoints."""
import argparse,shutil,subprocess,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[2];p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=root/'reports/transformer_stage2_20261012/dataset.npz');p.add_argument('--pretrained',type=Path,default=root/'reports/transformer_stage1_20261012/remote');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.out.exists() and any(a.out.iterdir()):p.error('--out must be empty')
with tempfile.TemporaryDirectory(prefix='xau-stage2-') as tmp:
    tmp=Path(tmp)
    for name in ['train.py','model.py','data.py','metrics.py']:shutil.copyfile(Path(__file__).with_name(name),tmp/name)
    shutil.copyfile(root/'experiments/transformer_stage1/model.py',tmp/'stage1_model.py');shutil.copyfile(root/'experiments/transformer_stage1/representation.py',tmp/'representation.py')
    subprocess.run([sys.executable,str(tmp/'train.py'),'--data',str(a.data.resolve()),'--pretrained',str(a.pretrained.resolve()),'--out',str(a.out.resolve())],check=True)
(a.out/'worker_result.json').write_text('{"returncode":0}\n')
