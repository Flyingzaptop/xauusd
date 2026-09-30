"""Exact immutable inputs for all v3.1 phases; hashing does not inspect labels."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch

SOURCE_FILES=['src/crossing_memory_v2.py','src/market_data_v3.py','src/stream_models_v3.py',
 'src/stream_batch_v3.py','scripts/run_market_v3.py','src/market_data_v31.py',
 'src/training_control_v31.py','src/calibration_v31.py','src/inference_v31.py',
 'src/provenance_v31.py','scripts/fit_market_v31.py','scripts/score_market_v31.py']


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def get_provenance(cfg):
    cache=Path(cfg['output'])
    audit=json.loads((cache/'audit.json').read_text())
    config_path=Path('configs/market-v31.json')
    if cfg!=json.loads(config_path.read_text()):raise RuntimeError('Configuration is not the frozen canonical file')
    config_hash=digest(config_path)
    cache_hashes={n:digest(cache/f'{n}.npz') for n in ['train','selection','calibration','evaluation']}
    data_hash=digest(cfg['data'])
    if (audit['provenance']['config_sha256']!=config_hash or
            audit['provenance']['cache_sha256']!=cache_hashes or
            audit['source_v3']['raw_data_sha256']!=data_hash):
        raise RuntimeError('Prepared cache, raw data or config differs from audited inputs')
    return dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        config_sha256=config_hash,source_sha256={p:digest(p) for p in SOURCE_FILES},
        data_sha256=data_hash,cache_sha256=cache_hashes,
        preprocessing_audit_sha256=digest(cache/'audit.json'),python=sys.version,torch=torch.__version__,numpy=np.__version__)
