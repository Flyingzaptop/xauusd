"""Post-run descriptive routing counts on train only; no model selection."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_experiment import prepare


cfg = json.loads(Path('configs/experiment.json').read_text())
data, prep = prepare(cfg)
stats, closures = data['train']['stats'], data['train']['closures']
active = stats[..., 0] != 0
transfers = []
for scale in range(2):
    child = closures[:, :, scale]
    parent_active = active[:, :, scale+1]
    transfers.append(dict(child_scale=scale, parent_scale=scale+1,
                          child_closures=int(child.sum()),
                          accepted_by_active_parent=int((child & parent_active).sum()),
                          discarded_by_idle_parent=int((child & ~parent_active).sum())))
result = dict(scope='Post-evaluation descriptive diagnostics; train windows only; no tuning',
              thresholds=prep['thresholds'], windows=len(stats), observations=stats.shape[0]*stats.shape[1],
              active_fraction=active.mean((0,1)).tolist(),
              closure_count=closures.sum((0,1)).tolist(),
              open_at_final_readout_fraction=(active[:, -1] & ~closures[:, -1]).mean(0).tolist(),
              transfers=transfers)
Path('reports/routing-diagnostics.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
