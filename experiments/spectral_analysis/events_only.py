import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np,json
from analyze import OUT,H,analyze_events
m=dict(np.load(OUT/'market.npz'));meta=json.loads((OUT/'features.json').read_text());x=np.column_stack([np.load(OUT/'base.npy'),np.load(OUT/'spectral.npy',mmap_mode='r')]);n=len(x);split=m['split'];masks=np.zeros((3,n,3),bool)
for s in range(3):
 for j,h in enumerate(H):
  same=np.zeros(n,bool);same[:-h]=split[:-h]==split[h:];masks[s,:,j]=(split==s)&m['valid'][:,j]&same
analyze_events(m,x,meta['base']+meta['spectral'],masks)
print('COMPLETE')
