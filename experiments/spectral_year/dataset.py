"""Read day/block slices without loading the whole year into memory."""
import csv
import json
from pathlib import Path
import h5py
import numpy as np

METHODS=['stft_128','stft_512','stft_8192','wavelet_returns','wavelet_logprice']
FIELDS=['coefficients','power','phase']


class SpectralYear:
    def __init__(self,root):
        self.root=Path(root)
        self.metadata=json.loads((self.root/'dataset.json').read_text())
        with (self.root/'days.csv').open() as f:
            self.days={r['date']:r for r in csv.DictReader(f)}

    def _read(self,f,rows,methods,fields,period_range):
        output={'candles':{key:d[rows] for key,d in f['candles'].items()}}
        for method in methods:
            if method not in METHODS:raise ValueError(method)
            g=f[method];per=g['periods'][:]
            if period_range is None:cols=slice(None)
            else:
                selected=np.flatnonzero((per>=period_range[0])&(per<=period_range[1]))
                cols=slice(int(selected[0]),int(selected[-1])+1) if len(selected) else slice(0,0)
            ref=g['reference_power'][cols]
            result={'periods':per[cols],'reference_power':ref}
            for field in fields:
                if field in FIELDS:result[field]=g[field][rows,cols]
                elif field=='relative_power_db':
                    result[field]=10*np.log10(np.maximum(g['power'][rows,cols],1e-20)/ref[None,:])
                elif field=='phase_at_end':
                    phase=g['phase'][rows,cols]
                    if method.startswith('stft_'):
                        length=int(g.attrs['window_length']);phase=phase+2*np.pi*(length-1)/per[cols][None,:]
                    result[field]=(phase+np.pi)%(2*np.pi)-np.pi
                else:raise ValueError(field)
            output[method]=result
        return output

    def load_day(self,date,methods=None,fields=None,period_range=None):
        row=self.days[str(date)]
        with h5py.File(self.root/row['file'],'r') as f:
            return self._read(f,slice(int(row['row_start']),int(row['row_stop'])),
                              methods or METHODS,fields or FIELDS,period_range)

    def iter_blocks(self,block_size=256,methods=None,fields=None,period_range=None):
        if block_size<1:raise ValueError('block_size must be positive')
        for entry in self.metadata['months']:
            with h5py.File(self.root/entry['file'],'r') as f:
                for start in range(0,entry['rows'],block_size):
                    yield self._read(f,slice(start,min(entry['rows'],start+block_size)),
                                     methods or METHODS,fields or FIELDS,period_range)
