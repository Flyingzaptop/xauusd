# Causal yearly spectral preparation

Data preparation only. No pictures, model training, forecasts, market-pattern
analysis, significance tests, signal selection or trading simulation.

Year: 2021, every observed M1 candle. All gaps are removed from the transform
coordinate; actual timestamps and gap lengths remain available. Context starts
in August 2020. Fixed references use December 2020 only.

STFT: trailing Hann windows 128, 512 and 8192, one result per candle. All FFT
bins in the existing period range >=4 are retained. Windows 128/512 exclude
k=1 as in the day example; the new 8192 window includes k=1 to reach a nominal
period of 8192. The longest bin contains only one cycle per window.

Wavelets: one-sided DC-corrected, L2-normalized complex Morlet bank. The first
48 periods 4–512 exactly match the prior day example; 27 periods extend to
8192. This is a causal filter bank, not centered CWT. Its largest finite kernel
has 65537 coefficients; its full history is available before calibration.

Output: 12 losslessly compressed HDF5 files, time-major arrays. Complex
coefficients are complex128, power and phase are float64. Each month contains
its own axes and fixed pre-year references. `days.csv` indexes all 365 dates,
including dates with no candles. See the generated SCHEMA.md and INDEX.md.

Reproduce from the archived selected input snapshot:

```bash
pip install -r experiments/spectral_year/requirements.txt
python experiments/spectral_year/prepare.py \
  --source reports/causal_spectral_year_2021_20261012/input_context.npz \
  --out reports/year_reproduction
python experiments/spectral_year/verify.py --out reports/year_reproduction
```

Optional compatibility audit against the earlier one-day example:

```bash
python experiments/spectral_year/verify.py --out reports/year_reproduction \
  --previous-day reports/causal_spectral_day_20261012/features.npz
```

The source input snapshot includes only raw timestamp, Close and original row
ID. It contains no previously calculated FFT predictions or labels. SHA256
of the recovered ten-year source is recorded in input_provenance.json.

Read one day without loading the whole year:

```python
import sys
sys.path.insert(0, 'experiments/spectral_year')
from dataset import SpectralYear

data = SpectralYear('reports/causal_spectral_year_2021_20261012')
day = data.load_day(
    '2021-06-16',
    methods=['stft_8192', 'wavelet_returns'],
    fields=['coefficients', 'power', 'phase', 'relative_power_db'],
    period_range=(32, 8192),
)
timestamps = day['candles']['available_timestamp']
coefficients = day['wavelet_returns']['coefficients']
```

For sequential processing use `iter_blocks(block_size=256, ...)`. Derived
relative_power_db and phase_at_end are computed on demand; the original
coefficient, power and raw phase remain available. Do not load the entire
multi-gigabyte year into one array. Files are ready for later authorized
analysis; this preparation does not make predictions.
