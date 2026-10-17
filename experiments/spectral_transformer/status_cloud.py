from pathlib import Path
import json
p=Path('/content/spectral');f=p/'results/status.json'
print(f.read_text() if f.exists() else (p/'prepare.log').read_text()[-2500:] if (p/'prepare.log').exists() else 'not prepared')
if (p/'prepared.json').exists():
 a=json.loads((p/'prepared.json').read_text());print({k:v for k,v in a.items() if k not in ['base_names','extra_names']})
if (p/'results/failure.json').exists():print((p/'results/failure.json').read_text())
