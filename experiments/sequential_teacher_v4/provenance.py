from pathlib import Path
import hashlib
ROOT=Path(__file__).parent
COMMON=['environment.py','learning.py','model.py','synthetic.py']
def fingerprint(names):
    if any(Path(n).name!=n or not n.endswith('.py') for n in names):raise ValueError('Only local Python source names allowed')
    return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
def current(evidence):
    hashes=evidence.get('code_hashes',{})
    return bool(hashes) and fingerprint(hashes)==hashes
