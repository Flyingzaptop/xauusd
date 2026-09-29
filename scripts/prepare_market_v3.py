"""Save continuous M1 split arrays and train-only preprocessing audit for v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.market_data_v3 import prepare


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/market-v3.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)

    def progress(name: str, done: int, total: int) -> None:
        print(f"routing {name}: {done:,}/{total:,}", file=sys.stderr, flush=True)

    splits, audit = prepare(cfg, progress=progress)
    audit["provenance"] = {
        "data_sha256": sha256(Path(cfg["data"])),
        "config_sha256": sha256(config_path),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    for name, split in splits.items():
        arrays = {key: value for key, value in split.items() if key != "tree"}
        arrays.update({f"tree_{key}": value for key, value in split["tree"].items()})
        np.savez_compressed(output / f"{name}.npz", **arrays)
        print(f"saved {name}: {len(split['timestamp']):,} rows, {int(split['valid'].sum()):,} supervised",
              file=sys.stderr, flush=True)
    report = ROOT / "reports/v3/data-preflight.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
