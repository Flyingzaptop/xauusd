"""Prepare independent v3.1 selection/calibration streams; reuse v3 outer caches."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.market_data_v31 import SPLITS, prepare, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/market-v31.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)

    def progress(name: str, done: int, total: int) -> None:
        print(f"routing {name}: {done:,}/{total:,}", file=sys.stderr, flush=True)

    splits, audit = prepare(cfg, progress=progress)
    source_cache = Path(cfg.get("source_cache", "artifacts/market-v3"))
    for name in SPLITS:
        target = output / f"{name}.npz"
        if name in ("train", "evaluation"):
            shutil.copyfile(source_cache / f"{name}.npz", target)
            if sha256(target) != audit["source_v3"][f"{name}_cache_sha256"]:
                raise RuntimeError(f"Copied {name} differs from frozen v3 cache")
        else:
            split = splits[name]
            arrays = {key: value for key, value in split.items() if key != "tree"}
            arrays.update({f"tree_{key}": value for key, value in split["tree"].items()})
            np.savez_compressed(target, **arrays)
        print(f"saved {name}: {audit['splits'][name]['rows']:,} rows, "
              f"{audit['splits'][name]['supervised_rows']:,} supervised",
              file=sys.stderr, flush=True)
    audit["provenance"] = {"config_sha256": sha256(config_path),
                           "cache_sha256": {name: sha256(output / f"{name}.npz") for name in SPLITS}}
    public = ROOT / "reports/v31/data-preflight.json"
    public.parent.mkdir(parents=True, exist_ok=True)
    public.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {public}")


if __name__ == "__main__":
    main()
