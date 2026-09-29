"""Post-hoc, no-fit probe of learned state in frozen v2 tree checkpoints.

Runs only after all 54 preregistered source fits are complete. On each selected
test split, compares the saved best model with the same model whose readout
inputs have their hidden-state channels zeroed. Routing metadata remains intact.
This is descriptive ablation of already selected weights, never model selection.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_mechanism_v2 import Classifier, datasets, evaluate


TASKS = ("easy_closed", "medium_closed", "hard_closed", "hard_open")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_reports(cfg: dict, report_dir: Path, artifact_dir: Path) -> dict:
    """Require the whole frozen source experiment before opening any checkpoint."""
    expected_total = sum(len(variants) * len(cfg["seeds"]) for variants in cfg["tasks"].values())
    if expected_total != 54:
        raise RuntimeError(f"Expected the preregistered 54 fits, configuration now specifies {expected_total}")
    common_provenance = None
    for task, variants in cfg["tasks"].items():
        report_path = report_dir / f"{task}.json"
        if not report_path.is_file():
            raise RuntimeError(f"Source fit report missing: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected = {(variant, seed) for seed in cfg["seeds"] for variant in variants}
        actual = [(run["variant"], run["seed"]) for run in report["runs"]]
        if not report.get("complete") or len(actual) != len(expected) or set(actual) != expected:
            raise RuntimeError(f"Source fits incomplete or duplicated for {task}")
        if report["config"] != cfg:
            raise RuntimeError(f"Source configuration differs in {report_path}")
        provenance = report["provenance"]
        if common_provenance is None:
            common_provenance = provenance
        elif provenance != common_provenance:
            raise RuntimeError("Source fit reports have inconsistent provenance")
        for variant, seed in expected:
            checkpoint = artifact_dir / f"{task}-{variant}-{seed}.pt"
            if not checkpoint.is_file():
                raise RuntimeError(f"Source checkpoint missing: {checkpoint}")
    for relative, recorded_hash in common_provenance["files_sha256"].items():
        source = ROOT / relative
        if not source.is_file() or sha256(source) != recorded_hash:
            raise RuntimeError(f"Frozen source differs from recorded SHA: {source}")
    return common_provenance


def zero_hidden_readout(model: Classifier):
    """Install reversible hooks affecting only h, leaving metadata untouched."""
    hidden_dim = model.memory.hidden_dim
    handles = []

    def zero_hidden(_module, inputs):
        readout_input = inputs[0].clone()
        readout_input[:, :hidden_dim] = 0
        return (readout_input,)

    for readout in model.memory.readout:
        handles.append(readout.register_forward_pre_hook(zero_hidden))
    return handles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mechanism-v2.json")
    parser.add_argument("--reports", default="reports/v2")
    parser.add_argument("--artifacts", default="artifacts/mechanism-v2")
    parser.add_argument("--output", default="reports/v2/hidden-state-probe.json")
    args = parser.parse_args()
    config_path = ROOT / args.config
    report_dir = ROOT / args.reports
    artifact_dir = ROOT / args.artifacts
    output_path = ROOT / args.output
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    source = verify_source_reports(cfg, report_dir, artifact_dir)
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)

    runs = []
    for task in TASKS:
        source_report = json.loads((report_dir / f"{task}.json").read_text(encoding="utf-8"))
        source_runs = {(run["variant"], run["seed"]): run for run in source_report["runs"]}
        for seed in cfg["seeds"]:
            data, counts = datasets(task, seed, cfg)
            selected = source_runs[("tree", seed)]
            if counts["test"]["sha256"] != next(
                entry["splits"]["test"]["sha256"]
                for entry in source_report["dataset"] if entry["seed"] == seed
            ):
                raise RuntimeError(f"Regenerated test data differs for {task}/{seed}")
            checkpoint = artifact_dir / f"{task}-tree-{seed}.pt"
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model = Classifier("tree", cfg)
            model.load_state_dict(saved["best"], strict=True)
            baseline = evaluate(model, data["test"])
            for metric in ("loss", "accuracy", "balanced_accuracy"):
                if abs(baseline[metric] - selected["selected_test"][metric]) > 1e-6:
                    raise RuntimeError(f"Saved selected-test metric mismatch: {task}/{seed}/{metric}")
            handles = zero_hidden_readout(model)
            try:
                zero_hidden = evaluate(model, data["test"])
            finally:
                for handle in handles:
                    handle.remove()
            runs.append(
                {
                    "task": task,
                    "seed": seed,
                    "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
                    "checkpoint_sha256": sha256(checkpoint),
                    "dataset_test_sha256": counts["test"]["sha256"],
                    "selected_epoch": selected["best_epoch"],
                    "baseline": baseline,
                    "zero_hidden_at_readout": zero_hidden,
                    "delta_balanced_accuracy": zero_hidden["balanced_accuracy"] - baseline["balanced_accuracy"],
                    "delta_loss": zero_hidden["loss"] - baseline["loss"],
                }
            )
    result = {
        "type": "post_hoc_descriptive_no_fit_no_selection",
        "meaning": "For a fixed selected tree checkpoint, zero only hidden-state inputs to each readout; retain routing metadata, readout weights and classifier weights.",
        "source_provenance": source,
        "config_sha256": sha256(config_path),
        "probe_script_sha256": sha256(Path(__file__)),
        "trained_parameter_updates": 0,
        "splits_evaluated": ["test"],
        "runs": runs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps({"output": str(output_path), "runs": len(runs)}))


if __name__ == "__main__":
    main()
