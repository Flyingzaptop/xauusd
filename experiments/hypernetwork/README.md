# Conditional weights for XAUUSD direction classification

The model predicts SHORT / NONE / LONG for the terminal Close eight M1 bars ahead using only the preceding 512 bars. Thresholds are strictly below −$2 and above +$2. See [protocol.md](protocol.md) for the frozen comparison and limitations.

`model.py` contains a TCN and a small hypernetwork that generates a rank-four correction to one classifier layer. A wider ordinary head matches the parameter count within 0.03%. `train.py` trains both architectures with two seeds and ordinary CE + Adam. Precision-focused checkpoint and signal selection reuse the preceding experiment's policy.

On a CUDA machine with Python, NumPy and PyTorch installed:

```bash
python experiments/hypernetwork/reproduce.py --out reports/hypernetwork_reproduction/remote
```

Training writes best/last checkpoints, validation/test probabilities, per-epoch curves and gradient diagnostics. The reproduction output directory must be empty. The original dataset is included in the study archive. PyTorch 2.11.0+cu130 and an NVIDIA RTX PRO 6000 Blackwell Server Edition were used; small hardware/kernel differences can change threshold selection on this small validation set.

To audit the downloaded experiment and regenerate CPU plots:

```bash
python experiments/hypernetwork/report.py
```

The report auditor requires `worker_result.json` from the cloud wrapper in addition to the trainer's outputs. For a manual reproduction, create it with `{"returncode": 0}` only after the trainer exits successfully, then pass `--out reports/hypernetwork_reproduction` to the report script.

No API credentials are part of these files. The test period is a reused research holdout, not a final evaluation of a trading system.
