# Reproduce the synthetic-only mechanism stage

Branch: `crossing-memory-mechanism-v2`.
Specification-before-implementation commit: `f2bc339f47b0d7d2022d880168982823f25ed954`.
Frozen training source/configuration: `104022f5cabe9dda532380604a6d86ea584c1080`.
V1 is preserved on `crossing-memory-v1` at `b160b4e523064c6c90990c22c60a5715c6a093aa`.

No market data is needed for this stage. Do not run `scripts/run_experiment.py`:
that is the preserved v1 market runner, not part of this stage. January–August
2026 is exposed and unavailable for a new untouched-test claim.

Use the recorded Python 3.14.3 environment and versions in `requirements.txt`.
Plotting additionally uses matplotlib 3.10.8 (`python -m pip install matplotlib==3.10.8`). Training is deterministic CPU PyTorch with one thread. The available GPU
is not required. Run from the repository root:

```powershell
python -m pytest -q
python scripts/diagnose_v1_synthetic.py
python scripts/run_mechanism_v2.py --task all
```

The runner always uses `configs/mechanism-v2.json` unless another config is
explicitly supplied. Individual fixed tasks can be reproduced with
`--task tiny_hard`, `easy_closed`, `medium_closed`, `hard_closed`, `hard_open`,
or `random_hard`. Do not change config or training source between tasks when
reproducing one frozen run. Reports contain the exact commit and SHA-256 of
executed source/config files and generated data. File-byte hashes can reflect
platform line endings; the Git commit is the cross-platform source identity.

Each completed fit writes its learning curve and metrics to the corresponding
`reports/v2/<task>.json`. `complete: false` means a partial task and must not be
reported as finished. There are 54 planned fits, all seeds 7/19/43 included.
`tiny_hard` deliberately evaluates its 16 training examples: its values establish
memorization only. Other train/validation/test path hashes are disjoint. Both
final and validation-selected test checkpoints are disclosed; neither is used
to change training. V1 reruns are new diagnostics and do not replace its old
negative results.

Selected and final weights are stored locally in
`C:/Users/r3d_flzp/Desktop/xauusd/artifacts/mechanism-v2/`, excluded from Git.
Every `.pt` file contains `best` and `final` state dictionaries. They can be
regenerated from synthetic inputs without access to external services. The
final report's artifact manifest lists their sizes and hashes.

Execution note for the recorded run: the initial sequential launcher completed
the tiny memorization fits. A task-boundary watcher stops that launcher when
the tiny report becomes complete; the five independent remaining tasks run as
separate `--task` processes using the same frozen source/configuration. They
overlap in wall time (including later tiny seeds), but share no model or optimizer
state and perform no curriculum transfer or selection from another task. Reported
durations therefore include CPU contention and are not isolated performance
benchmarks. Parallel execution changes scheduling, not samples, updates or seeds.

On the final report commit, postprocessing scripts plot the saved curves and
evaluate a labelled, nontraining hidden-state intervention on fixed selected
checkpoints. These scripts were added after training began, are not part of
model selection, and are documented separately in the final report. No new
market protocol or expansion is authorized before this stage's acceptance.
