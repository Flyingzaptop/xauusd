# XAUUSD

## Crossing Memory v2: mechanism-only review

Branch `crossing-memory-mechanism-v2` adds a temporal event tree, symmetric routing,
exactly-once message scheduling and streaming router/neural state. This stage
uses only generated data; no new market training is performed.

- [V2 acceptance report](reports/v2/acceptance-report.md) and [all-seed metrics](reports/v2/plots/seed_metrics.csv).
- [Specification and hand traces](docs/operator-v2.md), [invariant argument](docs/mechanism-v2-invariants.md), and [v1 synthetic diagnosis](docs/v1-synthetic-diagnosis.md).
- [Frozen synthetic protocol](docs/mechanism-v2-protocol.md) and [reproduction](docs/mechanism-v2-reproduction.md).
- [Learning curves](reports/v2/plots/) and [fixed-checkpoint hidden-state probe](reports/v2/hidden-state-probe.json).

V1 source and negative results below are preserved. January–August 2026 is an
exposed market test and cannot support a new untouched-test claim.

Research workspace for XAUUSD market data. The first bounded Crossing Memory experiment is implemented on branch `crossing-memory-v1` for review by Astra.

## Crossing Memory v1

- [Acceptance report](reports/crossing-memory-v1.md): full results, negative findings and limitations.
- [Frozen experiment protocol](docs/experiment-protocol.md) and [operator specification](docs/operator.md).
- [M1 audit](docs/data-audit.md), [reproduction commands](docs/reproduction.md) and [synthetic protocol](docs/synthetic.md).
- Machine-readable [market metrics](reports/market_metrics.json), [CSV](reports/market_metrics.csv), [synthetic metrics](reports/synthetic_metrics.json), and [local artifact manifest](reports/artifact-manifest.json).

Run checks with `python -m pytest -q`. This stage uses only M1; predictive scores are not a claim of trading profitability. The initial file inventory below describes the workspace at bootstrap.

## Local data

- `data/M1/xauusd_m1.parquet`: minute data; schema and coverage not yet verified.
- `data/L2/l2_consolidated.parquet`: consolidated depth data. Its manifest declares 411,785,189 rows, covering 2026-03-23 through 2026-07-24 UTC. Source files currently live under `data/L2/cache/`; the manifest refers to older paths without `cache/`.
- `data/MBO/`: ten monthly compressed Databento DBN files. Metadata identifies GLBX.MDP3, MBO schema, GC.FUT (gold futures), spanning October 2025 through July 2026. These are futures data, not spot XAUUSD.

The initial inspection found 29 files totaling 30,321,871,720 bytes. Raw data and provider download manifests remain local and are excluded from Git. `docs/data-inventory.csv` records file paths and sizes without uploading their contents. No application code or tests existed at initialization. Data content, integrity and completeness have not yet been validated.

## Coordination

Follow Astra's assigned plan in order and execute authorized work automatically. Publish progress, validation evidence, blockers and completion reports in GitHub, with durable reports under `reports/`. Await the first assignment before starting research or implementation.

When delegation is needed, use only GPT-6 SOL for complex subtasks and GPT-6 LUNA for the simplest subtasks. Enable verbose development logging and keep logs out of Git.
