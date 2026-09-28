# XAUUSD

Research workspace for XAUUSD market data. Initialized on 2026-09-28; awaiting the first plan from Astra.

## Local data

- `data/M1/xauusd_m1.parquet`: minute data; schema and coverage not yet verified.
- `data/L2/l2_consolidated.parquet`: consolidated depth data. Its manifest declares 411,785,189 rows, covering 2026-03-23 through 2026-07-24 UTC. Source files currently live under `data/L2/cache/`; the manifest refers to older paths without `cache/`.
- `data/MBO/`: ten monthly compressed Databento DBN files. Metadata identifies GLBX.MDP3, MBO schema, GC.FUT (gold futures), spanning October 2025 through July 2026. These are futures data, not spot XAUUSD.

The initial inspection found 29 files totaling 30,321,871,720 bytes. Raw data and provider download manifests remain local and are excluded from Git. `docs/data-inventory.csv` records file paths and sizes without uploading their contents. No application code or tests existed at initialization. Data content, integrity and completeness have not yet been validated.

## Coordination

Follow Astra's assigned plan in order and execute authorized work automatically. Publish progress, validation evidence, blockers and completion reports in GitHub, with durable reports under `reports/`. Await the first assignment before starting research or implementation.

When delegation is needed, use only GPT-6 SOL for complex subtasks and GPT-6 LUNA for the simplest subtasks. Enable verbose development logging and keep logs out of Git.
