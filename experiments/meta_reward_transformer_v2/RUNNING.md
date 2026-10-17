# STOPPED / SUPERSEDED

This attempt lost its VM after preparation. Active retry: session meta-reward-v2-fg, experiments/meta_reward_transformer_v2_foreground/RUNNING.md. Do not start v2 detached launcher again.

# Active v2 — do not start duplicates

Authorised new run, October15. Session `meta-reward-v2`, A100-SXM4-80GB,179GBRAM,12CPU, rate6.77units/hour; initial balance120.54. Previous session v1 is stopped and artifacts retained in reports/meta_reward_transformer_20261015. v2 root remote `/content/meta_reward`, local reports/meta_reward_transformer_v2_20261015. New initial seeds; not a continuation of the prior reward curve.

Launch succeeded and CPU objective/data/feature checks + CUDA replay checks all passed. Supervisor PID2956; preparation worker2958. Downloads222sources, recalculates75GBfeatures, then automatically starts train.py. Preparation previously21min. All22248feature instruments used. Root environment.json/prepared.json/audit.json retain facts.

Changes and assumptions in PROTOCOL_RU.md. Adaptive actor entropy; teacher reward delta of net-profit minus trade PnL L2 concentration, on14calendar days, dev28fixed calendar days, test84fixed calendar days. Coefficients explicitly set, not mathematically inferred as unique. No trade count bonus or guaranteed edge. Full10y preparation is not a full10y epoch for every student.

Full resume_state every outer cycle. Remote supervisor archive every300sec at `/content/meta-reward-v2-results.tar.gz`. Local watchdog process in reports/watchdog_process.json; monitor state reports/live_status.json. Missing training_status during preparation is expected. Missing archive_ready before completion is expected. Do not mistake download status failures for training failure. Monitor failures alone do not delete runtime. Budget7h from watchdog start, independent of monitoring. Remote worker6.75hcap; local releasesGPU after verified final archive, calls localreport forplots. If interrupted inspect remote/results/error.txt and supervisor.log; do not retrain automatically without checking saved states.

Use normal Colab CLI oauth2 default (not --auth adc): `/workspace/colab-tools/bin/colab`. Read-only CLI downloads can coexist; avoid concurrent kernel execs. Code immutable after runtime source hashes snapshot. Local report/watchdog changes are operational only and must not be presented as executed training-source changes. Preserve prior checkpoints. Feature tensor not included in compact archive; reconstructable in~21min but not durably saved as75GB file.
