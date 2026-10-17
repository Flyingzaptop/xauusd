# V3 — 2026-10-16
Session: meta-reward-v3. NVIDIA A100-SXM4-80GB, 167 GiB host RAM.
Foreground CLI and watchdog PIDs/start time: watchdog_process.json.
Training source: /workspace/xauusd/experiments/meta_reward_transformer_v3; protocol PROTOCOL_RU.md.
Warm start: V2 teacher cycle80; reset optimizer. Backup links/hash: v2_drive_backup.json.
Changes: direction-centered reward, up/down paired meta windows, accumulated four-cycle teacher updates, lower learning rates, monotone entropy coefficient schedule.
Evaluation: four 28-day development windows in 2024; new 84-day test April7–June30 2025; fresh student seeds888/889 for updated teacher / old teacher / direct money.
Budget: primary3h; remote supervisor6.75h; local watchdog7h total cap; automatic final verified download and GPU release.
Monitoring: foreground_console.log, live_status.json, watchdog.log; extracted snapshots remote/. Compressed duplicate removed only after full archive stream/source-hash verification.
Do not start another GPU session while this one is active. Do not replace foreground execution with a detached remote subprocess.
Resume state preserves teacher gradients (partial four-cycle accumulation), optimizer states, reward calibration, all student states, order/cursor and RNG. Recovery requires restoring those fields, not merely loading teacher weights.
