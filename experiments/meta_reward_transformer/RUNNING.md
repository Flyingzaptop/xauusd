# STOPPED — previous run v1

Session lost on 2026-10-15. Checkpoints and 96-cycle logs retained locally. New experiment uses session meta-reward-v2 and directory experiments/meta_reward_transformer_v2; do not restart v1 automatically.

# Active run — read before starting another GPU job

User authorised full preparation + training. Do NOT start another copy or reset the current runtime just because a new chat turn says «продолжай».

- Colab session: `meta-reward`, A100-SXM4-80GB,12CPU,179GBRAM. G4/H100 requests rejected by backend. Rate6.77units/hour; startbalance138.87.
- Remote root `/content/meta_reward`. All222 files48causalfeatures prepared,3,505,124M1anchors. `features.f16` shape222×3505124×48,74,701,202,688bytes;8clockchannels separate. Preparation1259sec. GPUcache2016–2024:62,152,377,408bytes. Finaltest not in cache.
- Actual immutable train.py hash `9ca5ebf01d05bfc3d1ef865b2310141c75c5b7cf52188f2ad7a296134aa9a767`. Runtime source snapshot/hash file inside results. Do not edit live training code and claim the updated file was executed.
- Actual models12blocks (6temporal+6market),hidden192,heads6,dropout.25,patch10. Student5,570,884parameters. Batch256. Separate teacher Beta reward policy,3cycling inner students. True outer score-function RL, not differentiation through student. Internal training is one-step contextual-bandit RL for entry; fixed exit executor, no learned close stage.
- Mainteacherbudget4h; earlystopaftercycle96with64cycleswithoutbestdev. Planned271outercycles for one combined pass, not one pass perstudent. Followed by matched fresh students learned/direct_money/untrained teacher, onefreshseed888. Exact counts recorded.
- Cost$.25/ounce/trade hypothetical, reportsensitivity0/.1/.25/.5/1. Exits<=120minutes orUTCboundary when quotes available; gaps can createovernight holds, disclosed/recorded. Not a guaranteed executable intraday live strategy.
- Remote `supervise.py` PID4876 waits untilprepared, launchestrain, periodicarchive15min, final `/content/meta-reward-results.tar.gz`, markerarchive_ready.json. Eight-hour remote supervisor safety.
- Local `/workspace/xauusd/experiments/meta_reward_transformer/watchdog.py`, PID inreports/.../watchdog_process.json (initial5084), polls45sec. Downloads/extracts/verifies snapshots periodically; finaldownload then stops ONLY meta-reward and runs localreport.py. Seven-hourlocalguard after~20minprep bounds GPUcost. If largeuncachedfeaturetensorisneeded, it is not incompactarchive: source/data/schema allowregeneration; don'tclaim fullmatrixpersistedtoDrive.
- Reports `/workspace/xauusd/reports/meta_reward_transformer_20261015`. `live_status.json`,watchdog.log are preferred for checking live progress without concurrentCLIexecs. Simultaneousmanualexec andwatchdogexec can disconnectonewebsocket, but backgroundtraining continues. ColabstatusIDLE describes Jupyter, not the background GPUworker.
- First completedcycle:8192examples,externalgain+9.575$/day on2meta days (NOTevidenceofedge),teachergradnorm.919, bestdevelopmentmean -9.209$/day. Firstcycle138secincludingcacheload. Source/replaychecks passed.
- Tests saved:meta_rl_checks.json,data_model_checks.json,gpu_meta_checks.json. Local CPUtorch installed.
- No finaloutcome yet. Afterfinishverify3matchedarmscompleted, inspectbootstrapdailycomparisonsandcosts. Ifmainfailedinspectremote/results/error.txt; guardstillarchivesandstops. Uploadfinalreport+checkpoints toDrive asbefore, verifymetadata. Don'tautostartclosingstage.
