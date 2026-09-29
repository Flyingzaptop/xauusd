# V1 synthetic target diagnosis

The frozen v1 synthetic target is identifiable from the closes supplied to
the models. Its near-chance result cannot be explained by two different
generator labels producing the same observed path. This conclusion concerns
the path grammar; it does not establish that the Crossing Memory architecture
can learn the task within its training budget.

`scripts/diagnose_v1_synthetic.py` rounds successive observed float32
log-close changes to integer steps and independently parses the episode and
padding grammar. It does not use the router to decode labels. In 4,000 new
paths from 20 seeds, all paths had one possible label matching the generator;
the largest step-rounding error was 0.000022. The six exact train/test datasets
from `reports/synthetic_metrics.json` were reproduced by SHA-256 and all 1,440
paths also had one correct label.

Exact enumeration covered all 1,081,344 derivations: 16 combinations of two
episode signs/orders, 66 allocations of ten padding pairs, and 1,024 padding
direction sequences. They produced 1,081,344 distinct full paths and **zero**
opposite-label collisions. Weighting each gap allocation by its actual
multinomial probability gives Bayes accuracy 1.0. Cutting each path before
the last episode's seven-step outer return and removing trailing padding
produces 327,696 distinct open prefixes, also with zero opposite-label
collisions and Bayes accuracy 1.0. The prefixes have variable lengths; a
fixed-window benchmark can add repeated initial closes at the left edge.

The new v2 `hard_open` stage keeps the **first three observed steps** of that
outer return, then stops with the large event still open. It left-pads repeated
initial closes to 69 observations. This adjustment removes a terminal-value
shortcut in the raw cut: immediately after the finer excursion, endpoint level
reveals the outer sign while the final step reveals outer sign times the target
order. After three return steps, every path ends at level `+4` with final step
`-1`, or level `-4` with final step `+1`, regardless of label. Exhaustive
enumeration of all 327,696 observable prefixes verifies that each of these
two terminal pairs occurs 81,924 times with each label. The path still retains
a unique grammar label; the independent oracle checks it for generated data.

The v1 test result remains a failed learnability check: mean balanced accuracy
was 0.478 for the full operator and 0.495 for the GRU, while the independent
random-label control was 0.514. The final observation of every closed path
clears its open state, explaining why the no-open ablation and full model had
identical final scores. The open-prefix task ends before that clearing step
and is therefore a more direct check of the open readout. An overfit check on
a tiny fixed sample would distinguish a representation or gradient problem
from an insufficient optimization budget, but no follow-up model fit has
been run as part of this diagnosis.

Machine-readable counts, hashes and parser results are in
`reports/v2/v1-diagnosis.json`. V1 source and results are preserved.
