# Synthetic premarket check

`python scripts/synthetic_checks.py` trains small models on generated price paths
before market fitting. The default run has three seeds (7, 19, 43), 320
independent training paths and 160 independent test paths per seed, ten epochs,
and a generous 600-second failure guard. It writes `reports/synthetic_metrics.json` with the
configuration, Git commit and working-tree state, code and data hashes, every
epoch's training result, final test result, parameter counts, and finite-gradient evidence.
The script errors if the time guard is exceeded; incomplete output is not a
passed check. Run from the repository root after the operator source and
specification are frozen.

Each 73-close path consists of two completed large excursions, with ten neutral
two-step loops distributed randomly before, between, and after them. Within
each large excursion, the price makes two opposed finer moves whose order is
random. The target is whether the **first** finer move inside the **last**
large excursion follows its large excursion's direction. The first excursion
is a distractor. The target is assigned by the path generator, without calling
the router. Both classes start and end at the same close and have identical
total variation and numbers of positive and negative moves. The large move's
sign, fine move order in the distractor, and loop locations vary independently.
Thus the last price, net return, and absolute movement do not determine the
label; the ordered history inside the appropriate excursion does.

The fixed log-price grid uses thresholds of 1, 3 and 6 path steps. All models
receive the same 73 observed closes. The routed models receive a two-feature
sequence (one-step log return and log price relative to the window start), the
six router statistics per scale, and closure events. The full model uses
interscale transfer and reads open memory. Ablations remove transfer or the
open-memory readout. A 36-hidden-unit GRU on the two base features supplies an
ordinary sequence baseline. Each model uses the same 32-unit classification
head, optimizer, epoch count and batch schedule; parameter counts are reported.
The test paths are generated from separate random streams for each seed and
are never used to update parameters. Report balanced accuracy as the primary
synthetic metric because realized random class balances need not be exact.

For the negative control, independent binary labels are drawn for the same
train and test paths from two additional random streams. The full operator and
identical training schedule must show finite loss and gradients; its test
accuracy should be near chance. Any train accuracy above chance on random
labels indicates capacity to memorize and is not evidence of signal. These
checks establish code learnability and a null behavior only; they do not
validate the market target or trading value. Model differences on this task
are descriptive, not a selection rule for changing the frozen market protocol.
