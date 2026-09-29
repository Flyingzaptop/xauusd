# Why the v2 scheduling invariants hold

This argument concerns transitions produced by `TemporalEventRouter` and consumed
by `TreeCrossingMemory`, not arbitrary externally fabricated routing tensors.
It does not prove lossless encoding in finite neural vectors or out-of-distribution
predictive performance. Concrete property and conservation tests accompany it.

## Temporal containment and unique ownership

Initially only permanent root 0 exists. A birth appends one child to the old
deepest live node. Its parent already existed at the previous observation,
and its parent ID never changes. Therefore a birth cannot introduce an orphan,
a cycle or two simultaneous children of one node. On return, the shallowest
qualifying node and its entire descendant suffix are removed in reverse depth
order. Every parent is still live when its child ends. Therefore each descendant
interval is contained in its parent's interval, including equal closure times
with a well-defined postorder. Children that did not independently return are
explicitly marked `ancestor_finalization`.

IDs increment only on birth and are never reused. A popped node cannot close
again. Each pop produces one terminal record/transfer, and no other transition
produces a transfer. Inductively `born = terminated + live_nonroot` and
`terminated = delivered`. Normal state stores only live nodes and counters;
the optional unbounded debug ledger is disabled by default.

## Neural message conservation versus compression

There is exactly one observer depth per observed close. Only its observation
GRU result is selected. For each closing depth, exactly one merge result updates
its immediate parent's slot; the child slot is then zeroed. If the parent also
closes, its now-updated summary moves upward, rather than a second direct copy
of the grandchild being separately sent to root. This is hierarchical aggregation.
Birth resets only an inactive slot; there is no live state to overwrite.

Replacing the learned cells by exact additive cells yields a checkable invariant:
the sum over all live slots equals the sum of all raw observation payloads so far.
`tests/test_message_conservation_v2.py` verifies this at every step for natural
and forced cascades with distinct power-of-two payloads, and verifies derivative
one for each payload. This checks actual neural scheduling in addition to router
counters. The learned GRUs may intentionally forget content; exactly-once delivery
must not be confused with information-theoretic losslessness. Whether useful
content survives is tested by learning, ablations and a fixed-checkpoint hidden-state
intervention. `no_transfer` is an explicit content-erasure experiment, not normal
message handling.

## Mirror symmetry

Under z' = 2*z0 - z, every stored origin/anchor maps by the same transformation.
Absolute displacements and threshold tests are unchanged. Birth direction reverses.
Direction times displacement is unchanged, so return tests, directional maxima,
retreat counts, ages and closure reasons are unchanged. Raw signed displacement
and latest signed move reverse. Unique IDs depend only on unchanged birth order,
so parent topology is unchanged by induction. The tolerance is positive and
proportional to the smallest fixed threshold; it is applied symmetrically.
This proves routing symmetry. Unconstrained trained weights do not imply mirror-
equivariant predictions, and this stage makes no such claim.

## Chunk equality and causality

The router transition reads only prior router state and the current observed close.
The neural transition reads only prior neural state, current features and current
routing. No chunk length or future close enters either transition. Chunking is
therefore function composition of the same transitions. Carrying both states and
using the same per-observation input features gives identical outputs and final
states; retaining the graph without detachment also gives the same gradients.
Tests compare full sequences, one-step calls, chunks, serialization/resume and
future-perturbed prefixes. External feature construction must also remain causal;
for this benchmark each one-step delta is computed on the continuous sequence,
not reset at a chunk boundary.

## Jumps

A jump is a single observed endpoint. The router opens at most one node at that
observation and performs one raw neural observation update. A cascade may close
several existing nodes in tree order; this order follows ownership, not an
invented sequence of intermediate prices. No same-observation reopening is allowed.
