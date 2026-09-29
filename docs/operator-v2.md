# Crossing Memory v2: temporal event tree (pre-implementation specification)

Date: 2026-09-29. V1 code, data, market protocol and negative reports remain unchanged.
This stage uses generated sequences only. January–August 2026 market test is exposed.

## Evidence motivating the change

V1 floors signed positions on independent spatial grids. Thus a tiny negative displacement opens events which the mirrored positive displacement does not. Its scales have no immutable parent event IDs or temporal containment. In the recorded train routing counts, 4183/15629 and 5082/14558 child closure messages were dropped when the corresponding coarser event was idle. V1 synthetic open-readout ablation ends with all open states reset and is therefore nondiagnostic. These are structural issues, distinct from the still-to-be-investigated optimization failure on nested-history labels.

## Tree and observation semantics

Operate on observed log closes z[t], not high/low paths. Use fixed amplitude thresholds coarse to fine, initially [6,3,1] in synthetic step units. They are hierarchical event thresholds, not a set of spatial grids. A permanent root has ID 0, depth 0, and never closes. At most one live child exists per node. The active nodes form a root-to-leaf chain of maximum depth S. Every new node has a monotonically increasing unique ID and immutable parent ID. Parent is live before the child's birth. Birth time is the current observation, not an inferred level-crossing time.

A parent's waiting-child anchor is its observed birth close, or the observed close at which its previous child ended. Root initially anchors at z[0]. When the deepest node's displacement from that anchor reaches the next threshold in absolute value, open one child with origin equal to that anchor, direction sign(displacement), and birth close z[t]. The new child's own waiting-child anchor is z[t]. Only ONE child may open per observation. A jump never creates hypothetical intermediate samples or several new generations. Threshold equality is inclusive with a small symmetric absolute tolerance, specified in code and tested on mirrored paths.

A non-root event naturally closes when direction*(z[t]-origin) <= tolerance. If an ancestor naturally closes, all its still-active descendants must be finalized at this same observed endpoint, deepest first, before that ancestor. Descendants are labelled `natural` if their own return condition holds, otherwise `ancestor_finalization`. The latter does not assert a price return that did not occur. No reopening occurs on a closure observation, even on overshoot. End-of-input does NOT flush unfinished events; their state is available to open-event readout and future chunks.

## Atomic order at each observation

1. On the first observed close initialize root and its anchor, then observe that close once.
2. For later closes, update routing metadata of existing live nodes (age in observations, signed progress, direction-aligned maximum excursion, internal retreats).
3. Test natural closures on the preexisting chain. If any closes, choose its shallowest depth; deliver the observed feature vector exactly once to the OLD deepest node, then finalize descendants and that node in descending depth order. Each message is absorbed by the recorded parent before that parent itself can close. Reset each closed child's slot and reset the surviving parent's waiting-child anchor to this observed close. Do not open a node this tick.
4. Otherwise, optionally open one child of the old deepest node if its threshold is met. Deliver the observed feature vector exactly once to the NEW deepest node (or unchanged deepest if no birth).
5. Read root and live-node state. Metadata and neural state are carried unchanged across chunk boundaries; chunking itself creates no events.

## Neural operator and bounded state

Routing produces per observation arrays over D=S+1 depth slots: metadata [B,D,F], birth mask [B,D], closure mask [B,D], observer depth [B], post-update active mask [B,D]. The implementation will expose the exact F ordering. Metadata contains direction, duration, signed progress, maximum excursion, internal retreats and latest observed move, normalized by the fixed depth threshold as applicable. Root is always active.

Neural persistent state is h [B,D,H]. Birth clears a previously inactive slot; the single observer updates its state using a learned GRUCell([x[t], metadata], h). Ancestors do not additionally ingest the same raw observation. A finalized child transfers its h exactly once into its actual parent using a learned merge GRUCell([child_h,parent_metadata], parent_h), then its slot is zeroed. Deepest-first order propagates descendant information through a simultaneously closing ancestor. A bounded vector compresses history; exact symbolic message accounting is NOT a claim of lossless neural encoding.

Readout is a learned projection of each live slot's [h,metadata], concatenated over depths, with inactive slots zeroed. `no_transfer` replaces child content by zero but preserves closure timing, parent metadata, routing and accounting; it tests learned content transfer, not timing. `no_open` hides all non-root slots at readout only, preserving updates/transfers. Root continues accepting every top-level closure. Root observes only when there is no live child, avoiding an unrestricted parallel raw-sequence shortcut through ancestors.

All learnable paths (observation, merge, readout and classifier) receive gradients; thresholds and routing are discrete. State may be detached explicitly for truncated backpropagation, never implicitly between chunks. Routing state includes previous close, time index, next event ID, live-node origins/anchors/ages/retreats/IDs/parent IDs and accounting counters. Optional debug records are external and may grow; operational active state stays O(S). Neural state stays O(B*D*H). No per-event archive is required during normal streaming.

## Invariants required before training

- A live child has exactly one live parent, born earlier; temporal intervals are contained. Closing a parent cannot leave an orphan.
- Each born event is either active or has exactly one terminal record; each terminated event has exactly one transfer to its immutable parent. No duplicate transfer or silent disappearance is allowed.
- Transferred child summaries are consumed in temporal completion order; simultaneous chain closure is strictly deepest-first. Hierarchical aggregation is intentional, not a second direct delivery of a descendant to root.
- Negating the entire log-close path about its initial point preserves IDs, parent IDs, births, closure times/reasons, ages and absolute excursions; directions, signed progress and latest returns reverse.
- Full sequence, one-step streaming and arbitrary chunks give equal routing decisions, final routing state, predictions and neural state. Future perturbations cannot change prefixes.
- No synthetic intraminute observation is inserted. A jump gives one observation update and at most one birth; it can trigger several closures with distinct documented reasons.

## Hand traces (thresholds [6,3,1], z0=0)

`0, 6, 9, 10, 9, 6, 0`: root observes 0. At 6 open E1(parent root, origin 0). At 9 open E2(parent E1, origin 6). At 10 open E3(parent E2, origin 9). At 9 observe E3 then close E3 -> E2. At 6 observe E2 then close E2 -> E1. At 0 observe E1 then close E1 -> root. Every event delivered once; root owns the final aggregate.

`0, 6, 9, 10, -2`: same births; final endpoint naturally closes all three; observe E3 once, merge E3 -> E2, E2 -> E1, E1 -> root, no opposite event born at -2. The log-close jump is not expanded into a price path.

`0, 6, 3, 2, 0`: E1 direction +, E2 direction - with origin 6, E3 direction - with origin 3. At 0 E1 naturally closes while E2/E3 are still moving away from their origins. Finalize E3 and E2 with `ancestor_finalization`, then close E1 naturally. Messages still arrive once in postorder.

`0, 20, 20, 20`: only E1 is born at 20. Child anchor is 20, so the repeated endpoint creates no finer event. Mirroring each trace by z -> -z yields the same tree and closure schedule with reversed directions.

`0, 6, 9, 10` stopped here: all E1/E2/E3 remain open. A following chunk beginning with 9 continues the same E3 and closes it; there is no reinitialization or end-of-chunk flush.

These traces and invariants are normative and precede v2 implementation. Subsequent experimental protocols must separately test recoverable labels, tiny-set memorization, independent increasing complexity, closed versus open endpoints, all seeds and matched-budget GRU comparisons. Negative results must be localized rather than followed by market expansion.

## Implementation contract (added after normative specification)

`src/crossing_memory_v2.py` exports `TemporalEventRouter(thresholds)` for a
single stream, `route_windows_v2(log_prices[N,T], thresholds)` for independent
windows, and `TreeCrossingMemory(input_dim, hidden_dim, depths, transfer=True,
read_open=True)`. Thresholds are strictly decreasing, coarse to fine; the
tensor depth count is one more than the threshold count. Router `step(z)` returns
one tick, `process(z_chunk)` returns stacked ticks without flushing, and
`state_dict`/`load_state_dict` carry bounded router state across chunks.
Neural `step(x_t, routing_t, state)` and `forward(x, routing, state=None)` both
return `(output, new_state)`; the state tensor has shape `[B,D,H]`. Neural
`forward` returns `[B,T,D*H]` outputs and never detaches an incoming state.

The six metadata channels, in exact order, are: (0) direction; (1) `log1p(age)`
in observed closes; (2) raw signed displacement from the node origin divided
by that depth's threshold; (3) maximum direction-aligned excursion divided by
threshold (root uses maximum absolute excursion); (4) `log1p(internal retreats)`;
and (5) latest signed observed log-close move divided by threshold. Root uses
the coarsest threshold for normalization. Closed nodes retain their current
metadata for the closing tick, while `active` describes the post-closure chain.
The router also emits birth/closure masks, observer depth, live IDs and parent
IDs, terminal IDs/parent IDs, and integer closure reasons (`1` natural, `2`
ancestor finalization, `0` none). The output dictionary is a trace; normal
persistent state stores only active nodes and accounting counters. Optional
terminal debug records are disabled by default.

The numeric comparison tolerance is `1e-10 * min(thresholds)` in log-close
units. It applies symmetrically to births, natural closures and retreat checks.
It is smaller than every positive threshold, so an unchanged close cannot open
an event even for very small threshold values. This implementation detail
refines the pre-implementation requirement for a symmetric absolute tolerance.
