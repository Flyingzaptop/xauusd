# Independent v2 event-tree review notes

These notes are a proposed implementation checklist for a **new** operator. They do not change the v1 router, model, checkpoints, or results. Prices below are log-price coordinates in units of a fixed base threshold; `[6,3,1]` means coarse, middle, and fine thresholds. Every number in a trace is an observed close, not an inferred path between closes.

## State and causal ordering

Keep a permanent root with an anchor at the first observed close, updated to the observed close when its coarse child closes. A nonroot node records an immutable event ID, parent ID, opening anchor, direction, birth close, and a mutable `child_anchor`. A parent's `child_anchor` begins at its birth close and becomes the observed close of each child closure. There is at most one active node at each depth and no child without its recorded active parent. The root may have a coarse child only when `abs(close-root_anchor) >= 6`; a coarse node may open a middle child at threshold 3, and a middle node a fine child at threshold 1. Direction is the sign of `close-anchor`. A parent must have been active at the **preceding** observed close before its child can open. One observed jump never creates a fictional sequence of nested threshold crossings.

At a new close, first determine natural closures among nodes active before that close. A node closes naturally if its signed displacement from its own immutable opening anchor has returned to zero or crossed it. If an ancestor qualifies, every still-open descendant must be finalized deepest-first at this same observed close. A descendant that independently qualifies for natural closure keeps that reason; otherwise its reason is `ancestor_finalization`. No node may reopen at the same observation where it closed. If no node closes, at most one new child depth may open, and only below a parent active at the preceding close. Opening uses the current observed close against the parent's fixed child anchor. Route metadata for all active nodes may be updated from the current close; no intraminute crossing order is invented.

Only the deepest active node owns the **raw observation** update. If a closure is pending, the old deepest node receives the current observation, then closure messages flow deepest-first, exactly once along recorded parent IDs. A parent incorporates a child's learned message but does not also consume the same raw observation. If there is no closure and a child opens, the new child receives that observation. The root receives raw observations only when it has no child; otherwise it receives closed coarse messages. The root never closes or resets. The `no_transfer` ablation must zero the child-message path, and a root-only readout ablation must exclude direct open-node readout. This ownership rule makes those ablations interpretable and prevents a direct observation bypass to root.

Memory can remain fixed size: one root state, at most one state per nonroot depth, and a scalar event-ID counter. An event ledger may be emitted for tests and diagnostics, but should not be needed for forward computation. For every closed event, the ledger should show exactly one finalization and one message to the node's recorded parent; the parent must still be live when it receives that message.

## Manual traces

| Observed closes | Required event behavior |
| --- | --- |
| `0, 6, 9, 10, 9, 6, 0` | At 6 coarse `C` opens from root anchor 0. At 9 middle `M` opens from `C.child_anchor=6`. At 10 fine `F` opens from `M.child_anchor=9`. At the next 9, `F` naturally closes and messages `M` once. At 6, `M` naturally closes and messages `C` once. At 0, `C` naturally closes and messages root once. Parent IDs never change. |
| `0, 6, 3, 2, 0` | `C` opens upward at 6; `M` opens downward at 3 from anchor 6; `F` opens downward at 2 from anchor 3. At 0, `C` naturally returns to origin while `M` and `F` do not return to their own origins. Finalize `F` then `M` with `ancestor_finalization`, deliver `F→M→C`, then naturally close `C→root`. Each child sends one message. |
| `0, 6, 9, 10, 0` | The last observed close crosses the opening anchors of `F`, `M`, and `C`. All three closures are independently natural, but are still processed `F→M→C→root` at that one timestamp. |
| `0, 6, 9, 6, 9` | `M` opens at first 9, naturally closes at 6, and a *new* middle event may open at the later 9 from refreshed `C.child_anchor=6`. It cannot reopen at the closing 6. |
| `0, 10` | The single jump opens only `C`, anchored at 0. It does not create `M` or `F` at the same close, even though the endpoint exceeds all three thresholds. At `0, 10, 10`, no child opens because the middle anchor is 10. |

## Invariants to test before any market experiment

1. Prefix causality: adding later closes leaves all earlier route outputs, event IDs, and messages unchanged.
2. Tree integrity: one active node per depth; every live child's immutable parent ID identifies a live parent; a closure produces exactly one message to that parent and no orphan or duplicate transfer.
3. Cascade order: a coarse return finalizes descendants at the same timestamp, deepest-first, and preserves natural versus ancestor-finalization reasons.
4. Observation ownership: exactly one node receives the raw close per time step, including opening and cascade steps; ancestors receive only child messages.
5. Mirror symmetry of routing: for mirrored log prices `L'_t = 2L_0 - L_t`, event times, IDs, parent links, closure reasons, ages and directional progress match; directions and raw signed displacements reverse. Arbitrary learned weights do **not** make model predictions mirror-symmetric without an explicit equivariant design.
6. Atomic jumps: no extra child event or message is invented between two observed closes; no closed node reopens on its closure observation.

The one-observation delay for child opening is a conservative consequence of using only observed closes. It may miss within-interval dynamics, which cannot be recovered from M1 close prices. The fixed root anchor also means a coarse event can open from cumulative movement over several closes, rather than requiring a single large one-minute return.
