# Crossing Memory v1: operator specification

This document fixes the operator before any market training or test inspection.

## Inputs and outputs

`route_windows(close, thresholds)` accepts positive finite closes `[N,T]` and
strictly increasing positive log-price thresholds `[S]`. Each threshold must be
an integer multiple of the preceding threshold. It returns `stats` as float32
`[N,T,S,6]` and `closures` as bool `[N,T,S]`. Windows are independent: the first
close anchors a shared grid and has zero statistics and no event.

The feature order is direction (`-1,0,+1`), `log1p(age)` in observed M1 closes,
signed progress from the event's origin in grid bins, maximum excursion in bins,
`log1p(internal_returns)`, and the latest signed log-close change divided by the
scale threshold. A closing event's summary remains in `stats` for its closing
observation, with `closures=True`; later observations use the new state.

`CrossingMemory(input_dim, hidden_dim=12, scales=S, transfer=True,
read_open=True)` accepts base features `x [B,T,D]`, routed statistics
`stats [B,T,S,6]`, and `closures [B,T,S]`. It returns features
`[B,T,S*hidden_dim]` (`output_dim`). Its `step` accepts one observation and an
explicit state, returning `(features_t, state_t)`. The state consists of
`open [B,S,H]` and `closed [B,S,H]`; both start at zero. A new window or unrelated
batch must call `initial_state`/`reset_state` or omit state from `forward`.

## Routing and event lifecycle

For window anchor `a=log(close[0])`, grid index at scale `s` is
`floor((log(close[t])-a)/threshold[s])`. Grid origins align because adjacent
scales have integer ratios. This is a spatial hierarchy of grid boundaries,
not a guarantee that a fine event remains open throughout a coarse event. A
coarse event can close before a fine event. The initial index is zero.
Only consecutive observed closes are compared. A crossing is any change in
index; movement within a bin is no crossing. The first crossing when idle opens
an event at the preceding index, with direction equal to the sign of the index
change. Its age is one observed update and its maximum excursion is the new
distance from origin. An open event ages once per subsequent close, including
closes without a crossing. The event closes when its signed index displacement
reaches zero or passes through it. A move toward the origin that stays on the
original side increments `internal_returns` once for that observation. Maximum
excursion is retained. After closure, the current index becomes the new idle
origin; an overshoot does not open a second event at the same observation.

A jump over multiple grid levels produces one atomic update at each affected
scale using the observed endpoint. Crossed level count appears in progress and
maximum excursion, but there are no synthetic intermediate prices or repeated
neural updates. Scales are processed fine to coarse at an observation. If a
fine event and its parent both close, the fine learned state is transferred to
the parent before the parent closes.

In v1 a child closure sends its learned state only to a parent event active at
that observation. If the parent is idle, the parent update is gated off and the
child message is discarded. No pending transfer queue exists. Consequently,
the parent need not aggregate every completed fine event.

OHLC, if used, may enter `x` only after the candle closes. The router uses closes
alone and makes no assumption about the order of high and low within a minute.
Thresholds are fixed from training data and unchanged through an open event.

## Learned state update and gradients

Each scale has an `open_update` GRUCell with input `[x_t, stats_t,
child_message_t]`. `child_message_t` is the fine scale's updated open state
when that event closes; otherwise it is zero. The `transfer=False` ablation
zeros this message. On closure, a separate learned `close_update` GRUCell
absorbs the just-updated open state into persistent closed memory, and open
memory resets to zero. Thus transfer is an actual trained path between scales,
including when the parent closes on the same observation.

Readout is a learned projection of `[closed, open]` at each scale, concatenated
across scales. `read_open=False` zeros the open input to the projection, so an
unfinished event cannot directly reach the prediction head. Gradients flow
through open updates, close updates, transfer paths, and readout; they do not
flow through discrete close routing or fixed thresholds. The closed memory is a
fixed-size vector, not an unbounded event archive. There are exactly `2*S*H`
state scalars per sample. Parameters are shared over time, and each independent
window starts with zero state. All routed statistics, closure decisions and
neural updates at time `t` depend only on closes and base inputs through `t`.
