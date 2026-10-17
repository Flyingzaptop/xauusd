"""CPU feasibility audit of one proposed entry target; no learning or tuning.

Input layout follows the recovered dataset: timestamp seconds, log_prices H,O,C,L.
The January 2024 period is already research-exposed. This is NOT an edge test.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def outcome(future, entry, risk, side):
    """First touch: +2R / -1R, next-open entry, 120 minutes.

    An Open crossing is known to occur before that bar's High/Low. If the
    first touch is otherwise both barriers in one M1 bar, order is unknown.
    """
    h, o, c, l = future.T
    fav = side * ((h if side == 1 else l) - entry) / risk
    adv = side * ((l if side == 1 else h) - entry) / risk
    opened = side * (o - entry) / risk
    hits = np.flatnonzero((fav >= 2) | (adv <= -1))
    if not len(hits):
        return 'timeout', float(side * (c[-1] - entry) / risk)
    j = hits[0]
    if opened[j] >= 2:
        return 'tp', None
    if opened[j] <= -1:
        return 'sl', None
    if fav[j] >= 2 and adv[j] <= -1:
        return 'ambiguous', None
    return ('tp' if fav[j] >= 2 else 'sl'), None


def risk_from_completed_history(past):
    # 71 bars: one previous Close, then fourteen completed 5-minute candles.
    bars = past[1:].reshape(14, 5, 4)
    h, l = bars[:, :, 0].max(1), bars[:, :, 3].min(1)
    c = bars[:, -1, 2]
    prev = np.r_[past[0, 2], c[:-1]]
    tr = np.maximum.reduce([h-l, abs(h-prev), abs(l-prev)])
    return float(tr.mean())  # ATR14, simple mean explicitly, not Wilder EMA.


def check_synthetic():
    def bar(h, o, c, l): return np.array([[h, o, c, l]], dtype=float)
    assert outcome(bar(103, 100, 102, 99.5), 100, 1, 1)[0] == 'tp'
    assert outcome(bar(100.5, 100, 99, 98.5), 100, 1, 1)[0] == 'sl'
    assert outcome(bar(102.5, 100, 100, 98.5), 100, 1, 1)[0] == 'ambiguous'
    assert outcome(bar(103, 102.2, 100, 98.5), 100, 1, 1)[0] == 'tp'
    assert outcome(bar(103, 98.8, 100, 98.5), 100, 1, 1)[0] == 'sl'
    assert outcome(bar(100.5, 100, 99.5, 99.3), 100, 1, 1) == ('timeout', -.5)
    assert outcome(bar(100.5, 100, 98, 97), 100, 1, -1)[0] == 'tp'
    assert outcome(bar(101.5, 100, 100, 99.5), 100, 1, -1)[0] == 'sl'
    assert outcome(bar(101.5, 100, 100, 97), 100, 1, -1)[0] == 'ambiguous'


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--source', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    args = a.parse_args()
    check_synthetic()
    with np.load(args.source) as z:
        ts = z['timestamp'] // 60
        price = np.exp(z['log_prices'])
        keys = list(z.files)
    assert np.all(np.diff(ts) > 0)
    assert np.all(price[:, 0] + 1e-8 >= price[:, 1:3].max(1))
    assert np.all(price[:, 3] - 1e-8 <= price[:, 1:3].min(1))
    start = int(np.datetime64('2024-01-01', 'm').astype('int64'))
    stop = int(np.datetime64('2024-02-01', 'm').astype('int64'))
    candidates = np.flatnonzero((ts >= start) & (ts < stop) & (ts % 5 == 0))
    counts = {s: {k: 0 for k in ('tp', 'sl', 'timeout', 'ambiguous')}
              for s in ('long', 'short')}
    timeouts = {'long': [], 'short': []}
    joint = {}
    risk_values, origins = [], []
    skipped = {'missing_history_or_future': 0, 'future_crosses_month': 0, 'zero_risk': 0}
    for i in candidates:
        # Origin is the first future minute, after the last known M1 Close.
        if i < 71 or i + 119 >= len(ts) or ts[i + 119] - ts[i - 71] != 190:
            skipped['missing_history_or_future'] += 1
            continue
        if ts[i + 119] >= stop:
            skipped['future_crosses_month'] += 1
            continue
        past, future = price[i - 71:i], price[i:i + 120]
        risk = risk_from_completed_history(past)
        if risk <= 1e-8:
            skipped['zero_risk'] += 1
            continue
        entry = future[0, 1]
        pair = []
        for name, side in [('long', 1), ('short', -1)]:
            label, realized = outcome(future, entry, risk, side)
            counts[name][label] += 1
            if label == 'timeout': timeouts[name].append(realized)
            pair.append(label)
        key = '/'.join(pair)
        joint[key] = joint.get(key, 0) + 1
        risk_values.append(risk)
        origins.append(i)
    n = len(origins)
    assert n > 0 and all(sum(c.values()) == n for c in counts.values())
    # Future perturbation must not change a risk computed from past only.
    i = origins[len(origins) // 2]
    probe = price[i - 71:i + 120].copy()
    before = risk_from_completed_history(probe[:71])
    probe[71:] *= np.linspace(.8, 1.2, 120)[:, None]
    assert risk_from_completed_history(probe[:71]) == before
    out = {
        'purpose': 'Label feasibility only; no model, no parameter search, no edge claim.',
        'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'available_npz_keys': keys,
        'volume_or_bid_ask_available_in_this_npz': False,
        'period': ['2024-01-01', '2024-02-01'],
        'period_previously_used_for_research': True,
        'timezone_provenance': 'Stored timestamp epoch alignment only; feed/session timezone unverified.',
        'origin_interval_minutes': 5,
        'horizon_minutes': 120,
        'risk': 'Simple mean of 14 completed M5 true ranges, fixed at entry.',
        'tp_r': 2, 'sl_r': 1,
        'entry': 'Next M1 Open after known history; hypothetical fill, no cost/latency/spread model.',
        'candidates': int(len(candidates)), 'eligible': n, 'skipped': skipped,
        'risk_dollars_q10_q50_q90': np.quantile(risk_values, [.1, .5, .9]).tolist(),
        'outcomes': counts,
        'outcome_fractions': {s: {k: v/n for k, v in c.items()} for s, c in counts.items()},
        'joint_long_short_outcomes': joint,
        'timeout_mean_gross_r': {s: float(np.mean(x)) if x else None for s, x in timeouts.items()},
        'checks': {'synthetic_first_touch_cases': 9, 'chronological_unique_timestamps': True,
                   'valid_ohlc': True, 'complete_past_and_future_minutes': True,
                   'risk_unchanged_by_future_mutation': True},
        'limitations': [
            'Overlapping windows are not independent trades.',
            'Unknown spread, fees, latency and intraminute order.',
            'Ambiguous first-touch bars explicitly retained; not silently discarded or called losses.',
            'No intraday-session-close restriction until feed timezone/calendar is verified.',
            'No compression filter or engineered features audited here.',
            'TP/SL frequencies are neither a prediction score nor a profitable strategy.'
        ]
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
