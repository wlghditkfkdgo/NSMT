"""MODI 2026-10-28: summary of rhythm_cause.py (cause analysis without training) with the pre-registered rules
(docs/MODI_PROJECT_LOG.md, 2026-10-28 pre-registration). Incomplete or failed-check runs make the affected verdict 'incomplete'.

B (T = N neocortex added to Spikformer-style, 90 epochs): every T = N neuron n vs the memoryless LIF neocortex, paired by seed
(the D2 pairing); the primary pair is LIF. A difference 'holds' when the mean is past the threshold in the stated direction and
the sign agrees in >= 2 of 3 seeds.
  H1 supported   own-patch readout R^2 (lag 0) lower by >= 0.02  AND  effective number of neocortex keys higher by >= 10 %
                 (one of the two: partly)
  H2 supported   T = N: |first-5-token / steady (tokens >= 20) - 1| >= 0.2 in >= 2 of 3 seeds, memoryless < 0.1 in >= 2 of 3,
                 for the firing rate or for the own-patch readout
  H3 supported   neocortex distance from init lower by >= 10 %  AND  neocortex / own-attention gradient norm lower by >= 10 %
                 (one of the two: partly)
  H4 supported   swap drop (base - next-window neocortex) < 0.5 %p (mean) for both the T = N and the memoryless LIF
                 memory content unused: T = N history-scramble drop < 0.5 %p (mean)
H5 (Spikformer-style, 90 epochs, mean of 3 seeds) supported: last-16-token score >= full - 2 %p AND full-shuffle (b = 1) score
   >= full - 2 %p. Neo-mixer leans on order more: mean b = 1 drop >= 2 x Spikformer-style's and >= 1 %p.
usage: python summarize_cause.py [--suite flif-v1-cause-cinc]
"""
import argparse
import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (42, 7, 2026)
B_CONDS = ('spkneo_lif', 'spkneo_flif', 'spkneo_pop_lif', 'spkneo_pop_flif')
NOMEM = 'spkneo_lif_nomem'
H5_CONDS = ('spk_ssa', 'mix_lif_bidir', 'mix_flif_bidir', 'mix_pop_lif_bidir', 'mix_pop_bidir')
LAB = {'spkneo_lif': 'B, T = N LIF', 'spkneo_flif': 'B, T = N f-LIF', 'spkneo_pop_lif': 'B, T = N population LIF',
       'spkneo_pop_flif': 'B, T = N population f-LIF', NOMEM: 'B, memoryless LIF', 'spk_ssa': 'Spikformer-style',
       'mix_lif_bidir': 'Neo-mixer LIF', 'mix_flif_bidir': 'Neo-mixer f-LIF', 'mix_pop_lif_bidir': 'Neo-mixer population LIF',
       'mix_pop_bidir': 'Neo-mixer population f-LIF'}
K_LIST, B_LIST = (4, 8, 16, 32, 64), (1, 4, 16)


def load(res):
    runs = {}
    for f in glob.glob(os.path.join(res, '*', 'cause.json')):
        r = json.load(open(f))
        runs[(r['cond'], r['seed'])] = r
    return runs


def get(runs, cond, key):
    """per-seed values; None when a run is missing, failed a check or the value is not finite."""
    out = []
    for s in SEEDS:
        r = runs.get((cond, s))
        v = None if (r is None or r['checks_failed']) else r
        for k in key.split('/'):
            v = v.get(k) if isinstance(v, dict) else None
        out.append(v if isinstance(v, (int, float)) and math.isfinite(v) else None)
    return out


def holds(a, b, direction, thr, rel=False):
    """'yes' / 'no' / 'incomplete': mean(a - b) past thr in direction (+1 higher, -1 lower) and sign in >= 2 of 3 seeds."""
    if any(v is None for v in a + b):
        return 'incomplete', None
    d = [(x - y) / y if rel else x - y for x, y in zip(a, b)] if not rel or all(y != 0 for y in b) else None
    if d is None:
        return 'incomplete', None
    ok = direction * np.mean(d) >= thr and sum(direction * v > 0 for v in d) >= 2
    return ('yes' if ok else 'no'), d


def fmt(v, sc=1., nd=3):
    return '—' if v is None else f'{v * sc:.{nd}f}'


def mean_row(vals, sc=1., nd=3):
    if any(v is None for v in vals):
        return 'incomplete'
    return f'{np.mean(vals) * sc:.{nd}f} (' + ' / '.join(f'{v * sc:.{nd}f}' for v in vals) + ')'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='flif-v1-cause-cinc')
    a = ap.parse_args()
    res = os.path.join(HERE, 'results', a.suite)
    runs = load(res)
    L = [f'# Cause analysis without training `{a.suite}` (CINC2017 fold 0, 90-epoch weights) — summary', '',
         'Best-validation weights, test windows. Values: mean (seed 42 / 7 / 2026). Pre-registered rules: docs/MODI_PROJECT_LOG.md '
         '(2026-10-28) and the docstring of summarize_cause.py.', '']
    missing = [f'{c} seed {s}' for c in B_CONDS + (NOMEM,) + H5_CONDS for s in SEEDS if (c, s) not in runs]
    failed = [f"{r['name']}: {r['checks_failed']}" for r in runs.values() if r['checks_failed']]
    L += [f'Runs found: {len(runs)} of {len(SEEDS) * (len(B_CONDS) + 1 + len(H5_CONDS))}; missing: {", ".join(missing) or "none"}; '
          f'failed checks: {"; ".join(failed) or "none"}.', '']

    # ---- B tables
    bc = (NOMEM,) + B_CONDS
    L += ['## B models: measurements', '',
          '| measurement | ' + ' | '.join(LAB[c] for c in bc) + ' |', '|---|' + '---|' * len(bc)]
    rows = [('test score (%)', 'score_base', 100, 2), ('drop: neocortex of the next window (%p)', None, 100, 2),
            ('drop: neocortex zeroed (%p)', None, 100, 2), ('drop: history scramble (%p)', None, 100, 2),
            ('neocortex term share of the attention output', 'neo_term_share', 1, 3),
            ('H1 readout R² own patch (lag 0)', 'readout_patch_lag', 1, 3), ('H1 readout R² patch lag 2', 'readout_patch_lag', 1, 3),
            ('H1 readout R² R-peak in token', 'readout_peak', 1, 3), ('H1 readout R² last RR interval', 'readout_rr', 1, 3),
            ('H1 adjacent-token cosine, output', 'adj_cos_out', 1, 3), ('H1 adjacent-token cosine, input', 'adj_cos_in', 1, 3),
            ('H1 effective number of neocortex keys', 'eff_keys_neo', 1, 1), ('H1 effective number of own keys', 'eff_keys_own', 1, 1),
            ('H2 output firing rate', 'rate_mean', 1, 4), ('H2 rate first 5 / steady', 'rate_first5_over_steady', 1, 3),
            ('H2 rate slope per 100 tokens (rel.)', 'rate_slope_per100', 1, 3), ('H2 readout first 5 / steady', 'r2_first5_over_steady', 1, 3),
            ('H3 distance from init, neocortex', 'h3_init_dist/neocortex', 1, 3), ('H3 distance from init, neocortex keys/values', 'h3_init_dist/neo_kv', 1, 3),
            ('H3 distance from init, own attention', 'h3_init_dist/attn_own', 1, 3), ('H3 gradient norm neocortex / own attention', 'h3_grad_neo_over_own', 1, 3)]
    vals = {}
    for c in bc:
        base = get(runs, c, 'score_base')
        for nm, key in (('swap', 'score_swap'), ('zero', 'score_zero'), ('scramble', 'score_scramble')):
            v = get(runs, c, key)
            vals[(c, nm)] = [None if x is None or y is None else x - y for x, y in zip(base, v)]
        for i, k in ((0, 'lag0'), (2, 'lag2')):
            vals[(c, k)] = [None if r is None or r['checks_failed'] else r['readout_patch_lag'][i] for r in (runs.get((c, s)) for s in SEEDS)]
    for nm, key, sc, nd in rows:
        cells = []
        for c in bc:
            if nm.startswith('drop'):
                v = vals[(c, {'drop: neocortex of the next window (%p)': 'swap', 'drop: neocortex zeroed (%p)': 'zero',
                              'drop: history scramble (%p)': 'scramble'}[nm])]
            elif key == 'readout_patch_lag':
                v = vals[(c, 'lag0' if 'lag 0' in nm else 'lag2')]
            else:
                v = get(runs, c, key)
            cells.append(mean_row(v, sc, nd))
        L.append(f'| {nm} | ' + ' | '.join(cells) + ' |')
    L.append('')

    # ---- B verdicts
    L += ['## B models: pre-registered verdicts (each T = N neuron vs the memoryless LIF, paired by seed; primary = LIF)', '',
          '| hypothesis | T = N neuron | part | T = N − memoryless per seed | holds | verdict |', '|---|---|---|---|---|---|']
    nm_ = vals
    for c in B_CONDS:
        r1, d1 = holds(nm_[(c, 'lag0')], nm_[(NOMEM, 'lag0')], -1, 0.02)
        r2, d2 = holds(get(runs, c, 'eff_keys_neo'), get(runs, NOMEM, 'eff_keys_neo'), +1, 0.10, rel=True)
        v = 'incomplete' if 'incomplete' in (r1, r2) else ('supported' if r1 == r2 == 'yes' else ('partly' if 'yes' in (r1, r2) else 'not supported'))
        L.append(f'| H1 token blurring | {LAB[c]} | own-patch R² lower ≥ 0.02 | {" / ".join(fmt(x) for x in d1) if d1 else "—"} | {r1} | {v} |')
        L.append(f'| | | neocortex keys higher ≥ 10 % (rel.) | {" / ".join(fmt(x, 100, 1) + " %" for x in d2) if d2 else "—"} | {r2} | |')
        rr, drr = holds(get(runs, c, 'readout_rr'), get(runs, NOMEM, 'readout_rr'), +1, 0.02)
        L.append(f'| (memory probe) | | last-RR readout higher ≥ 0.02 | {" / ".join(fmt(x) for x in drr) if drr else "—"} | {rr} | |')
        h2 = []
        for key in ('rate_first5_over_steady', 'r2_first5_over_steady'):
            t, m = get(runs, c, key), get(runs, NOMEM, key)
            if any(x is None for x in t + m):
                h2.append('incomplete')
            else:
                h2.append('yes' if sum(abs(x - 1) >= 0.2 for x in t) >= 2 and sum(abs(x - 1) < 0.1 for x in m) >= 2 else 'no')
        v2 = 'incomplete' if 'incomplete' in h2 else ('supported' if 'yes' in h2 else 'not supported')
        L.append(f'| H2 position drift | {LAB[c]} | rate / readout first 5 vs steady | rate {h2[0]}, readout {h2[1]} | | {v2} |')
        r3a, d3a = holds(get(runs, c, 'h3_init_dist/neocortex'), get(runs, NOMEM, 'h3_init_dist/neocortex'), -1, 0.10, rel=True)
        r3b, d3b = holds(get(runs, c, 'h3_grad_neo_over_own'), get(runs, NOMEM, 'h3_grad_neo_over_own'), -1, 0.10, rel=True)
        v3 = 'incomplete' if 'incomplete' in (r3a, r3b) else ('supported' if r3a == r3b == 'yes' else ('partly' if 'yes' in (r3a, r3b) else 'not supported'))
        L.append(f'| H3 less learning | {LAB[c]} | neocortex distance from init lower ≥ 10 % | {" / ".join(fmt(x, 100, 1) + " %" for x in d3a) if d3a else "—"} | {r3a} | {v3} |')
        L.append(f'| | | gradient neocortex / own lower ≥ 10 % | {" / ".join(fmt(x, 100, 1) + " %" for x in d3b) if d3b else "—"} | {r3b} | |')
    sw_t, sw_m = nm_[('spkneo_lif', 'swap')], nm_[(NOMEM, 'swap')]
    if any(x is None for x in sw_t + sw_m):
        v4 = 'incomplete'
    else:
        v4 = 'supported' if np.mean(sw_t) < 0.005 and np.mean(sw_m) < 0.005 else 'not supported'
    L.append(f'| H4 neocortex unused | LIF and memoryless LIF | swap drop < 0.5 %p (mean) | T = N {mean_row(sw_t, 100, 2)}, memoryless {mean_row(sw_m, 100, 2)} | | {v4} |')
    for c in B_CONDS:
        sc_ = nm_[(c, 'scramble')]
        v = 'incomplete' if any(x is None for x in sc_) else ('unused' if np.mean(sc_) < 0.005 else 'used')
        L.append(f'| H4 memory content | {LAB[c]} | history-scramble drop < 0.5 %p (mean) | {mean_row(sc_, 100, 2)} | | {v} |')
    L.append('')

    # ---- H5
    L += ['## H5: last k tokens (tiled) and token-block shuffles (test score %, 90-epoch weights)', '',
          '| model | full | ' + ' | '.join(f'last {k}' for k in K_LIST) + ' | ' + ' | '.join(f'shuffle b={b}' for b in B_LIST) + ' |',
          '|---|---|' + '---|' * (len(K_LIST) + len(B_LIST))]
    for c in H5_CONDS:
        cells = [mean_row(get(runs, c, 'score_base'), 100, 2)]
        cells += [mean_row(get(runs, c, f'score_last{k}'), 100, 2) for k in K_LIST]
        cells += [mean_row(get(runs, c, f'score_shuf{b}'), 100, 2) for b in B_LIST]
        L.append(f'| {LAB[c]} | ' + ' | '.join(cells) + ' |')
    L.append('')
    base, l16, sh1 = (get(runs, 'spk_ssa', k) for k in ('score_base', 'score_last16', 'score_shuf1'))
    if any(x is None for x in base + l16 + sh1):
        v5 = 'incomplete'
    else:
        v5 = 'supported' if np.mean(l16) >= np.mean(base) - 0.02 and np.mean(sh1) >= np.mean(base) - 0.02 else 'not supported'
    L += [f'- H5 (Spikformer-style): full {np.mean(base) * 100:.2f}, last 16 {np.mean(l16) * 100:.2f}, shuffle b=1 {np.mean(sh1) * 100:.2f} → **{v5}**'
          if v5 != 'incomplete' else '- H5 (Spikformer-style): **incomplete**']
    if v5 != 'incomplete':
        ssa_drop = np.mean(base) - np.mean(sh1)
        for c in H5_CONDS[1:]:
            b_, s_ = get(runs, c, 'score_base'), get(runs, c, 'score_shuf1')
            if any(x is None for x in b_ + s_):
                L.append(f'- {LAB[c]}: incomplete'); continue
            d = np.mean(b_) - np.mean(s_)
            lean = 'leans on order more' if d >= 2 * ssa_drop and d >= 0.01 else 'not more'
            L.append(f'- {LAB[c]}: shuffle b=1 drop {d * 100:.2f} %p vs Spikformer-style {ssa_drop * 100:.2f} %p → {lean}')
    L.append('')
    out = os.path.join(res, 'summary.md')
    open(out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
