"""Summary and pre-registered verdicts of the MIT-BIH suite (MODI_PROJECT_LOG 2026-10-19).

Reads results/<suite>/raw.txt (RESULT lines of run_mitbih.sh) and results/<suite>/analysis/<cond>_seed<s>/ (cls_analysis.py)
and writes results/<suite>/summary.md, runs.csv, conditions.csv.

Verdict rule (paired by seed, test macro-F1, last-epoch weights): A better than B if the mean difference is >= +0.5 %p and
A > B in at least 2 of 3 seeds; worse if <= -0.5 %p and A < B in at least 2 of 3; otherwise no clear difference.
A pair with a missing seed or a non-finite value is 'incomplete' (never a pass).
"""
import argparse
import math
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (42, 7, 2026)
CONDS = ('base', 'neo_lif_both', 'neo_flif_both', 'neo_lif_mlp', 'neo_flif_mlp', 'no_neo', 'hippo_only',
         'neo_pop_both', 'bidir_lif_both', 'bidir_flif_both', 'bidir_pop_both',
         'mix_none_fwd', 'mix_lif_bidir', 'mix_flif_bidir', 'mix_pop_bidir', 'mix_pop_fwd', 'mix_pop_bidir_L2')
LABEL = {'base': 'ECG model (spikingjelly LIF)', 'neo_lif_both': 'spikeDE LIF, both', 'neo_flif_both': 'f-LIF, both',
         'neo_lif_mlp': 'spikeDE LIF, MLP only', 'neo_flif_mlp': 'f-LIF, MLP only', 'no_neo': 'no neocortex (ECG ablation)',
         'hippo_only': 'hippo-only (decision 7)',
         'neo_pop_both': 'population f-LIF, both', 'bidir_lif_both': 'bidirectional spikeDE LIF', 'bidir_flif_both': 'bidirectional f-LIF',
         'bidir_pop_both': 'bidirectional population f-LIF',
         'mix_none_fwd': 'mixer none (aligned, no mixing)', 'mix_lif_bidir': 'mixer bidirectional spikeDE LIF',
         'mix_flif_bidir': 'mixer bidirectional f-LIF', 'mix_pop_bidir': 'mixer bidirectional population f-LIF',
         'mix_pop_fwd': 'mixer forward population f-LIF', 'mix_pop_bidir_L2': 'mixer bidirectional population f-LIF, 2 blocks'}
NEO_CONDS = ('base', 'neo_lif_both', 'neo_flif_both', 'neo_lif_mlp', 'neo_flif_mlp', 'neo_pop_both', 'bidir_lif_both', 'bidir_flif_both', 'bidir_pop_both',
             'mix_lif_bidir', 'mix_flif_bidir', 'mix_pop_bidir', 'mix_pop_fwd', 'mix_pop_bidir_L2')
CLS = ('N', 'S', 'V', 'F', 'Q')


def parse_raw(path):
    rows = []
    for line in open(path):
        if not line.startswith('RESULT '):
            continue
        d = dict(re.findall(r'(\S+?)=(\S+)', line))
        rows.append(d)
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in ('exp', 'cond', 'dataset', 'sha', 'dir')]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['seed'] = df['seed'].astype(int)
    return df.drop_duplicates(subset=['cond', 'seed'], keep='last')


def verdict(df, a, b, col='f1'):
    va, vb = [], []
    for s in SEEDS:
        ra, rb = df[(df.cond == a) & (df.seed == s)], df[(df.cond == b) & (df.seed == s)]
        if len(ra) != 1 or len(rb) != 1:
            return dict(a=a, b=b, verdict='incomplete', diffs='missing seed', mean=float('nan'))
        x, y = float(ra[col].iloc[0]), float(rb[col].iloc[0])
        if not (math.isfinite(x) and math.isfinite(y)):
            return dict(a=a, b=b, verdict='incomplete', diffs='non-finite', mean=float('nan'))
        va.append(x); vb.append(y)
    d = [100 * (x - y) for x, y in zip(va, vb)]
    m = float(np.mean(d))
    if m >= 0.5 and sum(v > 0 for v in d) >= 2:
        v = 'better'
    elif m <= -0.5 and sum(v < 0 for v in d) >= 2:
        v = 'worse'
    else:
        v = 'no clear difference'
    return dict(a=a, b=b, verdict=v, diffs=' / '.join(f'{x:+.2f}' for x in d), mean=m)


def ms(x, scale=100., nd=2):
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or not np.all(np.isfinite(x)):
        return 'n/a'
    return f'{scale * x.mean():.{nd}f} ± {scale * x.std(ddof=1) if len(x) > 1 else 0.:.{nd}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='flif-v1-cls')
    ap.add_argument('--ref_suite', default=None, help='stage 6: an earlier suite whose runs (same code path, same training) join the tables')
    a = ap.parse_args()
    R = os.path.join(HERE, 'results', a.suite)
    df = parse_raw(os.path.join(R, 'raw.txt'))
    suites = {a.suite: R}
    if a.ref_suite:
        RR = os.path.join(HERE, 'results', a.ref_suite)
        ref = parse_raw(os.path.join(RR, 'raw.txt'))
        ref = ref[~ref.cond.isin(set(df.cond))]                       # this suite's own runs win
        df = pd.concat([df, ref], ignore_index=True); suites[a.ref_suite] = RR
    df.to_csv(os.path.join(R, 'runs.csv'), index=False)
    L = [f'# MIT-BIH suite `{a.suite}` — summary' + (f' (with reference runs from `{a.ref_suite}`)' if a.ref_suite else ''), '',
         f"Test file: 32,833 beats (30% split). Last-epoch weights after {'/'.join(str(int(e)) for e in sorted(df.epochs.unique()))} epochs (the ECG framework protocol). "
         'Numbers are mean ± standard deviation over seeds 42 / 7 / 2026, in %.', '']
    # condition table
    L += ['## Performance and energy', '',
          '| condition | n | accuracy | macro-F1 | F1 S | F1 V | F1 F | F1 Q | synaptic µJ | total µJ (FFT / direct) | neocortex firing |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    crow = []
    for c in CONDS:
        r = df[df.cond == c]
        if not len(r):
            continue
        rho = 'n/a' if r['rho_neo'].isna().all() else ms(r['rho_neo'], 1., 4)
        L.append(f"| {LABEL[c]} | {len(r)} | {ms(r.overall_acc)} | {ms(r.f1)} | {ms(r.f1_1)} | {ms(r.f1_2)} | {ms(r.f1_3)} | {ms(r.f1_4)} | "
                 f"{ms(r.syn_energy_uJ, 1.)} | {ms(r.total_fft_uJ, 1.)} / {ms(r.total_direct_uJ, 1.)} | {rho} |")
        crow.append(dict(cond=c, n=len(r), acc_mean=r.overall_acc.mean(), acc_std=r.overall_acc.std(ddof=1), f1_mean=r.f1.mean(),
                         f1_std=r.f1.std(ddof=1), syn_uJ=r.syn_energy_uJ.mean(), total_fft_uJ=r.total_fft_uJ.mean(),
                         total_direct_uJ=r.total_direct_uJ.mean(), time_s=r.time_s.mean()))
    pd.DataFrame(crow).to_csv(os.path.join(R, 'conditions.csv'), index=False, float_format='%.6f')
    # verdicts
    L += ['', '## Pre-registered comparisons (macro-F1, paired by seed, %p)', '',
          '| question | A | B | A − B per seed (42 / 7 / 2026) | mean | verdict |', '|---|---|---|---|---|---|']
    Q = [('Q1 f-LIF vs spikeDE LIF', 'neo_flif_both', 'neo_lif_both'), ('Q1 f-LIF vs spikeDE LIF', 'neo_flif_mlp', 'neo_lif_mlp'),
         ('Q2 f-LIF vs ECG model', 'neo_flif_both', 'base'), ('Q2 f-LIF vs ECG model', 'neo_flif_mlp', 'base'),
         ('Q3 neocortex contribution', 'base', 'no_neo'), ('Q3 neocortex contribution', 'base', 'hippo_only'),
         ('Q4 population f-LIF vs f-LIF', 'neo_pop_both', 'neo_flif_both'),                                 # stage 6
         ('Q5 bidirectional vs forward (f-LIF)', 'bidir_flif_both', 'neo_flif_both'),
         ('Q5 bidirectional vs forward (spikeDE LIF)', 'bidir_lif_both', 'neo_lif_both'),
         ('Q6 both vs f-LIF', 'bidir_pop_both', 'neo_flif_both'), ('Q6 both vs bidirectional f-LIF', 'bidir_pop_both', 'bidir_flif_both'),
         ('Q7 best stage-6 vs hippo-only', 'bidir_pop_both', 'hippo_only'), ('Q7 vs hippo-only', 'bidir_flif_both', 'hippo_only'),
         ('Q7 vs hippo-only', 'neo_pop_both', 'hippo_only'),
         ('Q8 mixer vs no mixing', 'mix_pop_bidir', 'mix_none_fwd'), ('Q8 mixer vs no mixing', 'mix_flif_bidir', 'mix_none_fwd'),   # stage 7
         ('Q8 mixer vs no mixing', 'mix_lif_bidir', 'mix_none_fwd'), ('Q8 mixer vs no mixing', 'mix_pop_fwd', 'mix_none_fwd'),
         ('Q9 mixer vs attention (hippo-only)', 'mix_pop_bidir', 'hippo_only'), ('Q9 mixer vs attention', 'mix_flif_bidir', 'hippo_only'),
         ('Q9 mixer vs attention', 'mix_lif_bidir', 'hippo_only'), ('Q9 mixer vs attention', 'mix_pop_bidir_L2', 'hippo_only'),
         ('Q10 population vs f-LIF (mixer)', 'mix_pop_bidir', 'mix_flif_bidir'), ('Q10 bidirectional vs forward (mixer)', 'mix_pop_bidir', 'mix_pop_fwd'),
         ('Q10 2 blocks vs 1', 'mix_pop_bidir_L2', 'mix_pop_bidir'), ('ref: no mixing vs no_neo', 'mix_none_fwd', 'no_neo')]
    Q = [(q, x, y) for q, x, y in Q if (df.cond == x).any() and (df.cond == y).any()]
    for q, x, y in Q:
        v = verdict(df, x, y)
        L.append(f"| {q} | {LABEL[x]} | {LABEL[y]} | {v['diffs']} | {v['mean']:+.2f} | {v['verdict']} |")
    L += ['', 'Accuracy, same rule:', '', '| A | B | A − B per seed | mean | verdict |', '|---|---|---|---|---|']
    for q, x, y in Q:
        v = verdict(df, x, y, col='overall_acc')
        L.append(f"| {LABEL[x]} | {LABEL[y]} | {v['diffs']} | {v['mean']:+.2f} | {v['verdict']} |")

    # analysis
    AD = os.path.join(R, 'analysis')
    def load(c, s, f):
        for sr in suites.values():                                     # this suite first, then the reference suite
            p = os.path.join(sr, 'analysis', f'{c}_seed{s}', f)
            if os.path.exists(p):
                return pd.read_csv(p)
        return None
    if os.path.isdir(AD):
        L += ['', '## A1 replay tests: macro-F1 change when the neocortex output read by the hippocampus is replaced (%p, mean ± sd over seeds)', '',
              '| condition | normal macro-F1 | another beat | same class | other class | all zero | all one |', '|---|---|---|---|---|---|---|']
        for c in NEO_CONDS:
            tabs = [load(c, s, 'replay.csv') for s in SEEDS]
            tabs = [t for t in tabs if t is not None]
            if not tabs:
                continue
            base = [t.set_index('test').loc['normal', 'macro_f1'] for t in tabs]
            cells = []
            for mode in ('shift', 'same_class', 'other_class', 'zero', 'ones'):
                cells.append(ms([t.set_index('test').loc[mode, 'macro_f1'] - t.set_index('test').loc['normal', 'macro_f1'] for t in tabs]))
            L.append(f"| {LABEL[c]} (n={len(tabs)}) | {ms(base)} | " + ' | '.join(cells) + ' |')
        L += ['', '## A2 linear probes: test macro-F1 of a class-balanced logistic regression (%, mean ± sd over seeds)', '',
              '| condition | raw beat | neocortex embedding spikes | neocortex output spikes | neocortex output, firing per token | hippocampus head input |',
              '|---|---|---|---|---|---|']
        for c in CONDS:
            tabs = [load(c, s, 'probe.csv') for s in SEEDS]
            tabs = [t.set_index('feature') for t in tabs if t is not None]
            if not tabs:
                continue
            def cell(f):
                v = [t.loc[f, 'macro_f1'] for t in tabs if f in t.index]
                return ms(v) if v else '—'
            L.append(f"| {LABEL[c]} (n={len(tabs)}) | {cell('raw')} | {cell('neo_emb_spikes')} | {cell('neo_out_spikes')} | "
                     f"{cell('neo_out_token_rate')} | {cell('hippo_head_input')} |")
        L += ['', '## A3 token-axis memory: share of neocortex output spikes that change d tokens after the replaced token (%, mean over seeds)', '',
              '| condition | ' + ' | '.join(f'd={d}' for d in range(11)) + ' | changed predictions, token 0 | max change before the replaced token (expected > 0 for bidirectional) |',
              '|---|' + '---|' * 13]
        for c in NEO_CONDS:
            tabs = [load(c, s, 'reach_lag.csv') for s in SEEDS]; preds = [load(c, s, 'reach_pred.csv') for s in SEEDS]
            tabs = [t for t in tabs if t is not None]; preds = [p for p in preds if p is not None]
            if not tabs:
                continue
            lag = np.mean([t['out'].values for t in tabs], axis=0)
            pc = np.mean([p.loc[p.token == 0, 'changed_pred'].iloc[0] for p in preds]) if preds else float('nan')
            ac = []
            for s in SEEDS:
                for sr in suites.values():
                    p = os.path.join(sr, 'analysis', f'{c}_seed{s}', 'meta.json')
                    if os.path.exists(p):
                        import json
                        ac.append(json.load(open(p)).get('acausal_max', float('nan'))); break
            L.append(f"| {LABEL[c]} (n={len(tabs)}) | " + ' | '.join(f'{100 * v:.2f}' for v in lag) + f" | {100 * pc:.2f} | {max(ac) if ac else float('nan'):.1e} |")
        L += ['', '## A4 neocortex output firing per token, by class (%, mean over seeds; tokens = 32-sample patches, stride 16)', '']
        for c in ('base', 'neo_lif_both', 'neo_flif_both', 'neo_pop_both', 'bidir_flif_both', 'mix_pop_bidir', 'mix_flif_bidir'):
            tabs = [load(c, s, 'firing_by_class.csv') for s in SEEDS]
            tabs = [t for t in tabs if t is not None]
            if not tabs:
                continue
            L += [f'{LABEL[c]} (n={len(tabs)}):', '', '| class | ' + ' | '.join(f't{t}' for t in range(11)) + ' |', '|---|' + '---|' * 11]
            for k in range(5):
                vals = np.mean([[t.loc[t.cls == k, f'out_t{j}'].iloc[0] for j in range(11)] for t in tabs], axis=0)
                L.append(f'| {CLS[k]} | ' + ' | '.join(f'{100 * v:.1f}' for v in vals) + ' |')
            L.append('')
    open(os.path.join(R, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
