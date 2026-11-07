"""Verdict tables of the test-time query replacement (2026-11-07, exp/f-lif-pop-v3; PROJECT_LOG query-swap pre-registration). No model and no checkpoint is read: only results/qswap/runs/*.json and the RESULT lines.
Fail closed (exit 2, nothing written): a missing, extra or smoke run; a run whose name, seed or cell disagrees with the file name; a failed check field; a clean metric that differs from the RESULT line;
a missing or non-finite value before any comparison.
Rules (pre-registered, per neuron and condition; `other` and `perm` = mean of the 3 draws of a checkpoint):
  ETT      r = 100 (MSE_replaced - MSE_clean) / MSE_clean per pair, R = mean of the 4 cell means. used: R >= +1 % and >= 8/12 pairs higher; lower: R <= -1 % and >= 8/12 pairs lower; else no clear effect
  MIT-BIH  d = macro-F1 replaced - clean (%p) per seed, mean of 3. used: <= -0.5 %p and >= 2/3 seeds lower; higher: >= +0.5 %p and >= 2/3 seeds higher; else no clear effect
Report only: per-cell values, MAE, per-class F1, draw spread, firing rates, bit agreement, Neocortex output rate by token step.
usage: python -B summarize_qswap.py [--dry DIR]   ->   results/qswap/summary.md (or DIR/summary.md)
"""
import argparse
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'results', 'qswap', 'runs')
NEURONS = ('pop_lif', 'pop_flif', 'lif', 'flif')
NAME = {'pop_lif': 'population LIF', 'pop_flif': 'population f-LIF', 'lif': 'LIF', 'flif': 'f-LIF'}
SEEDS = (42, 7, 2026)
CELLS = (('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720))
SWAPS = ('other', 'perm', 'zero', 'attn_off')
SNAME = {'other': 'other (window content)', 'perm': 'perm (token order)', 'zero': 'zero (Neocortex output 0)', 'attn_off': 'attn_off (attention term removed)'}
CLS = ('N', 'S', 'V', 'F', 'Q')


def stop(msg):
    print('STOP:', msg)
    sys.exit(2)


def fin(v, what):
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        stop(f'{what}: {v}')
    return float(v)


def mean(v):
    if not v:
        stop('mean of nothing')
    return sum(v) / len(v)


def result_line(raw, run_dir):
    hit = [ln for ln in open(raw) if ln.startswith('RESULT') and ln.rstrip().endswith('dir=' + run_dir.rstrip('/'))]
    if len(hit) != 1:
        stop(f'{len(hit)} RESULT lines for {run_dir}')
    return dict(re.findall(r'(\w+)=(\S+)', hit[0]))


def load(runs=RUNS):
    got = {}
    want = {f'ett_{d}_H{p}_{n}_s{s}' for d, p in CELLS for n in NEURONS for s in SEEDS} | {f'mitbih_{n}_s{s}' for n in NEURONS for s in SEEDS}
    for f in glob.glob(os.path.join(runs, '*.json')):
        k = os.path.basename(f)[:-5]
        if k not in want:
            stop(f'unexpected run {k}')
        r = json.load(open(f))
        if r.get('name') != k or r.get('smoke') or 'metrics' not in r:
            stop(f'{k}: wrong name, smoke run or no metrics')
        c = r['checks']
        if not (c['split_max_abs'] <= 1e-5 and c['neo_repeat_equal'] is True and c['other_min_distance'] >= c['gap'] and c['perm_rows_changed'] > 0 and c['zero_all_zero'] is True and c['other_bits_changed'] > 0):
            stop(f'{k}: failed check {c}')
        ref = result_line(r['raw'], r['run_dir'])
        n = re.search(r'_(pop_flif|pop_lif|flif|lif)_s(\d+)$', k)
        if ref['cond'] != f'nsmt4q_{n[1]}' or int(ref['seed']) != int(n[2]):
            stop(f'{k}: RESULT line is {ref["cond"]} seed {ref["seed"]}')
        m = r['metrics']
        if set(m) != {'clean', 'zero', 'attn_off', 'other', 'perm'} or len(m['other']) != 3 or len(m['perm']) != 3:
            stop(f'{k}: conditions {sorted(m)}')
        if r['task'] == 'ett':
            if f"ett_{ref['ds']}_H{int(ref['pl'])}_" not in k + '_':
                stop(f'{k}: RESULT line is {ref["ds"]} H{ref["pl"]}')
            if abs(fin(m['clean']['mse'], k) - float(ref['mse'])) / float(ref['mse']) > 1e-5:
                stop(f'{k}: clean MSE {m["clean"]["mse"]} vs {ref["mse"]}')
        elif abs(fin(m['clean']['f1'], k) - float(ref['f1'])) > 1e-6:
            stop(f'{k}: clean macro-F1 {m["clean"]["f1"]} vs {ref["f1"]}')
        got[k] = r
    if set(got) != want:
        stop(f'missing runs: {sorted(want - set(got))[:6]} ({len(want - set(got))})')
    return got


def val(r, sw, key):
    m = r['metrics'][sw]
    return mean([fin(x[key], f"{r['name']} {sw} {key}") for x in m]) if isinstance(m, list) else fin(m[key], f"{r['name']} {sw} {key}")


def ett_table(run, key='mse'):
    """{(neuron, swap): dict(R, up, down, cell={cell: mean r}, pairs=[...])}"""
    out = {}
    for n in NEURONS:
        for sw in SWAPS:
            cell, pairs = {}, []
            for d, p in CELLS:
                rs = []
                for s in SEEDS:
                    r = run[f'ett_{d}_H{p}_{n}_s{s}']
                    c = fin(r['metrics']['clean'][key], r['name'])
                    if c <= 0:
                        stop(f"{r['name']}: clean {key} {c}")
                    rs.append(100 * (val(r, sw, key) - c) / c)
                cell[(d, p)] = mean(rs); pairs += rs
            R = mean(list(cell.values()))
            up, down = sum(x > 0 for x in pairs), sum(x < 0 for x in pairs)
            v = 'used' if R >= 1 and up >= 8 else ('lower' if R <= -1 and down >= 8 else 'no clear effect')
            out[n, sw] = dict(R=R, up=up, down=down, cell=cell, pairs=pairs, verdict=v)
    return out


def mit_table(run):
    out = {}
    for n in NEURONS:
        for sw in SWAPS:
            ds, cls = [], [[] for _ in CLS]
            for s in SEEDS:
                r = run[f'mitbih_{n}_s{s}']
                c = r['metrics']['clean']
                ds.append(100 * (val(r, sw, 'f1') - fin(c['f1'], r['name'])))
                m = r['metrics'][sw]
                for i in range(len(CLS)):
                    v = mean([x['f1_cls'][i] for x in m]) if isinstance(m, list) else m['f1_cls'][i]
                    cls[i].append(100 * (fin(v, r['name']) - fin(c['f1_cls'][i], r['name'])))
            d = mean(ds)
            lo, hi = sum(x < 0 for x in ds), sum(x > 0 for x in ds)
            v = 'used' if d <= -0.5 and lo >= 2 else ('higher' if d >= 0.5 and hi >= 2 else 'no clear effect')
            out[n, sw] = dict(d=d, lo=lo, hi=hi, seeds=ds, cls=[mean(x) for x in cls], verdict=v)
    return out


def of(run, task, n):
    """The runs of exactly one neuron (fullmatch: 'lif' must not pick up pop_lif)."""
    rr = [r for k, r in run.items() if r['task'] == task and re.fullmatch(rf'(mitbih|ett_ETT[hm]2_H\d+)_{n}_s\d+', k)]
    if len(rr) != (12 if task == 'ett' else 3):
        stop(f'{len(rr)} runs of {n} on {task}')
    return rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry')
    a = ap.parse_args()
    run = load()
    E, EA, M = ett_table(run), ett_table(run, 'mae'), mit_table(run)
    L = ['# Test-time replacement of the Neocortex query in the plain query model', '',
         'Generated by summarize_qswap.py from results/qswap/runs/*.json (60 checkpoints; no training, no selection). Changes relative to the clean forward of the same checkpoint.', '',
         '## ETT: test MSE, R % (mean of the 4 cells), pairs higher of 12, verdict; per cell ETTh2 H96 / ETTh2 H720 / ETTm2 H96 / ETTm2 H720', '',
         '| neuron | condition | R % | pairs higher | verdict | per cell % | MAE R % |', '|---|---|--:|--:|---|---|--:|']
    for n in NEURONS:
        for sw in SWAPS:
            e = E[n, sw]
            L.append(f"| {NAME[n]} | {SNAME[sw]} | {e['R']:+.2f} | {e['up']}/12 | {e['verdict']} | " + ' / '.join(f'{e["cell"][c]:+.2f}' for c in CELLS) + f" | {EA[n, sw]['R']:+.2f} |")
    L += ['', '## MIT-BIH: macro-F1, change in %p (mean of 3 seeds), seeds lower, verdict; per seed 42 / 7 / 2026; per class N / S / V / F / Q', '',
          '| neuron | condition | change %p | seeds lower | verdict | per seed | per class %p |', '|---|---|--:|--:|---|---|---|']
    for n in NEURONS:
        for sw in SWAPS:
            e = M[n, sw]
            L.append(f"| {NAME[n]} | {SNAME[sw]} | {e['d']:+.2f} | {e['lo']}/3 | {e['verdict']} | " + ' / '.join(f'{x:+.2f}' for x in e['seeds']) + ' | ' + ' / '.join(f'{x:+.2f}' for x in e['cls']) + ' |')
    L += ['', '## Clean metric (reproduces the RESULT lines)', '', '| neuron | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | MIT-BIH macro-F1 % |', '|---|--:|--:|--:|--:|--:|']
    for n in NEURONS:
        L.append(f'| {NAME[n]} | ' + ' | '.join(f"{mean([run[f'ett_{d}_H{p}_{n}_s{s}']['metrics']['clean']['mse'] for s in SEEDS]):.4f}" for d, p in CELLS)
                 + f" | {100 * mean([run[f'mitbih_{n}_s{s}']['metrics']['clean']['f1'] for s in SEEDS]):.2f} |")
    L += ['', '## Report only: spread of the 3 draws (largest |draw - mean of draws| of any checkpoint, ETT % of the clean MSE, MIT-BIH %p)', '', '| condition | ETT | MIT-BIH |', '|---|--:|--:|']
    for sw in ('other', 'perm'):
        se, sm = 0., 0.
        for r in run.values():
            key = 'mse' if r['task'] == 'ett' else 'f1'
            v = [x[key] for x in r['metrics'][sw]]
            sp = max(abs(x - mean(v)) for x in v)
            if r['task'] == 'ett':
                se = max(se, 100 * sp / r['metrics']['clean']['mse'])
            else:
                sm = max(sm, 100 * sp)
        L.append(f'| {sw} | {se:.2f} | {sm:.2f} |')
    L += ['', '## Report only: firing rates (mean over seeds and, for ETT, cells)', '',
          '| task | neuron | Neocortex output | bit agreement with another window / chance | query neuron clean / other / perm / zero | attention term clean / other / perm / zero | Hippocampus output clean / attn_off |',
          '|---|---|--:|--:|--:|--:|--:|']
    for task in ('ett', 'mitbih'):
        for n in NEURONS:
            rr = of(run, task, n)
            ag = lambda f: mean([f(r) for r in rr])
            q = ' / '.join(f"{ag(lambda r: val(r, c, 'rate_query')):.4f}" for c in ('clean', 'other', 'perm', 'zero'))
            at = ' / '.join(f"{ag(lambda r: val(r, c, 'rate_attn')):.4f}" for c in ('clean', 'other', 'perm', 'zero'))
            hp = ' / '.join(f"{ag(lambda r: val(r, c, 'rate_hippo')):.4f}" for c in ('clean', 'attn_off'))
            L.append(f"| {'ETT' if task == 'ett' else 'MIT-BIH'} | {NAME[n]} | {ag(lambda r: r['neo']['rate']):.4f} | {ag(lambda r: r['neo']['agree_other']):.4f} / {ag(lambda r: r['neo']['agree_chance']):.4f} | {q} | {at} | {hp} |")
    L += ['', '## Report only: Neocortex output firing rate by token step (mean over seeds and, for ETT, cells)', '']
    for task, N in (('ett', None), ('mitbih', None)):
        rr0 = [r for r in run.values() if r['task'] == task]
        N = rr0[0]['N']
        steps = list(range(N)) if N <= 12 else [0, 1, 2, 3, 4, 6, 8, 12, 16, 20, N - 1]
        L += [f"### {'ETT' if task == 'ett' else 'MIT-BIH'} ({N} token steps)", '', '| neuron | ' + ' | '.join(str(s) for s in steps) + ' | first step at half of the last-third mean |', '|---|' + '--:|' * (len(steps) + 1)]
        for n in NEURONS:
            rr = of(run, task, n)
            if {len(r['neo']['rate_by_token']) for r in rr} != {N}:
                stop('token count differs')
            v = [mean([r['neo']['rate_by_token'][i] for r in rr]) for i in range(N)]
            tail = mean(v[-max(1, N // 3):])
            half = next((i for i in range(N) if v[i] >= tail / 2), None)
            L.append(f'| {NAME[n]} | ' + ' | '.join(f'{v[s]:.3f}' for s in steps) + f' | {half} |')
        L.append('')
    d = a.dry or os.path.join(HERE, 'results', 'qswap')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'summary.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('written', os.path.join(d, 'summary.md'))


if __name__ == '__main__':
    main()
