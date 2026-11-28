"""Summary of the SF-de position-information-off test (2026-11-28, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; user "SF-de의 위치
정보 끄기(ssa_pe 0): MIT-BIH 3 run, ETTh1/m1 12 run 수행하고 비교"; rules fixed before training).
SF-de = Spikformer-style (SSA, spikeDE LIF everywhere, NALL=lif). Its only position information in the body is the SSA relative
position bias; with --ssa_pe 0 (runner SSAPE=0, tag _pe0) the SSA is permutation-equivariant, so on MIT-BIH (token-mean head) the
whole model is order-blind, and on ETT only the flat forecast head knows the token positions.
Arms (paired seeds 42 / 7 / 2026):
  MIT-BIH  nope  SF-de PE off  flif-v1-mitbih-sfde-nope   spk_ssa_lr0.01_L2_nalif_pe0   (new, 3 rows)
           sfde  SF-de PE on   flif-v1-mitbih-sfde        spk_ssa_lr0.01_L2_nalif
           mux   SpkMixer muxlin ADD, LIF memory      flif-v1-mitbih-spkmux-tune  spkmx_lif_muxlin_add_lr0.02_L2
           off   SpkMixer muxlin ADD, memoryless      flif-v1-mitbih-memoff       spkmx_lif_muxlin_add_lr0.02_L2_tmoff
  ETT      nope  SF-de PE off  flif-v1-ett-sfde-nope      ETTh1 spk_ssa_lr0.0005_L1_nalif_pe0, ETTm1 spk_ssa_lr0.0001_L2_pt16_nalif_pe0 (new, 12)
           sfde  SF-de PE on   flif-v1-ett-sfde-h1m1      the same tags without _pe0
           mux   muxlin IAND one block (retune reference)  flif-v1-ett-spkmux-retune  (REF_TAG below)
           off   muxlin IAND memoryless                     flif-v1-ett-memoff          REF_TAG + _tmoff
Comparisons (A vs B):
  MIT-BIH  M1 nope vs sfde (does SF-de need the position bias?)   M2 MAIN mux vs nope (no explicit position information in either body:
           neuron dynamics vs order-blind attention)   M3 report off vs nope (both order-blind)   M4 report mux vs sfde
           d = 100 (F1_A - F1_B) per seed, %p; 'A better' if mean >= +0.5 and >= 2 of 3 seeds > 0, 'A worse' if mean <= -0.5 and
           >= 2 of 3 < 0, else 'no clear difference'
  ETT      E1 nope vs sfde   E2 MAIN mux vs nope   E3 report off vs nope
           r = 100 (MSE_A - MSE_B) / MSE_B per (cell, seed), R = mean over the 4 cells of the seed-mean r; 'A better' if R <= -1 % and
           >= 8 of 12 lower, 'A worse' if R >= +1 % and >= 8 higher, else 'no clear difference' (MAE the same way, reported)
  energy   syops (logged syn_energy_uJ), ratios of seed means (ETT: mean over cells of the cell ratios)
Checks (fail closed, exit 2, nothing written): SKIP line or a row of another tag in a new suite, not exactly the wanted (cell, seed)
rows, duplicates, non-finite values, mixed training commits in a new suite, missing energy recount rows, recomputed macro-F1 != logged
(MIT-BIH, 1e-6), the ETT energy identity off (body_syn_uJ + head_MAC x 4.6 pJ, 1e-3); order shuffles of the trained PE-off models:
MIT-BIH SSA and full shuffle |drop| < 0.05 %p every run (the model is order-blind), ETT SSA shuffle |change| < 0.05 % every run.
usage (from cinc_mixer_v1/): python -B results/check-sfnope-20261128/summarize_sfnope.py [--root results] [--order_dir ...] [--out_dir ...]
"""
import argparse
import csv
import json
import math
import os
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--root', default=os.path.join(C, 'results'))
ap.add_argument('--order_dir', default=HERE)
ap.add_argument('--out_dir', default=HERE)
a = ap.parse_args()
SEEDS = ('42', '7', '2026')
CELLS = [(d, p) for d in ('ETTh1', 'ETTm1') for p in ('96', '720')]
REF_TAG = {'ETTh1': 'spkmx_lif_muxlin_iand_lr0.00025_L1_e32_mlp4_wd0.1', 'ETTm1': 'spkmx_lif_muxlin_iand_lr0.0005_L2_pt16_e32_mlp0.5'}
SF_TAG = {'ETTh1': 'spk_ssa_lr0.0005_L1_nalif', 'ETTm1': 'spk_ssa_lr0.0001_L2_pt16_nalif'}
mean = lambda x: sum(x) / len(x)


def stop(msg):
    print(f'STOP: {msg}')
    sys.exit(2)


def fin(r, k, where):
    try:
        v = float(r[k])
    except (KeyError, ValueError, TypeError):
        stop(f'{k} missing ({where})')
    if not math.isfinite(v):
        stop(f'{k} not finite ({where})')
    return v


def rows_of(suite):
    p = os.path.join(a.root, suite, 'raw.txt')
    if not os.path.exists(p):
        stop(f'{p} missing')
    lines = open(p).read().splitlines()
    return [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in lines if l.startswith('RESULT')], sum(l.startswith('SKIP') for l in lines)


# ---------------- MIT-BIH ----------------
MKEYS = ('f1', 'overall_acc', 'f1_1', 'syn_energy_uJ')


def load_m(suite, tag, new=False):
    rows, nskip = rows_of(suite)
    if new:
        if nskip:
            stop(f'{suite}: {nskip} SKIP line(s)')
        other = [r.get('tag') for r in rows if r.get('tag') != tag]
        if other:
            stop(f'{suite}: unexpected rows {other}')
    V, shas = {}, set()
    for r in rows:
        if r.get('tag') != tag:
            continue
        s = r.get('seed')
        if s in V:
            stop(f'duplicate {suite} {tag} seed {s}')
        V[s] = {k: fin(r, k, f'{suite} {tag} seed {s}') for k in MKEYS}
        shas.add(r.get('sha'))
    if set(V) != set(SEEDS):
        stop(f'{suite} {tag}: seeds {sorted(V)} (want {sorted(SEEDS)})')
    if new:
        if len(shas) != 1:
            stop(f'{suite}: mixed training commits {sorted(shas)}')
        ep = os.path.join(a.root, suite, 'energy_recount.csv')
        if not os.path.exists(ep):
            stop(f'{suite}: energy_recount.csv missing')
        en = {e['seed']: e for e in csv.DictReader(open(ep)) if e.get('tag') == tag}
        for s in SEEDS:
            if s not in en:
                stop(f'{suite}: no energy recount row for seed {s}')
            if abs(fin(en[s], 'f1_recomputed', f'{suite} recount {s}') - V[s]['f1']) > 1e-6:
                stop(f'{suite} seed {s}: recomputed macro-F1 != logged')
    return V, shas


MN, mshas = load_m('flif-v1-mitbih-sfde-nope', 'spk_ssa_lr0.01_L2_nalif_pe0', new=True)
MS, _ = load_m('flif-v1-mitbih-sfde', 'spk_ssa_lr0.01_L2_nalif')
MM, _ = load_m('flif-v1-mitbih-spkmux-tune', 'spkmx_lif_muxlin_add_lr0.02_L2')
MO, _ = load_m('flif-v1-mitbih-memoff', 'spkmx_lif_muxlin_add_lr0.02_L2_tmoff')
MARMS = {'SF-de PE off (new)': MN, 'SF-de PE on': MS, 'SpkMixer LIF memory': MM, 'SpkMixer memoryless': MO}

op = os.path.join(a.order_dir, 'order_sfdenope.json')
if not os.path.exists(op):
    stop(f'{op} missing')
OM = json.load(open(op))
if OM.get('tag') != 'spk_ssa_lr0.01_L2_nalif_pe0' or sorted(x['seed'] for x in OM['runs']) != sorted(SEEDS):
    stop('order_sfdenope.json: wrong arm or seeds')
for x in OM['runs']:
    for var in ('mixer', 'full'):
        dr = x[var]['drop_f1_pp']
        if not (math.isfinite(dr) and abs(dr) < 0.05):
            stop(f'MIT-BIH PE-off seed {x["seed"]}: {var} shuffle drop {dr} %p (an order-blind model must give ~0)')


def mcomp(A, B):
    d = [100. * (A[s]['f1'] - B[s]['f1']) for s in SEEDS]
    m = mean(d)
    v = ('A better' if m >= 0.5 and sum(x > 0 for x in d) >= 2 else
         ('A worse' if m <= -0.5 and sum(x < 0 for x in d) >= 2 else 'no clear difference'))
    return dict(d_pp=d, mean_pp=m, verdict=v, energy_ratio=mean([A[s]['syn_energy_uJ'] for s in SEEDS]) / mean([B[s]['syn_energy_uJ'] for s in SEEDS]))


MCOMPS = [('M1', 'SF-de PE off (new)', 'SF-de PE on'), ('M2 (main)', 'SpkMixer LIF memory', 'SF-de PE off (new)'),
          ('M3', 'SpkMixer memoryless', 'SF-de PE off (new)'), ('M4', 'SpkMixer LIF memory', 'SF-de PE on')]

# ---------------- ETT ----------------
ENUM = ('mse', 'mae', 'syn_energy_uJ')


def load_e(suite, pick, new=False, tagf=None):
    rows, nskip = rows_of(suite)
    if new and nskip:
        stop(f'{suite}: {nskip} SKIP line(s)')
    V, shas = {}, set()
    en = None
    if new:
        ep = os.path.join(a.root, suite, 'energy_recount.csv')
        if not os.path.exists(ep):
            stop(f'{suite}: energy_recount.csv missing')
        en = {(e['tag'], e['ds'], e['pl'], e['seed']): e for e in csv.DictReader(open(ep))}
    for r in rows:
        if not pick(r):
            if new:
                stop(f'{suite}: unexpected row {r.get("tag")} {r.get("ds")} H{r.get("pl")}')
            continue
        k = (r['ds'], r['pl'], r['seed'])
        if k in V:
            stop(f'duplicate {suite} {k}')
        w = f'{suite} {k}'
        V[k] = {m: fin(r, m, w) for m in ENUM}
        shas.add(r.get('sha'))
        if new:
            e = en.get((r['tag'], r['ds'], r['pl'], r['seed']))
            if e is None:
                stop(f'{suite}: no energy recount row for {k}')
            if abs(fin(e, 'body_syn_uJ', w) + fin(e, 'head_MAC', w) * 4.6e-6 - V[k]['syn_energy_uJ']) > 1e-3 * V[k]['syn_energy_uJ']:
                stop(f'{suite}: energy identity off for {k}')
    want = {(d, p, s) for d, p in CELLS for s in SEEDS}
    if set(V) != want:
        stop(f'{suite}: {len(V)} of 12 wanted runs')
    if new and len(shas) != 1:
        stop(f'{suite}: mixed training commits {sorted(shas)}')
    return V, shas


EN, eshas = load_e('flif-v1-ett-sfde-nope', lambda r: r.get('cond') == 'spk_ssa' and r['ds'] in SF_TAG and r.get('tag') == SF_TAG[r['ds']] + '_pe0'
                   and r['pl'] in ('96', '720'), new=True)
ES, _ = load_e('flif-v1-ett-sfde-h1m1', lambda r: r.get('cond') == 'spk_ssa' and r['ds'] in SF_TAG and r.get('tag') == SF_TAG[r['ds']] and r['pl'] in ('96', '720'))
EM, _ = load_e('flif-v1-ett-spkmux-retune', lambda r: r['ds'] in REF_TAG and r.get('tag') == REF_TAG[r['ds']] and r['pl'] in ('96', '720') and r.get('blocks') == '1')
EO, _ = load_e('flif-v1-ett-memoff', lambda r: r['ds'] in REF_TAG and r.get('tag') == REF_TAG[r['ds']] + '_tmoff' and r['pl'] in ('96', '720'))
EARMS = {'SF-de PE off (new)': EN, 'SF-de PE on': ES, 'muxlin IAND LIF memory': EM, 'muxlin IAND memoryless': EO}

oe = os.path.join(a.order_dir, 'order_sfde1n.json')
if not os.path.exists(oe):
    stop(f'{oe} missing')
OE = json.load(open(oe))
if sorted((x['ds'], x['pl'], x['seed']) for x in OE['runs']) != sorted((d, p, s) for d, p in CELLS for s in SEEDS):
    stop('order_sfde1n.json: wrong runs')
for x in OE['runs']:
    dm = x['mixer']['d_mse_pct']
    if not (math.isfinite(dm) and abs(dm) < 0.05):
        stop(f'ETT PE-off {x["ds"]} H{x["pl"]} seed {x["seed"]}: SSA shuffle changes MSE by {dm} % (an equivariant SSA must give ~0)')


def ecomp(A, B, met):
    rr, lo, hi = [], 0, 0
    for d, p in CELLS:
        rr.append(mean([100. * (A[(d, p, s)][met] - B[(d, p, s)][met]) / B[(d, p, s)][met] for s in SEEDS]))
        lo += sum(A[(d, p, s)][met] < B[(d, p, s)][met] for s in SEEDS)
        hi += sum(A[(d, p, s)][met] > B[(d, p, s)][met] for s in SEEDS)
    R = mean(rr)
    return dict(R_pct=R, lower=lo, higher=hi, r_cells=rr,
                verdict='A better' if R <= -1 and lo >= 8 else ('A worse' if R >= 1 and hi >= 8 else 'no clear difference'))


def eer(A, B):
    return mean([mean([A[(d, p, s)]['syn_energy_uJ'] for s in SEEDS]) / mean([B[(d, p, s)]['syn_energy_uJ'] for s in SEEDS]) for d, p in CELLS])


ECOMPS = [('E1', 'SF-de PE off (new)', 'SF-de PE on'), ('E2 (main)', 'muxlin IAND LIF memory', 'SF-de PE off (new)'),
          ('E3', 'muxlin IAND memoryless', 'SF-de PE off (new)')]

# ---------------- output ----------------
out = dict(mitbih=dict(sha=sorted(mshas)[0], arms={n: {k: mean([V[s][k] for s in SEEDS]) for k in MKEYS} for n, V in MARMS.items()},
                       comparisons=[dict(kind=k, A=x, B=y, **mcomp(MARMS[x], MARMS[y])) for k, x, y in MCOMPS],
                       order=[dict(seed=x['seed'], ssa_drop_pp=x['mixer']['drop_f1_pp'], full_drop_pp=x['full']['drop_f1_pp']) for x in OM['runs']]),
           ett=dict(sha=sorted(eshas)[0],
                    mean4={n: {m: mean([mean([V[(d, p, s)][m] for s in SEEDS]) for d, p in CELLS]) for m in ENUM} for n, V in EARMS.items()},
                    cells={n: {f'{d} {p}': {m: mean([V[(d, p, s)][m] for s in SEEDS]) for m in ENUM} for d, p in CELLS} for n, V in EARMS.items()},
                    comparisons=[dict(kind=k, A=x, B=y, mse=ecomp(EARMS[x], EARMS[y], 'mse'), mae=ecomp(EARMS[x], EARMS[y], 'mae'),
                                      energy_ratio=eer(EARMS[x], EARMS[y])) for k, x, y in ECOMPS],
                    order={v: dict(mean_pct=mean([x[v]['d_mse_pct'] for x in OE['runs']]),
                                   by_ds={d: mean([x[v]['d_mse_pct'] for x in OE['runs'] if x['ds'] == d]) for d in ('ETTh1', 'ETTm1')},
                                   max_abs_pct=max(abs(x[v]['d_mse_pct']) for x in OE['runs'])) for v in ('mixer', 'head', 'full')}))
os.makedirs(a.out_dir, exist_ok=True)
json.dump(out, open(os.path.join(a.out_dir, 'summary_sfnope.json'), 'w'), indent=1)
L = ['# SF-de without position information (ssa_pe 0) on MIT-BIH and ETTh1 / ETTm1', '',
     f'Rules: summarize_sfnope.py docstring. New runs: MIT-BIH 3 (commit `{out["mitbih"]["sha"]}`), ETT 12 (commit `{out["ett"]["sha"]}`).', '',
     '## MIT-BIH (seed means)', '', '| arm | macro-F1 % | accuracy % | class S F1 % | energy µJ (syops) |', '|---|--:|--:|--:|--:|']
for n in MARMS:
    m = out['mitbih']['arms'][n]
    L.append(f'| {n} | {100 * m["f1"]:.2f} | {100 * m["overall_acc"]:.2f} | {100 * m["f1_1"]:.2f} | {m["syn_energy_uJ"]:.2f} |')
L += ['', '| kind | A | B | d %p (42 / 7 / 2026) | mean %p | verdict | energy A/B |', '|---|---|---|---|--:|---|--:|']
for c in out['mitbih']['comparisons']:
    L.append(f'| {c["kind"]} | {c["A"]} | {c["B"]} | ' + ' / '.join(f'{x:+.2f}' for x in c['d_pp']) +
             f' | {c["mean_pp"]:+.2f} | **{c["verdict"]}** | {c["energy_ratio"]:.3f} |')
L += ['', 'Order check (trained PE-off SF-de, mean of 3 permutation seeds): SSA shuffle drop ' +
      ' / '.join(f'{x["ssa_drop_pp"]:+.4f}' for x in out['mitbih']['order']) + ' %p, full shuffle drop ' +
      ' / '.join(f'{x["full_drop_pp"]:+.4f}' for x in out['mitbih']['order']) + ' %p (|drop| < 0.05 required)', '',
      '## ETT (4-cell means of seed means)', '', '| arm | MSE | MAE | energy µJ (syops) |', '|---|--:|--:|--:|']
for n in EARMS:
    m = out['ett']['mean4'][n]
    L.append(f'| {n} | {m["mse"]:.4f} | {m["mae"]:.4f} | {m["syn_energy_uJ"]:.2f} |')
L += ['', '| kind | A | B | MSE R % | lower / higher | MSE verdict | MAE R % | MAE verdict | energy A/B |', '|---|---|---|--:|---|---|--:|---|--:|']
for c in out['ett']['comparisons']:
    L.append(f'| {c["kind"]} | {c["A"]} | {c["B"]} | {c["mse"]["R_pct"]:+.2f} | {c["mse"]["lower"]} / {c["mse"]["higher"]} | **{c["mse"]["verdict"]}** | '
             f'{c["mae"]["R_pct"]:+.2f} | {c["mae"]["verdict"]} | {c["energy_ratio"]:.3f} |')
L += ['', '| arm | ' + ' | '.join(f'{d} H{p}' for d, p in CELLS) + ' |', '|---|' + '---|' * len(CELLS)]
for n in EARMS:
    cc = out['ett']['cells'][n]
    L.append(f'| {n} | ' + ' | '.join(f'{cc[f"{d} {p}"]["mse"]:.4f} / {cc[f"{d} {p}"]["mae"]:.4f}' for d, p in CELLS) + ' |')
L += ['', 'Order shuffle of the trained PE-off SF-de (test MSE change %, mean over 12 runs; ETTh1 / ETTm1): ' +
      '; '.join(f'{v} {o["mean_pct"]:+.2f} ({o["by_ds"]["ETTh1"]:+.2f} / {o["by_ds"]["ETTm1"]:+.2f}, max |.| {o["max_abs_pct"]:.3f})'
                for v, o in out['ett']['order'].items()) + ' (SSA shuffle |change| < 0.05 % required)']
open(os.path.join(a.out_dir, 'summary_sfnope.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
