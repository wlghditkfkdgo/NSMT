"""Summary of the MIT-BIH residual test (2026-11-28, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; user "1과 2를 진행해줘", item 1
"MIT-BIH SpkMixer의 energy 줄이기"; rules fixed before training).
The ADD residual (x + m) gives integer inputs to the next Linear, which syops counts as MAC (4.6 pJ) instead of AC (0.9 pJ); MIT-BIH
SpkMixer ADD = 37.06 uJ = 4.75x SF-de. Two binary-input residuals, each with its own lr (select_mitres.py, validation only), retrained on
the whole training file (suite flif-v1-mitbih-res, val_fold -1, seeds 42 / 7 / 2026, 30 epochs, last-epoch weights, test 30 %):
  IAND  x (1 - m)                    (the ETT base residual)
  MS    membrane shortcut: the sum stays in a membrane u, only spikes reach a Linear (connect_f MS, 11-08)
References (paired seeds): ADD base spkmx_lif_muxlin_add_lr0.02_L2 (flif-v1-mitbih-spkmux-tune), SF-de PE on spk_ssa_lr0.01_L2_nalif
(flif-v1-mitbih-sfde), SF-de PE off spk_ssa_lr0.01_L2_nalif_pe0 (flif-v1-mitbih-sfde-nope); report: the 11-08 MS rows at lr 0.02
(flif-v1-mitbih-spkmux-ms).
Comparisons (A vs B): R1 IAND vs ADD, R2 MS vs ADD, R3 IAND vs SF-de, R4 MS vs SF-de, R5 IAND vs MS (report), R6 / R7 IAND / MS vs SF-de
PE off (report). d = 100 (F1_A - F1_B) per seed (%p); 'A better' if mean >= +0.5 and >= 2 of 3 seeds > 0, 'A worse' mirror, else 'no
clear difference'. Energy = logged syops syn_energy_uJ, ratio of seed means. Adoption bar (user 11-27) vs SF-de: mean d >= +0.1 %p and
energy <= 1.5x.
Recommendation rule: candidates = residuals whose R vs ADD is not 'A worse' AND whose energy <= 1.5x SF-de; IAND if it is a candidate
(the same residual as the ETT base -> one model for both tasks), else MS if it is, else none (keep ADD).
Fails closed (exit 2, nothing written): SKIP line, not exactly 24 validation + 6 final rows of the expected (cond, lr, seed), selected_lr
missing or not matching the final rows, mixed commits, non-finite values, missing energy recount rows or recomputed macro-F1 != logged.
usage (from cinc_mixer_v1/): python -B results/check-mitres-20261128/summarize_mitres.py [--root results] [--out_dir ...]
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
ap.add_argument('--out_dir', default=HERE)
a = ap.parse_args()
SEEDS = ('42', '7', '2026')
LRS = ('0.0025', '0.005', '0.01', '0.02')
RES = {'IAND': 'spkmx_lif_muxlin_iand', 'MS': 'spkmx_lif_muxlin_ms'}
KEYS = ('f1', 'overall_acc', 'f1_0', 'f1_1', 'f1_2', 'f1_3', 'f1_4', 'syn_energy_uJ')
mean = lambda x: sum(x) / len(x)


def stop(m):
    print(f'STOP: {m}')
    sys.exit(2)


def fin(r, k):
    try:
        v = float(r[k])
    except (KeyError, ValueError, TypeError):
        stop(f'{k} missing ({r.get("tag")} seed {r.get("seed")})')
    if not math.isfinite(v):
        stop(f'{k} not finite ({r.get("tag")} seed {r.get("seed")})')
    return v


def raw(suite):
    p = os.path.join(a.root, suite, 'raw.txt')
    if not os.path.exists(p):
        stop(f'{p} missing')
    L = open(p).read().splitlines()
    return [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in L if l.startswith('RESULT')], sum(l.startswith('SKIP') for l in L)


rows, nskip = raw('flif-v1-mitbih-res')
if nskip:
    stop(f'{nskip} SKIP line(s)')
if len({r['sha'] for r in rows}) != 1:
    stop(f'mixed training commits {sorted({r["sha"] for r in rows})}')
sp = os.path.join(a.root, 'flif-v1-mitbih-res', 'selected_lr.json')
if not os.path.exists(sp):
    stop('selected_lr.json missing')
SEL = json.load(open(sp))
want_val = {(RES[k], lr, s) for k in RES for lr in LRS for s in SEEDS}
want_fin = {(RES[k], SEL['selected'][k], s) for k in RES for s in SEEDS}
got_val = [(r['cond'], r['lr'], r['seed']) for r in rows if r.get('val_fold') == '0']
got_fin = [(r['cond'], r['lr'], r['seed']) for r in rows if r.get('val_fold') == '-1']
if len(rows) != 30 or len(set(got_val)) != 24 or set(got_val) != want_val or len(set(got_fin)) != 6 or set(got_fin) != want_fin:
    stop(f'rows: {len(rows)} (want 30), validation {len(set(got_val))} / final {len(set(got_fin))} or not the expected (cond, lr, seed)')
ep = os.path.join(a.root, 'flif-v1-mitbih-res', 'energy_recount.csv')
if not os.path.exists(ep):
    stop('energy_recount.csv missing')
en = {(e['tag'], e['seed']): e for e in csv.DictReader(open(ep))}
for r in rows:
    e = en.get((r['tag'], r['seed']))
    if e is None or abs(fin(e, 'f1_recomputed') - fin(r, 'f1')) > 1e-6:
        stop(f'energy recount missing or recomputed macro-F1 != logged ({r["tag"]} seed {r["seed"]})')


def arm(rr, label):
    d = {}
    for r in rr:
        if r['seed'] in d:
            stop(f'duplicate {label} seed {r["seed"]}')
        d[r['seed']] = {k: fin(r, k) for k in KEYS}
    if sorted(d) != sorted(SEEDS):
        stop(f'{label}: seeds {sorted(d)}')
    return d


def ref(suite, tag):
    rr, _ = raw(suite)
    return arm([r for r in rr if r.get('tag') == tag], f'{suite} {tag}')


A = {'IAND (new)': arm([r for r in rows if r['cond'] == RES['IAND'] and r.get('val_fold') == '-1'], 'IAND final'),
     'MS (new)': arm([r for r in rows if r['cond'] == RES['MS'] and r.get('val_fold') == '-1'], 'MS final'),
     'ADD base': ref('flif-v1-mitbih-spkmux-tune', 'spkmx_lif_muxlin_add_lr0.02_L2'),
     'SF-de PE on': ref('flif-v1-mitbih-sfde', 'spk_ssa_lr0.01_L2_nalif'),
     'SF-de PE off': ref('flif-v1-mitbih-sfde-nope', 'spk_ssa_lr0.01_L2_nalif_pe0'),
     'MS lr 0.02 (11-08, report)': ref('flif-v1-mitbih-spkmux-ms', 'spkmx_lif_muxlin_ms_lr0.02_L2')}


def comp(x, y):
    d = [100. * (A[x][s]['f1'] - A[y][s]['f1']) for s in SEEDS]
    m = mean(d)
    v = 'A better' if m >= 0.5 and sum(t > 0 for t in d) >= 2 else ('A worse' if m <= -0.5 and sum(t < 0 for t in d) >= 2 else 'no clear difference')
    er = mean([A[x][s]['syn_energy_uJ'] for s in SEEDS]) / mean([A[y][s]['syn_energy_uJ'] for s in SEEDS])
    return dict(kind=None, A=x, B=y, d_pp=d, mean_pp=m, verdict=v, energy_ratio=er, adoption_bar=(m >= 0.1 and er <= 1.5))


CS = [('R1', 'IAND (new)', 'ADD base'), ('R2', 'MS (new)', 'ADD base'), ('R3', 'IAND (new)', 'SF-de PE on'), ('R4', 'MS (new)', 'SF-de PE on'),
      ('R5', 'IAND (new)', 'MS (new)'), ('R6', 'IAND (new)', 'SF-de PE off'), ('R7', 'MS (new)', 'SF-de PE off')]
comps = []
for k, x, y in CS:
    c = comp(x, y); c['kind'] = k; comps.append(c)
cd = {c['kind']: c for c in comps}
cand = [k for k, rk, sk in (('IAND', 'R1', 'R3'), ('MS', 'R2', 'R4')) if cd[rk]['verdict'] != 'A worse' and cd[sk]['energy_ratio'] <= 1.5]
rec = 'IAND' if 'IAND' in cand else ('MS' if 'MS' in cand else 'none (keep ADD)')
out = dict(sha=rows[0]['sha'], selected=SEL['selected'], val_scores=SEL['scores'], arms={n: {k: mean([v[s][k] for s in SEEDS]) for k in KEYS} for n, v in A.items()},
           comparisons=comps, candidates=cand, recommendation=rec)
os.makedirs(a.out_dir, exist_ok=True)
json.dump(out, open(os.path.join(a.out_dir, 'summary_mitres.json'), 'w'), indent=1)
L = ['# MIT-BIH residual test: IAND and membrane shortcut (MS) vs ADD (LIF SpkMixer muxlin, 2 mixer layers)', '',
     f'Rules: summarize_mitres.py docstring. 30 new runs (24 validation + 6 final), commit `{out["sha"]}`.', '',
     '## lr selection (seed-mean last-epoch validation macro-F1 %, fold 0)', '', '| residual | ' + ' | '.join(LRS) + ' | selected |', '|---|' + '--:|' * len(LRS) + '---|']
for k in RES:
    L.append(f'| {k} | ' + ' | '.join(f'{SEL["scores"][k][lr]:.3f}' for lr in LRS) + f' | {SEL["selected"][k]} |')
L += ['', '## Test (seed means)', '', '| arm | macro-F1 % | accuracy % | F1 N / S / V / F / Q % | energy µJ (syops) |', '|---|--:|--:|---|--:|']
for n, m in out['arms'].items():
    L.append(f'| {n} | {100 * m["f1"]:.2f} | {100 * m["overall_acc"]:.2f} | ' + ' / '.join(f'{100 * m[f"f1_{i}"]:.2f}' for i in range(5)) + f' | {m["syn_energy_uJ"]:.2f} |')
L += ['', '| kind | A | B | d %p (42 / 7 / 2026) | mean %p | verdict | energy A/B | adoption bar (>= +0.1 %p, <= 1.5x) |', '|---|---|---|---|--:|---|--:|---|']
for c in comps:
    L.append(f'| {c["kind"]} | {c["A"]} | {c["B"]} | ' + ' / '.join(f'{t:+.2f}' for t in c['d_pp']) + f' | {c["mean_pp"]:+.2f} | **{c["verdict"]}** | '
             f'{c["energy_ratio"]:.3f} | {"passes" if c["adoption_bar"] else "does not pass"} |')
L += ['', f'Candidates (not worse than ADD, energy <= 1.5x SF-de): {", ".join(cand) if cand else "none"}; recommendation (pre-registered rule): **{rec}**']
open(os.path.join(a.out_dir, 'summary_mitres.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
