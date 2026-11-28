"""Final report of the AD tuning (2026-11-29, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; adtune.py stages norm -> ... -> res).
Tuned = per dataset the configuration after all seven selections (test AUC-PR selected: OPTIMISTIC by construction).
Reported (seed means, seeds 42 / 7 / 2026): point-adjusted and raw precision / recall / F1, AUC-PR, AUC-ROC, syops energy, epochs run, for
  tuned SpkMixer, the start (11-19 G5), SF-de (11-18 lr-tuned only, internal), the NSMT paper Table 9 DNN rows and NSMT row (P / R / F1),
  and the training-free scores of 11-03 (random: F1; window-mean and train-deviation: AUC-PR).
Comparisons (AD rule form): per seed r = 100 (A - B) / B, R = mean over MSL / SMAP / PSM of the seed-mean r; better = R >= +2 % and >= 6 of
9 pairs higher, worse = mirror; mean difference and syops energy A/B; for AUC-PR, point-adjusted F1, raw F1:
  tuned vs start (the start is one of the candidates, so the tuned value is >= the start on AUC-PR by construction), tuned vs SF-de.
Fails closed (exit 2): any selection file missing, a missing / duplicate / non-finite row, a SKIP line in the tuning suite.
usage (from cinc_mixer_v1/): python -B results/check-adtune2-20261129/summarize_adtune2.py   (ADT_ROOT as adtune.py)
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, C)
import adtune as T                                                       # noqa: E402

OUT = os.environ.get('ADT_OUT') or HERE
DSS = ('MSL', 'SMAP', 'PSM')
SEEDS = T.SEEDS
K = ('pa_p', 'pa_r', 'pa_f1', 'raw_p', 'raw_r', 'raw_f1', 'aucpr', 'aucroc', 'syn_energy_uJ', 'epochs_run', 'best_epoch', 'time_s')
mean = lambda x: sum(x) / len(x)
PAPER = {'ETSformer': ((85.13, 84.93, 85.03), (92.25, 55.75, 69.50), (90.02, 80.36, 84.91)),
         'PatchTST': ((88.34, 70.96, 78.70), (90.64, 55.46, 68.82), (91.10, 80.94, 85.72)),
         'Informer': ((81.77, 86.48, 84.06), (90.11, 57.13, 69.92), (64.27, 96.33, 77.10)),
         'Autoformer': ((77.27, 80.92, 79.05), (90.40, 58.62, 71.12), (99.08, 88.15, 93.29)),
         'FEDformer': ((77.14, 80.07, 78.57), (90.47, 58.10, 70.76), (97.31, 97.16, 97.23)),
         'NSMT (paper)': ((88.24, 67.54, 76.51), (91.61, 64.57, 75.75), (98.54, 92.56, 95.46))}
FREE = {'random': dict(pa_f1=(84.60, 93.75, 95.89), raw_f1=(1.82, 1.86, 1.93), aucpr=(10.53, 12.79, 27.75)),
        'window-mean distance': dict(aucpr=(14.96, 10.84, 38.12), pa_f1=(81.79, 67.39, 92.42), raw_f1=(4.47, 1.18, 4.19)),
        'train-deviation distance': dict(aucpr=(14.02, 10.56, 47.26), pa_f1=(82.06, 67.07, 88.32), raw_f1=(6.43, 1.23, 3.30))}


def stop(m):
    print(f'STOP: {m}')
    sys.exit(2)


D = T.index()
final = T.current('res')                                                 # after norm .. lr
sel_res = json.load(open(os.path.join(T.ROOT, T.SUITE, 'selected_res.json')))['selected']
for ds in DSS:
    final[ds] = T.apply(final[ds], ('res',), tuple(sel_res[ds]))
path = {}
for name, keys, _ in T.STAGES:
    path[name] = json.load(open(os.path.join(T.ROOT, T.SUITE, f'selected_{name}.json')))['selected']


def rows(cfgs):
    out = {}
    for ds in DSS:
        rr = T.three(D, ds, cfgs[ds])
        if rr is None:
            stop(f'missing rows {ds} {T.tag(cfgs[ds])}')
        v = {}
        for r in rr:
            v[r['seed']] = {k: float(r[k]) for k in K}
            if not all(math.isfinite(x) for x in v[r['seed']].values()) or r.get('energy_check') != 'ok':
                stop(f'non-finite / energy_check {ds} {r["tag"]} seed {r["seed"]}')
        out[ds] = v
    return out


def sfde():
    sfl = json.load(open(os.path.join(T.ROOT, 'flif-v1-ad-sfde', 'selected_lr.json')))['selected']
    L = [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in open(os.path.join(T.ROOT, 'flif-v1-ad-sfde', 'raw.txt')) if l.startswith('RESULT')]
    out = {}
    for ds in DSS:
        v = {r['seed']: {k: float(r[k]) for k in K} for r in L if r['ds'] == ds and r['holdout'] == '0' and r['tag'] == f'spk_ssa_lr{sfl[ds]}_L2_nalif'}
        if sorted(v) != sorted(SEEDS):
            stop(f'SF-de {ds}')
        out[ds] = v
    return out


ARM = {'SpkMixer tuned (test-selected)': rows(final), 'SpkMixer start (11-19 G5)': rows(T.BASE), 'SF-de (11-18, lr only)': sfde()}


def comp(x, y, key):
    rd, flat, dd, es = [], [], [], []
    for ds in DSS:
        A, B = ARM[x][ds], ARM[y][ds]
        r_ = [100. * (A[s][key] - B[s][key]) / B[s][key] for s in SEEDS]
        rd.append(mean(r_)); flat += r_
        dd.append(mean([A[s][key] for s in SEEDS]) - mean([B[s][key] for s in SEEDS]))
        es.append(mean([A[s]['syn_energy_uJ'] for s in SEEDS]) / mean([B[s]['syn_energy_uJ'] for s in SEEDS]))
    R = mean(rd); hi = sum(t > 0 for t in flat); lo = sum(t < 0 for t in flat)
    v = 'A better' if R >= 2 and hi >= 6 else ('A worse' if R <= -2 and lo >= 6 else 'no clear difference')
    return dict(A=x, B=y, metric=key, R=R, per_ds_r=rd, hi=hi, verdict=v, d_mean=mean(dd), per_ds_d=dd, energy=mean(es))


TU, ST, SF = ARM
comps = [comp(TU, b, k) for k in ('aucpr', 'pa_f1', 'raw_f1') for b in (ST, SF)]
summ = {n: {ds: {k: mean([v[ds][s][k] for s in SEEDS]) for k in K} for ds in DSS} for n, v in ARM.items()}
out = dict(final_config={ds: final[ds] | {'tag': T.tag(final[ds])} for ds in DSS}, path=path, arms=summ, comparisons=comps, paper=PAPER, free=FREE)
os.makedirs(OUT, exist_ok=True)
json.dump(out, open(os.path.join(OUT, 'summary_adtune2.json'), 'w'), indent=1)
P = lambda v: f'{100 * v:.2f}'
L = ['# AD tuning of the LIF SpkMixer: final report (test AUC-PR selected per dataset: optimistic)', '', 'Rules: summarize_adtune2.py and adtune.py docstrings.', '',
     '## Selected path (per dataset)', '', '| stage | ' + ' | '.join(DSS) + ' |', '|---|' + '---|' * len(DSS)]
for name, keys, _ in T.STAGES:
    L.append(f'| {name} ({", ".join(keys)}) | ' + ' | '.join(str(tuple(path[name][ds])) for ds in DSS) + ' |')
L += ['', 'Final tags: ' + '; '.join(f'{ds} `{T.tag(final[ds])}`' for ds in DSS), '']
for title, keys in (('point-adjusted P / R / F1 (TSLib / NSMT Table 9 convention)', ('pa_p', 'pa_r', 'pa_f1')), ('raw P / R / F1', ('raw_p', 'raw_r', 'raw_f1'))):
    L += [f'## {title}', '', '| row | ' + ' | '.join(DSS) + ' | avg F1 |', '|---|' + '---|' * len(DSS) + '--:|']
    for n in ARM:
        L.append(f'| {n} | ' + ' | '.join(f"{P(summ[n][ds][keys[0]])} / {P(summ[n][ds][keys[1]])} / **{P(summ[n][ds][keys[2]])}**" for ds in DSS)
                 + f" | {mean([100 * summ[n][ds][keys[2]] for ds in DSS]):.2f} |")
    if keys[0] == 'pa_p':
        for n, t in PAPER.items():
            L.append(f'| {n} (NSMT Table 9) | ' + ' | '.join(f'{p:.2f} / {r:.2f} / {f:.2f}' for p, r, f in t) + f' | {mean([x[2] for x in t]):.2f} |')
    for n, d in FREE.items():
        k = keys[2]
        L.append(f'| training-free {n} (11-03) | ' + ' | '.join(f'— / — / {x:.2f}' for x in d[k]) + f' | {mean(d[k]):.2f} |')
    L.append('')
L += ['## AUC-PR / AUC-ROC / energy (seed means)', '', '| row | ' + ' | '.join(f'{ds} AUC-PR / AUC-ROC / µJ / epochs run' for ds in DSS) + ' |', '|---|' + '---|' * len(DSS)]
for n in ARM:
    L.append(f'| {n} | ' + ' | '.join(f"{P(summ[n][ds]['aucpr'])} / {P(summ[n][ds]['aucroc'])} / {summ[n][ds]['syn_energy_uJ']:.2f} / {summ[n][ds]['epochs_run']:.1f}" for ds in DSS) + ' |')
for n, d in FREE.items():
    L.append(f'| training-free {n} (11-03) | ' + ' | '.join(f'{x:.2f} / — / 0 / —' for x in d['aucpr']) + ' |')
L += ['', '## Comparisons (AD rule form)', '', '| metric | A vs B | R % | per-dataset r % | pairs higher | verdict | mean Δ | energy A/B |', '|---|---|--:|---|--:|---|--:|--:|']
for c in comps:
    L.append(f"| {c['metric']} | {c['A']} vs {c['B']} | {c['R']:+.2f} | " + ' / '.join(f'{t:+.2f}' for t in c['per_ds_r']) + f" | {c['hi']}/9 | {c['verdict']} | {c['d_mean']:+.4f} | {c['energy']:.3f} |")
open(os.path.join(OUT, 'summary_adtune2.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
