"""AD residual re-analysed with precision / recall / F1 (2026-11-28, exp/f-lif-pop-v3; user "AD 잔차에 대해서: 지금 기준으로 삼고있는 AUC-PR
말고, F1 score로 계산했을 때, 다시 결과를 분석해줘. (논문 작성은 Pre, Recall, F1으로 보고할 예정)"). No training: logged RESULT rows only.
POST HOC: the pre-registered metric of the residual tests was AUC-PR; this reading was asked for after those results.
Metrics in every row (train_ad.py, TSLib protocol: threshold = top anomaly_ratio 1 % of the train + test score distribution):
  pa_p / pa_r / pa_f1  point-adjusted (a whole labelled anomaly segment counts as found if one point in it is flagged; TSLib, NSMT Table 9)
  raw_p / raw_r / raw_f1  per point, no adjustment
Arms (LIF SpkMixer muxlin, one mixer layer per direction, lr 2e-3, whole training file, seeds 42 / 7 / 2026):
  IAND-all  MSL / SMAP flif-v1-ad-iand, PSM flif-v1-ad-energy (IAND)
  G5        MSL / SMAP OR, PSM IAND (flif-v1-ad-energy)
  ADD       the validation-selected residual of 11-19 on all three ((B) M1 tuned, flif-v1-ad-energy)
  SF-de(tuned)  flif-v1-ad-sfde (lr MSL 2e-3, SMAP 2e-3, PSM 5e-4)
Comparisons (the AD rule form of the AUC-PR tests, applied to F1): per seed r = 100 (A - B) / B; R = mean over the datasets of the seed-mean
r; better = R >= +2 % and >= 2/3 of the pairs higher, worse = mirror. Adoption-bar form: mean over the datasets of the seed-mean F1
difference >= 0.001 and syops energy (syn_energy_uJ) A/B <= 1.5.
References (no pairing): NSMT paper Table 9 rows (DNN: ETSformer, PatchTST, Informer, Autoformer, FEDformer; and NSMT), copied from
NSMT/papers/NSMT.pdf (text layer); the training-free random score of 11-03 (posthoc_trivial_baselines.md; F1 only).
usage (from cinc_mixer_v1/): python -B results/ad-f1-20261128/ad_f1.py
"""
import json
import math
import os
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
RS = os.path.join(C, 'results')
DSS, SEEDS = ('MSL', 'SMAP', 'PSM'), ('42', '7', '2026')
MK = ('pa_p', 'pa_r', 'pa_f1', 'raw_p', 'raw_r', 'raw_f1', 'aucpr', 'syn_energy_uJ')
mean = lambda x: sum(x) / len(x)


def stop(m):
    print(f'STOP: {m}')
    sys.exit(2)


def rows(suite):
    L = open(os.path.join(RS, suite, 'raw.txt')).read().splitlines()
    if any(l.startswith('SKIP') for l in L):
        stop(f'SKIP line in {suite}')
    return [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in L if l.startswith('RESULT')]


def pick(rr, ds, tag):
    v = {}
    for r in rr:
        if r['ds'] == ds and r['tag'] == tag and r['holdout'] == '0':
            if r['seed'] in v:
                stop(f'duplicate {ds} {tag} {r["seed"]}')
            if r.get('energy_check') != 'ok':
                stop(f'energy_check {ds} {tag}')
            v[r['seed']] = {k: float(r[k]) for k in MK}
            if not all(math.isfinite(x) for x in v[r['seed']].values()):
                stop(f'non-finite {ds} {tag}')
    if sorted(v) != sorted(SEEDS):
        stop(f'{ds} {tag}: seeds {sorted(v)}')
    return v


NEW, ENE, SFD = rows('flif-v1-ad-iand'), rows('flif-v1-ad-energy'), rows('flif-v1-ad-sfde')
sfl = json.load(open(os.path.join(RS, 'flif-v1-ad-sfde', 'selected_lr.json')))['selected']
sel = json.load(open(os.path.join(RS, 'flif-v1-ad-energy', 'selected_res.json')))['selected']
if sel['res'] != {'MSL': 'add', 'SMAP': 'add', 'PSM': 'add'} or sel['spike'] != {'MSL': 'or', 'SMAP': 'or', 'PSM': 'iand'}:
    stop(f'11-19 selection changed: {sel}')
T = lambda res: f'spkmx_lif_muxlin_{res}_lr0.002_L1'
ARM = {'IAND-all': {'MSL': pick(NEW, 'MSL', T('iand')), 'SMAP': pick(NEW, 'SMAP', T('iand')), 'PSM': pick(ENE, 'PSM', T('iand'))},
       'G5 (MSL / SMAP OR, PSM IAND)': {'MSL': pick(ENE, 'MSL', T('or')), 'SMAP': pick(ENE, 'SMAP', T('or')), 'PSM': pick(ENE, 'PSM', T('iand'))},
       'ADD (11-19 validation choice)': {ds: pick(ENE, ds, T('add')) for ds in DSS},
       'SF-de(tuned)': {ds: pick(SFD, ds, f'spk_ssa_lr{sfl[ds]}_L2_nalif') for ds in DSS}}
# NSMT paper Table 9 (P / R / F1 %, point-adjusted), MSL / SMAP / PSM
PAPER = {'ETSformer': ((85.13, 84.93, 85.03), (92.25, 55.75, 69.50), (90.02, 80.36, 84.91)),
         'PatchTST': ((88.34, 70.96, 78.70), (90.64, 55.46, 68.82), (91.10, 80.94, 85.72)),
         'Informer': ((81.77, 86.48, 84.06), (90.11, 57.13, 69.92), (64.27, 96.33, 77.10)),
         'Autoformer': ((77.27, 80.92, 79.05), (90.40, 58.62, 71.12), (99.08, 88.15, 93.29)),
         'FEDformer': ((77.14, 80.07, 78.57), (90.47, 58.10, 70.76), (97.31, 97.16, 97.23)),
         'NSMT (paper)': ((88.24, 67.54, 76.51), (91.61, 64.57, 75.75), (98.54, 92.56, 95.46))}
RANDOM = {'pa_f1': (84.60, 93.75, 95.89), 'raw_f1': (1.82, 1.86, 1.93)}   # 11-03 posthoc_trivial_baselines.md


def comp(x, y, dss, key):
    rd, flat, dabs, es = [], [], [], []
    for ds in dss:
        A, B = ARM[x][ds], ARM[y][ds]
        r_ = [100. * (A[s][key] - B[s][key]) / B[s][key] for s in SEEDS]
        rd.append(mean(r_)); flat += r_
        dabs.append(mean([A[s][key] for s in SEEDS]) - mean([B[s][key] for s in SEEDS]))
        es.append(mean([A[s]['syn_energy_uJ'] for s in SEEDS]) / mean([B[s]['syn_energy_uJ'] for s in SEEDS]))
    R, need = mean(rd), 2 * len(flat) // 3
    hi, lo = sum(t > 0 for t in flat), sum(t < 0 for t in flat)
    v = 'A better' if R >= 2 and hi >= need else ('A worse' if R <= -2 and lo >= need else 'no clear difference')
    return dict(A=x, B=y, metric=key, datasets=list(dss), R=R, per_ds_r=rd, hi=hi, n=len(flat), verdict=v, d_mean=mean(dabs), per_ds_d=dabs,
                energy=mean(es), adoption_bar=bool(mean(dabs) >= 0.001 and mean(es) <= 1.5))


I, G, AD, SF = ARM
Q = [('F1', I, SF, DSS), ('F2', I, G, ('MSL', 'SMAP')), ('F0', G, SF, DSS), ('F3', AD, SF, DSS), ('F4', I, AD, DSS)]
comps = [dict(kind=k, **comp(x, y, d, key)) for key in ('pa_f1', 'raw_f1', 'aucpr') for k, x, y, d in Q]
rec = {}
for key in ('pa_f1', 'raw_f1'):
    c = {x['kind']: x for x in comps if x['metric'] == key}
    rec[key] = 'IAND for AD' if c['F2']['verdict'] != 'A worse' and c['F1']['adoption_bar'] else 'keep G5'
summ = {n: {ds: {k: mean([v[ds][s][k] for s in SEEDS]) for k in MK} | {k + '_sd': (sum((v[ds][s][k] - mean([v[ds][t][k] for t in SEEDS])) ** 2 for s in SEEDS) / 3) ** .5
                                                                        for k in ('pa_f1', 'raw_f1')} for ds in DSS} for n, v in ARM.items()}
out = dict(arms=summ, comparisons=comps, recommendation_same_rule=rec, paper_table9=PAPER, random_score=RANDOM, post_hoc=True)
json.dump(out, open(os.path.join(HERE, 'summary_ad_f1.json'), 'w'), indent=1)
P = lambda v: f'{100 * v:.2f}'
L = ['# AD residual with precision / recall / F1 (post hoc; no training)', '', 'Rules and sources: ad_f1.py docstring. Seed means (%); ± = seed std of F1.', '']
for tag, keys in (('point-adjusted (TSLib / NSMT Table 9 convention)', ('pa_p', 'pa_r', 'pa_f1')), ('raw (per point, no adjustment)', ('raw_p', 'raw_r', 'raw_f1'))):
    L += [f'## {tag}', '', '| arm | MSL P / R / F1 | SMAP P / R / F1 | PSM P / R / F1 | avg F1 |', '|---|---|---|---|--:|']
    for n in ARM:
        cells = [f"{P(summ[n][ds][keys[0]])} / {P(summ[n][ds][keys[1]])} / **{P(summ[n][ds][keys[2]])}** ± {P(summ[n][ds][keys[2] + '_sd'])}" for ds in DSS]
        L.append(f'| {n} | ' + ' | '.join(cells) + f' | {mean([100 * summ[n][ds][keys[2]] for ds in DSS]):.2f} |')
    if keys[0] == 'pa_p':
        for n, t in PAPER.items():
            L.append(f'| {n} (NSMT Table 9) | ' + ' | '.join(f'{p:.2f} / {r:.2f} / {f:.2f}' for p, r, f in t) + f' | {mean([x[2] for x in t]):.2f} |')
    rk = 'pa_f1' if keys[0] == 'pa_p' else 'raw_f1'
    L.append(f'| training-free random score (11-03) | — / — / {RANDOM[rk][0]:.2f} | — / — / {RANDOM[rk][1]:.2f} | — / — / {RANDOM[rk][2]:.2f} | {mean(RANDOM[rk]):.2f} |')
    L.append('')
L += ['## Comparisons (AD rule form: R >= +2 % and >= 2/3 pairs; adoption-bar form: mean F1 difference >= 0.001, syops energy <= 1.5x)', '',
      '| metric | kind | A vs B | datasets | R % | per-dataset r % | pairs higher | verdict | mean Δ | energy A/B | bar |', '|---|---|---|---|--:|---|--:|---|--:|--:|---|']
for c in comps:
    L.append(f"| {c['metric']} | {c['kind']} | {c['A']} vs {c['B']} | {' / '.join(c['datasets'])} | {c['R']:+.2f} | " + ' / '.join(f'{t:+.2f}' for t in c['per_ds_r'])
             + f" | {c['hi']}/{c['n']} | {c['verdict']} | {c['d_mean']:+.4f} | {c['energy']:.3f} | {'pass' if c['adoption_bar'] else 'fail'} |")
L += ['', f"Same recommendation rule as the AUC-PR test (F2 not worse and F1 passes the bar): point-adjusted F1 -> **{rec['pa_f1']}**, raw F1 -> **{rec['raw_f1']}**"]
open(os.path.join(HERE, 'summary_ad_f1.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
