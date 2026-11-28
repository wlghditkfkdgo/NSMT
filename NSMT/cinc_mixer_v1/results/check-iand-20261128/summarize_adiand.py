"""AD residual unified to IAND (2026-11-28, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; user "1번 AD IAND 통일 진행해줘" after the
21:28 decision "ETT도 IAND를 선택하고 있다면 MIT-BIH도 IAND를 선택하는 게 맞지"; rules fixed before training).
  IAND-all  LIF SpkMixer muxlin, one mixer layer per direction, lr 2e-3, residual IAND on all three datasets:
            MSL / SMAP = the 6 new runs (suite flif-v1-ad-iand, tag spkmx_lif_muxlin_iand_lr0.002_L1, whole training file),
            PSM = the 11-19 G5 runs (flif-v1-ad-energy, same tag; IAND was already the validation choice there)
  G5        the 11-19 arm: MSL / SMAP OR, PSM IAND (flif-v1-ad-energy, M 1, lr 2e-3)
  SF-de     tuned (flif-v1-ad-sfde, selected_lr.json: MSL 2e-3, SMAP 2e-3, PSM 5e-4; tag spk_ssa_lr<lr>_L2_nalif)
Test AUC-PR of the best-validation weights, seeds 42 / 7 / 2026, paired by seed.
  r = 100 (A - B) / B per seed; R = mean over the datasets of the seed-mean r.
  J1  IAND-all vs SF-de over MSL / SMAP / PSM: better = R >= +2 % and >= 6 of 9 pairs higher, worse = mirror (11-03 / 11-19 rule)
  J2  IAND-all vs G5 over MSL / SMAP (PSM is the same runs): better = R >= +2 % and >= 4 of 6 pairs higher, worse = mirror
  J0  (report) G5 vs SF-de, re-read under the 11-24 energy rule
  Adoption bar (user 11-27, AD form fixed here): mean over the datasets of (seed-mean AUC-PR_A - seed-mean AUC-PR_B) >= 0.001 and
  energy A/B <= 1.5. Energy (user 11-24) = syn_energy_uJ (syops count only, no neuron term); A/B = mean over the datasets of the ratio
  of seed means. total_fft_uJ (syops + f-LIF-paper neuron term, the 11-19 judging column) is reported next to it.
  Recommendation rule: "IAND for AD" if J2 is not "worse" and J1 passes the adoption bar; otherwise "keep G5" (the decision is the user's).
  Reproduction check (report, stated before training): PSM IAND seed 42 rerun at the current commit (suite check-adiand-repro) must give
  the 11-19 row's val_mse_best / aucpr / syn_energy_uJ exactly; if not, the mixed IAND-all arm (new MSL / SMAP + 11-19 PSM) is flagged.
Fails closed (exit 2, nothing written): a SKIP line or a row of another tag / holdout in the new suite, not 6 rows (MSL / SMAP x 3 seeds),
mixed commits, a missing / duplicate reference row, a non-finite or non-positive value, energy_check != ok.
usage (from cinc_mixer_v1/): python -B results/check-iand-20261128/summarize_adiand.py [--new <suite dir>] [--repro <suite dir>] [--out_dir ...]
"""
import argparse
import json
import math
import os
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
RS = os.path.join(C, 'results')
ap = argparse.ArgumentParser()
ap.add_argument('--new', default=os.path.join(RS, 'flif-v1-ad-iand'))
ap.add_argument('--repro', default=os.path.join(RS, 'check-adiand-repro'))
ap.add_argument('--out_dir', default=HERE)
a = ap.parse_args()
DSS, SEEDS = ('MSL', 'SMAP', 'PSM'), ('42', '7', '2026')
TAG_I = 'spkmx_lif_muxlin_iand_lr0.002_L1'
TAG_O = 'spkmx_lif_muxlin_or_lr0.002_L1'
KEYS = ('aucpr', 'aucroc', 'pa_f1', 'raw_f1', 'syn_energy_uJ', 'total_fft_uJ', 'val_mse_best', 'best_epoch', 'time_s')
mean = lambda x: sum(x) / len(x)


def stop(m):
    print(f'STOP: {m}')
    sys.exit(2)


def rows(path):
    p = os.path.join(path, 'raw.txt')
    if not os.path.exists(p):
        stop(f'{p} missing')
    L = open(p).read().splitlines()
    return L, [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in L if l.startswith('RESULT')]


def num(r, k):
    try:
        v = float(r[k])
    except (KeyError, ValueError, TypeError):
        stop(f'{k} missing ({r.get("ds")} {r.get("tag")} seed {r.get("seed")})')
    if not math.isfinite(v) or (k not in ('best_epoch',) and v <= 0):
        stop(f'{k} = {v} not finite / not positive ({r.get("ds")} {r.get("tag")} seed {r.get("seed")})')
    return v


def pick(rr, ds, tag, where):
    out = {}
    for r in rr:
        if r.get('ds') == ds and r.get('tag') == tag and r.get('holdout') == '0':
            if r['seed'] in out:
                stop(f'duplicate {where} {ds} {tag} seed {r["seed"]}')
            if r.get('energy_check') != 'ok':
                stop(f'energy_check != ok: {where} {ds} {tag} seed {r["seed"]}')
            out[r['seed']] = r
    if sorted(out) != sorted(SEEDS):
        stop(f'{where} {ds} {tag}: seeds {sorted(out)}')
    return {s: {k: num(out[s], k) for k in KEYS} for s in SEEDS}


# new suite: exactly MSL / SMAP x 3 seeds of TAG_I, one commit, no SKIP
L, NR = rows(a.new)
if any(l.startswith('SKIP') for l in L):
    stop('SKIP line in the new suite')
if any(r.get('tag') != TAG_I or r.get('holdout') != '0' or r.get('ds') not in ('MSL', 'SMAP') for r in NR) or len(NR) != 6:
    stop(f'new suite: {len(NR)} rows or unexpected rows (want 6 of {TAG_I}, MSL / SMAP, holdout 0)')
if len({r['sha'] for r in NR}) != 1:
    stop(f'new suite: mixed commits {sorted({r["sha"] for r in NR})}')
sha = NR[0]['sha']
_, ER = rows(os.path.join(RS, 'flif-v1-ad-energy'))
_, SR = rows(os.path.join(RS, 'flif-v1-ad-sfde'))
sfsel = json.load(open(os.path.join(RS, 'flif-v1-ad-sfde', 'selected_lr.json')))['selected']
if sfsel != {'MSL': '0.002', 'SMAP': '0.002', 'PSM': '0.0005'}:
    stop(f'SF-de selection changed: {sfsel}')
ARM = {'IAND-all (new MSL / SMAP)': {ds: pick(NR, ds, TAG_I, 'new') for ds in ('MSL', 'SMAP')},
       'G5 (MSL / SMAP OR, PSM IAND)': {ds: pick(ER, ds, TAG_O if ds != 'PSM' else TAG_I, 'ad-energy') for ds in DSS},
       'SF-de(tuned)': {ds: pick(SR, ds, f'spk_ssa_lr{sfsel[ds]}_L2_nalif', 'ad-sfde') for ds in DSS}}
ARM['IAND-all (new MSL / SMAP)']['PSM'] = ARM['G5 (MSL / SMAP OR, PSM IAND)']['PSM']
IA, G5, SF = ARM


def comp(x, y, dss, need):
    rd, hi, lo, dabs, es, et, cells = [], 0, 0, [], [], [], []
    for ds in dss:
        A, B = ARM[x][ds], ARM[y][ds]
        r_ = [100. * (A[s]['aucpr'] - B[s]['aucpr']) / B[s]['aucpr'] for s in SEEDS]
        rd.append(mean(r_)); hi += sum(t > 0 for t in r_); lo += sum(t < 0 for t in r_)
        dabs.append(mean([A[s]['aucpr'] for s in SEEDS]) - mean([B[s]['aucpr'] for s in SEEDS]))
        es.append(mean([A[s]['syn_energy_uJ'] for s in SEEDS]) / mean([B[s]['syn_energy_uJ'] for s in SEEDS]))
        et.append(mean([A[s]['total_fft_uJ'] for s in SEEDS]) / mean([B[s]['total_fft_uJ'] for s in SEEDS]))
        cells.append(dict(ds=ds, r=r_, mean_r=mean(r_), d_aucpr=dabs[-1], energy_syn=es[-1], energy_total=et[-1]))
    R = mean(rd)
    v = 'A better' if R >= 2 and hi >= need else ('A worse' if R <= -2 and lo >= need else 'no clear difference')
    E, D = mean(es), mean(dabs)
    return dict(A=x, B=y, datasets=list(dss), R=R, hi=hi, n=3 * len(dss), verdict=v, d_aucpr=D, energy_syn=E, energy_total=mean(et),
                adoption_bar=bool(D >= 0.001 and E <= 1.5), cells=cells)


comps = [dict(kind='J1', **comp(IA, SF, DSS, 6)), dict(kind='J2', **comp(IA, G5, ('MSL', 'SMAP'), 4)),
         dict(kind='J0 (report)', **comp(G5, SF, DSS, 6))]
J1, J2 = comps[0], comps[1]
rec = 'IAND for AD' if J2['verdict'] != 'A worse' and J1['adoption_bar'] else 'keep G5'
# reproduction check (report)
RL, RR = rows(a.repro)
rep = dict(ok=False, note='')
if any(l.startswith('SKIP') for l in RL) or len(RR) != 1 or RR[0].get('tag') != TAG_I or RR[0].get('ds') != 'PSM' or RR[0].get('seed') != '42':
    rep['note'] = f'reproduction suite: SKIP or unexpected rows ({len(RR)})'
else:
    old = [r for r in ER if r.get('ds') == 'PSM' and r.get('tag') == TAG_I and r.get('holdout') == '0' and r.get('seed') == '42']
    vals = {k: (RR[0].get(k), old[0].get(k)) for k in ('val_mse_best', 'aucpr', 'syn_energy_uJ', 'best_epoch', 'epochs_run')}
    rep = dict(ok=all(x == y for x, y in vals.values()), values=vals, sha=RR[0].get('sha'), note='')
out = dict(sha=sha, arms={n: {ds: {k: mean([v[ds][s][k] for s in SEEDS]) for k in KEYS} for ds in DSS} for n, v in ARM.items()},
           comparisons=comps, recommendation=rec, reproduction=rep, sfde_lr=sfsel)
os.makedirs(a.out_dir, exist_ok=True)
json.dump(out, open(os.path.join(a.out_dir, 'summary_adiand.json'), 'w'), indent=1)
M = ['# AD residual unified to IAND (one mixer layer per direction, lr 2e-3)', '',
     f'Rules: summarize_adiand.py docstring. 6 new runs (MSL / SMAP), commit `{sha}`; PSM IAND = the 11-19 G5 runs.', '',
     '| comparison | datasets | R (AUC-PR, %) | pairs higher | verdict | mean ΔAUC-PR | energy A/B syops (syn) | energy A/B syops + neuron | adoption bar |',
     '|---|---|--:|--:|---|--:|--:|--:|---|']
for c in comps:
    M.append(f"| {c['kind']} {c['A']} vs {c['B']} | {' / '.join(c['datasets'])} | {c['R']:+.3f} | {c['hi']}/{c['n']} | **{c['verdict']}** | "
             f"{c['d_aucpr']:+.4f} | {c['energy_syn']:.4f} | {c['energy_total']:.4f} | {'pass' if c['adoption_bar'] else 'fail'} |")
M += ['', '| comparison | per-dataset seed-mean r % (seeds 42 / 7 / 2026) ; ΔAUC-PR ; energy A/B syn |', '|---|---|']
for c in comps:
    M.append(f"| {c['kind']} | " + ' ; '.join(f"{x['ds']} {x['mean_r']:+.2f} ({', '.join(f'{t:+.2f}' for t in x['r'])}), {x['d_aucpr']:+.4f}, "
                                           f"{x['energy_syn']:.3f}" for x in c['cells']) + ' |')
M += ['', f'Recommendation (pre-registered rule): **{rec}**', '']
for ds in DSS:
    M += [f'## {ds} (seed mean)', '', '| arm | AUC-PR % | AUC-ROC % | PA-F1 % | raw F1 % | energy µJ syn | energy µJ syn + neuron | best epoch | time s |',
          '|---|--:|--:|--:|--:|--:|--:|--:|--:|']
    for n in ARM:
        m = out['arms'][n][ds]
        M.append(f"| {n} | {100 * m['aucpr']:.2f} | {100 * m['aucroc']:.2f} | {100 * m['pa_f1']:.2f} | {100 * m['raw_f1']:.2f} | "
                 f"{m['syn_energy_uJ']:.2f} | {m['total_fft_uJ']:.2f} | {m['best_epoch']:.1f} | {m['time_s']:.0f} |")
    M.append('')
M.append('Reproduction (report): ' + (('PSM IAND seed 42 at commit `%s` = the 11-19 row: ' % rep['sha']) +
                                     ', '.join(f'{k} {x} vs {y}' for k, (x, y) in rep['values'].items()) + (' -> **same**' if rep['ok'] else ' -> **NOT the same**')
                                     if 'values' in rep else rep['note']))
open(os.path.join(a.out_dir, 'summary_adiand.md'), 'w').write('\n'.join(M) + '\n')
print('\n'.join(M))
