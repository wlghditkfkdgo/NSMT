"""Dry run of summarize_t8.py and recheck_t8.py (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG T-test pre-registration). No training. The
new suite is fabricated from the real muxlin IAND T-4 rows (results/dry_t8_*, removed at the end); the references are real.
  D1 muxlin T8 = T4 x 0.98, Pearson-add T8 = muxlin T8 x 1.02 -> M8 vs M4 'better' (R -2 %, 12 lower), P8 vs M8 'worse' (R +2 %);
     energy ratio M8 / M4 = hand value 1.9; 3 comparisons; recheck passes
  D2 muxlin T8 = T4 x 1.005 -> 'no clear difference'
  N1-N6 fail closed (exit 2, no summary): SKIP line, 23 rows, a T-4 row in the suite, nan, mixed commits, energy identity off
usage (from cinc_mixer_v1/): python -B results/check-t8-20261127/dry_t8.py
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.dirname(os.path.dirname(HERE))
PY = sys.executable
fails = 0
REF = {'ETTh1': 'spkmx_lif_muxlin_iand_lr0.00025_L1_e32_mlp4_wd0.1', 'ETTm1': 'spkmx_lif_muxlin_iand_lr0.0005_L2_pt16_e32_mlp0.5'}


def check(label, c, detail=''):
    global fails
    fails += not c
    print(f'{"PASS" if c else "FAIL"} {label}{": " + detail if detail else ""}')


ref = []
for l in open(os.path.join(C, 'results', 'flif-v1-ett-spkmux-retune', 'raw.txt')):
    if l.startswith('RESULT'):
        r = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
        if r['ds'] in REF and r['tag'] == REF[r['ds']] and r['pl'] in ('96', '720') and r['blocks'] == '1':
            ref.append(r)
assert len(ref) == 12


def build(f, g=1.02, sha='dry0000'):
    raw, en = [], ['ds,pl,cond,tag,seed,body_syn_uJ,body_syn_int_uJ,head_MAC,head_AC_corr,rule\n']
    for comb, fac in (('muxlin', f), ('msxdpca', f * g)):
        for r in ref:
            tag = r['tag'].replace('muxlin_iand', f'{comb}_iand') + '_T8'
            syn = 1.9 * float(r['syn_energy_uJ'])
            raw.append(f'RESULT exp=x ds={r["ds"]} pl={r["pl"]} cond=spkmx_lif_{comb}_iand tag={tag} blocks=1 seed={r["seed"]} '
                       f'mse={float(r["mse"]) * fac!r} mae={float(r["mae"]) * fac!r} val_loss_best=0.5 syn_energy_uJ={syn!r} sha={sha} dir=x\n')
            en.append(f'{r["ds"]},{r["pl"]},spkmx_lif_{comb}_iand,{tag},{r["seed"]},{syn - 1.0!r},{syn - 2.0!r},{1.0 / 4.6e-6!r},1000000,binary\n')
    return raw, en


def run(name, raw, en):
    d = os.path.join(C, 'results', name)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, 'raw.txt'), 'w').writelines(raw)
    open(os.path.join(d, 'energy_recount.csv'), 'w').writelines(en)
    r = subprocess.run([PY, '-B', os.path.join(C, 'summarize_t8.py'), '--suite', name], capture_output=True, text=True, cwd=C)
    return r.returncode, d


raw1, en1 = build(0.98)
rc, d = run('dry_t8_D1', raw1, en1)
S = json.load(open(os.path.join(d, 'summary.json'))) if rc == 0 else None
check('D1 exit 0', rc == 0)
if S:
    cm = {(c['A'], c['B']): c for c in S['comparisons']}
    m = cm[('muxlin IAND T8', 'muxlin IAND T4 (ref)')]
    check('D1 M8 vs M4: R -2 %, 12 lower, better', abs(m['mse']['R_pct'] + 2) < 1e-9 and m['mse']['pairs_lower'] == 12 and m['mse']['verdict'] == 'better')
    check('D1 energy ratio M8 / M4 1.9', abs(m['energy_syops'] - 1.9) < 1e-12, f'{m["energy_syops"]:.6f}')
    m = cm[('Pearson add + MUX IAND T8', 'muxlin IAND T8')]
    check('D1 P8 vs M8: R +2 %, worse', abs(m['mse']['R_pct'] - 2) < 1e-9 and m['mse']['verdict'] == 'worse')
    check('D1 3 comparisons', len(S['comparisons']) == 3)
    r = subprocess.run([PY, '-B', os.path.join(HERE, 'recheck_t8.py'), '--suite', 'dry_t8_D1'], capture_output=True, text=True, cwd=C)
    check('D1 independent recheck passes', r.returncode == 0 and 'ALL CHECKS PASSED' in r.stdout, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-300:])
raw2, en2 = build(1.005, g=1.)
rc, d = run('dry_t8_D2', raw2, en2)
S = json.load(open(os.path.join(d, 'summary.json'))) if rc == 0 else None
check('D2 x 1.005 -> M8 vs M4 no clear difference', S is not None and S['comparisons'][0]['mse']['verdict'] == 'no clear difference')
bad = {'N1 SKIP line': (raw1 + ['SKIP(rc=1) x\n'], en1), 'N2 23 rows': (raw1[:-1], en1),
       'N3 a T-4 row': ([raw1[0].replace('_T8 ', ' ', 1)] + raw1[1:], en1),
       'N4 nan': ([raw1[0].replace(' mse=', ' mse=nan x=')] + raw1[1:], en1),
       'N5 mixed commits': ([raw1[0].replace('sha=dry0000', 'sha=dry1111')] + raw1[1:], en1),
       'N6 energy identity off': ([raw1[0].replace('syn_energy_uJ=', 'syn_energy_uJ=99 x=')] + raw1[1:], en1)}
for i, (nm, (rw, ev)) in enumerate(bad.items()):
    rc, d = run(f'dry_t8_N{i + 1}', rw, ev)
    check(f'{nm}: exit 2, no summary', rc == 2 and not os.path.exists(os.path.join(d, 'summary.json')), f'rc {rc}')
for nm in os.listdir(os.path.join(C, 'results')):
    if nm.startswith('dry_t8_'):
        shutil.rmtree(os.path.join(C, 'results', nm))
print('ALL PASS' if fails == 0 else f'{fails} FAIL')
sys.exit(0 if fails == 0 else 1)
