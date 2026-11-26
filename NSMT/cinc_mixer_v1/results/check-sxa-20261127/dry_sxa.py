"""Dry run of summarize_sxa.py and recheck_sxa.py on fabricated rows (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG stage-1
pre-registration). No training. Temporary suites results/dry_sxa_* are removed at the end; the IAND reference rows are the real ones.
  D1 arm 1 = base; sxdpc shared x 0.97 on all 12 pairs -> 'better' (R -3 %); sxd indep. x 1.02 -> 'worse'; the other arms = base
     -> 'no clear difference'; validation: sxppc indep. x 0.98 (others x 1) -> selected sxppc indep.; energy ratios = hand values;
     the independent recheck passes
  D2 two Pearson arms with the same validation score -> the one with the lower energy is selected
  D3 R about -1.46 % but only 7 of 12 pairs lower -> 'no clear difference'
  N1-N8 fail closed (exit 2, no summary / selection written): SKIP line, missing run, duplicate run, nan, mixed commits, missing
     energy row, energy identity off, blocks 2
usage (from cinc_mixer_v1/): python -B results/check-sxa-20261127/dry_sxa.py
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.dirname(os.path.dirname(HERE))
PY = sys.executable
CELLS = [(d, p) for d in ('ETTh1', 'ETTm1') for p in ('96', '720')]
SEEDS = ('42', '7', '2026')
ARMS = [(c, s) for c in ('muxlin', 'sxd', 'sxdpc', 'sxp', 'sxppc') for s in (False, True)]
fails = 0


def check(label, c, detail=''):
    global fails
    fails += not c
    print(f'{"PASS" if c else "FAIL"} {label}{": " + detail if detail else ""}')


def build(fac, vfac=lambda arm, k: 1., body=lambda arm: 5. + ARMS.index(arm), sha='dry0000'):
    raw, en = [], ['ds,pl,cond,tag,seed,body_syn_uJ,body_syn_int_uJ,head_MAC,head_AC_corr,rule\n']
    k = 0
    for arm in ARMS:
        cond = f'spkmx_lif_{arm[0]}_add'
        for d, p in CELLS:
            tag = f'{cond}_dry{d}' + ('_ws' if arm[1] else '')
            for s in SEEDS:
                base = 0.3 + 0.01 * CELLS.index((d, p)) + 0.001 * SEEDS.index(s)
                f = fac(arm, k)
                b = body(arm)
                syn = b + 1e6 * 4.6e-6
                raw.append(f'RESULT exp=x ds={d} pl={p} cond={cond} tag={tag} blocks=1 seed={s} mse={base * f!r} mae={base * f!r} '
                           f'val_loss_best={0.5 * vfac(arm, k)!r} syn_energy_uJ={syn!r} sha={sha} dir=x\n')
                en.append(f'{d},{p},{cond},{tag},{s},{b!r},{b - 1!r},1000000,2000000,binary\n')
                k += 1
    return raw, en


def run(name, raw, en):
    d = os.path.join(C, 'results', name)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, 'raw.txt'), 'w').writelines(raw)
    open(os.path.join(d, 'energy_recount.csv'), 'w').writelines(en)
    r = subprocess.run([PY, '-B', os.path.join(C, 'summarize_sxa.py'), '--suite', name], capture_output=True, text=True, cwd=C)
    return r.returncode, d


def find(S, kind, A, B):
    return [c for c in S['comparisons'] if c['kind'] == kind and c['A'] == A and c['B'] == B][0]


f1 = lambda arm, k: 0.97 if arm == ('sxdpc', True) else (1.02 if arm == ('sxd', False) else 1.)
v1 = lambda arm, k: 0.98 if arm == ('sxppc', False) else 1.
raw1, en1 = build(f1, v1)
rc, d = run('dry_sxa_D1', raw1, en1)
S = json.load(open(os.path.join(d, 'summary.json'))) if rc == 0 else None
check('D1 summarizer exit 0', rc == 0)
if S:
    c = find(S, 'main', 'sxdpc shared', 'muxlin indep.')
    check('D1 sxdpc shared vs arm 1: R -3 %, 12 lower, better', abs(c['mse']['R_pct'] + 3) < 1e-9 and c['mse']['pairs_lower'] == 12 and c['mse']['verdict'] == 'better')
    c = find(S, 'main', 'sxd indep.', 'muxlin indep.')
    check('D1 sxd indep. vs arm 1: R +2 %, 12 higher, worse', abs(c['mse']['R_pct'] - 2) < 1e-9 and c['mse']['pairs_higher'] == 12 and c['mse']['verdict'] == 'worse')
    want = (5 + ARMS.index(('sxd', False)) + 4.6) / (5 + 0 + 4.6)
    check('D1 energy ratio sxd indep. / arm 1 = hand value', abs(c['energy_syops'] - want) < 1e-12, f'{c["energy_syops"]:.6f} vs {want:.6f}')
    wi = (4 + ARMS.index(('sxd', False)) + 1.8) / (4 + 1.8)
    check('D1 integer-rule ratio = hand value', abs(c['energy_int'] - wi) < 1e-12)
    c = find(S, 'sharing', 'sxp shared', 'sxp indep.')
    check('D1 equal arms: no clear difference (0 lower / 0 higher)', c['mse']['verdict'] == 'no clear difference' and c['mse']['pairs_lower'] == 0)
    check('D1 ADD comparison against the real IAND reference exists (12 pairs)', find(S, 'ADD', 'muxlin indep.', 'IAND ref')['mse']['pairs_lower']
          + find(S, 'ADD', 'muxlin indep.', 'IAND ref')['mse']['pairs_higher'] <= 12)
    check('D1 selection = sxppc indep. (validation x 0.98)', S['selection']['selected'] == 'sxppc indep.'
          and json.load(open(os.path.join(d, 'selected_s1.json'))) ['comb'] == 'sxppc')
    check('D1 22 comparisons (8 main, 4 mask, 4 projection, 5 sharing, 1 ADD)', len(S['comparisons']) == 22)
    r = subprocess.run([PY, '-B', os.path.join(HERE, 'recheck_sxa.py'), '--suite', 'dry_sxa_D1'], capture_output=True, text=True, cwd=C)
    check('D1 independent recheck passes', r.returncode == 0 and 'ALL CHECKS PASSED' in r.stdout, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-300:])
raw2, en2 = build(lambda arm, k: 1., lambda arm, k: 0.97 if arm in (('sxdpc', False), ('sxppc', True)) else 1.,
                  body=lambda arm: 9. if arm == ('sxdpc', False) else 5. + ARMS.index(arm) * 0.1)
rc, d = run('dry_sxa_D2', raw2, en2)
S = json.load(open(os.path.join(d, 'summary.json'))) if rc == 0 else None
check('D2 tie in validation -> lower energy (sxppc shared) selected', S is not None and S['selection']['selected'] == 'sxppc shared')
raw3, en3 = build(lambda arm, k: (0.975 if (k % 12) < 7 else 1.) if arm == ('sxp', False) else 1.)
rc, d = run('dry_sxa_D3', raw3, en3)
S = json.load(open(os.path.join(d, 'summary.json'))) if rc == 0 else None
if S:
    c = find(S, 'main', 'sxp indep.', 'muxlin indep.')
    check('D3 R <= -1 % with only 7 lower -> no clear difference', c['mse']['R_pct'] <= -1 and c['mse']['pairs_lower'] == 7
          and c['mse']['verdict'] == 'no clear difference', f'R {c["mse"]["R_pct"]:.3f}, {c["mse"]["pairs_lower"]} lower')
else:
    check('D3 summarizer exit 0', False)
bad = {'N1 SKIP line': (raw1 + ['SKIP(rc=1, no result) ds=ETTh1\n'], en1),
       'N2 missing run': (raw1[:-1], en1),
       'N3 duplicate run': (raw1 + [raw1[0]], en1),
       'N4 nan': ([raw1[0].replace(' mse=', ' mse=nan x=')] + raw1[1:], en1),
       'N5 mixed commits': ([raw1[0].replace('sha=dry0000', 'sha=dry1111')] + raw1[1:], en1),
       'N6 missing energy row': (raw1, en1[:-1]),
       'N7 energy identity off': ([raw1[0].replace('syn_energy_uJ=', 'syn_energy_uJ=99 x=')] + raw1[1:], en1),
       'N8 blocks 2': ([raw1[0].replace('blocks=1', 'blocks=2')] + raw1[1:], en1)}
for i, (nm, (rw, ev)) in enumerate(bad.items()):
    rc, d = run(f'dry_sxa_N{i + 1}', rw, ev)
    check(f'{nm}: exit 2, nothing written', rc == 2 and not os.path.exists(os.path.join(d, 'summary.json'))
          and not os.path.exists(os.path.join(d, 'selected_s1.json')), f'rc {rc}')
for nm in os.listdir(os.path.join(C, 'results')):
    if nm.startswith('dry_sxa_'):
        shutil.rmtree(os.path.join(C, 'results', nm))
print('ALL PASS' if fails == 0 else f'{fails} FAIL')
sys.exit(0 if fails == 0 else 1)
