"""Dry run of summarize_sxa2.py and recheck_sxa2.py on fabricated rows (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG stage-2
pre-registration). No training. Temporary suites results/dry_sxa2_* are removed at the end.
  D1 stage-1 selection sxppc shared; A1 = S = base; OT x 0.97 -> 'better' vs S and vs A1; PM = base; OM x 1.02 -> 'worse' vs PM and
     vs OT; validation PM x 0.98 -> selected msxppc shared; energy ratios = hand values; the independent recheck passes
  N1-N8 fail closed (exit 2, nothing written): SKIP line, 35 rows, a row of an unexpected arm, nan, mixed commits, missing energy
     row, energy identity off, a stage-1 S row missing
usage (from cinc_mixer_v1/): python -B results/check-sxa-20261127/dry_sxa2.py
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
fails = 0


def check(label, c, detail=''):
    global fails
    fails += not c
    print(f'{"PASS" if c else "FAIL"} {label}{": " + detail if detail else ""}')


def rows(arms, fac, vfac, body, sha='dry0000'):
    raw, en = [], ['ds,pl,cond,tag,seed,body_syn_uJ,body_syn_int_uJ,head_MAC,head_AC_corr,rule\n']
    for comb, sh in arms:
        cond = f'spkmx_lif_{comb}_add'
        for d, p in CELLS:
            tag = f'{cond}_dry{d}' + ('_ws' if sh else '')
            for s in SEEDS:
                base = 0.3 + 0.01 * CELLS.index((d, p)) + 0.001 * SEEDS.index(s)
                b = body[(comb, sh)]
                raw.append(f'RESULT exp=x ds={d} pl={p} cond={cond} tag={tag} blocks=1 seed={s} mse={base * fac[(comb, sh)]!r} '
                           f'mae={base * fac[(comb, sh)]!r} val_loss_best={0.5 * vfac[(comb, sh)]!r} syn_energy_uJ={b + 4.6!r} sha={sha} dir=x\n')
                en.append(f'{d},{p},{cond},{tag},{s},{b!r},{b - 1!r},1000000,2000000,binary\n')
    return raw, en


def write(name, raw, en, sel=None):
    d = os.path.join(C, 'results', name)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, 'raw.txt'), 'w').writelines(raw)
    open(os.path.join(d, 'energy_recount.csv'), 'w').writelines(en)
    if sel is not None:
        json.dump(sel, open(os.path.join(d, 'selected_s1.json'), 'w'))
    return d


A1, S_ = ('muxlin', False), ('sxppc', True)
OT, PM, OM = ('sxpot', True), ('msxppc', True), ('msxpot', True)
fac = {A1: 1., S_: 1., OT: 0.97, PM: 1., OM: 1.02}
vfac = {A1: 1., S_: 1., OT: 1., PM: 0.98, OM: 1.}
body = {A1: 5., S_: 6., OT: 7., PM: 8., OM: 9.}
r1, e1 = rows([A1, S_], fac, vfac, body)
r2, e2 = rows([OT, PM, OM], fac, vfac, body)


def run(case, raw2, en2, raw1=r1, en1=e1):
    write(f'dry_sxa2_{case}_s1', raw1, en1, dict(comb='sxppc', shared=True, score=0.9))
    d = write(f'dry_sxa2_{case}', raw2, en2)
    r = subprocess.run([PY, '-B', os.path.join(C, 'summarize_sxa2.py'), '--suite', f'dry_sxa2_{case}', '--s1_suite', f'dry_sxa2_{case}_s1'],
                       capture_output=True, text=True, cwd=C)
    return r.returncode, d


rc, d = run('D1', r2, e2)
S = json.load(open(os.path.join(d, 'summary.json'))) if rc == 0 else None
check('D1 summarizer exit 0', rc == 0)
if S:
    cm = {(c['A'], c['B']): c for c in S['comparisons']}
    c = cm[('sxpot shared', 'sxppc shared')]
    check('D1 OT vs S: R -3 %, 12 lower, better', abs(c['mse']['R_pct'] + 3) < 1e-9 and c['mse']['pairs_lower'] == 12 and c['mse']['verdict'] == 'better')
    c = cm[('msxpot shared', 'msxppc shared')]
    check('D1 OM vs PM: R +2 %, worse', abs(c['mse']['R_pct'] - 2) < 1e-9 and c['mse']['verdict'] == 'worse')
    c = cm[('msxpot shared', 'sxpot shared')]
    check('D1 OM vs OT (MUX): worse', c['mse']['verdict'] == 'worse')
    c = cm[('sxpot shared', 'muxlin indep.')]
    check('D1 OT vs A1: better; energy ratio (7 + 4.6) / (5 + 4.6)', c['mse']['verdict'] == 'better' and abs(c['energy_syops'] - 11.6 / 9.6) < 1e-12
          and abs(c['energy_int'] - (6 + 1.8) / (4 + 1.8)) < 1e-12)
    check('D1 7 comparisons', len(S['comparisons']) == 7)
    check('D1 selection = msxppc shared (validation x 0.98)', S['selection']['selected'] == 'msxppc shared'
          and json.load(open(os.path.join(d, 'selected_s2.json')))['comb'] == 'msxppc')
    r = subprocess.run([PY, '-B', os.path.join(HERE, 'recheck_sxa2.py'), '--suite', 'dry_sxa2_D1', '--s1_suite', 'dry_sxa2_D1_s1'],
                       capture_output=True, text=True, cwd=C)
    check('D1 independent recheck passes', r.returncode == 0 and 'ALL CHECKS PASSED' in r.stdout, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-300:])
ext, _ = rows([('sxdot', True)], {('sxdot', True): 1.}, {('sxdot', True): 1.}, {('sxdot', True): 5.})
bad = {'N1 SKIP line': (r2 + ['SKIP(rc=1) x\n'], e2, r1, e1),
       'N2 35 rows': (r2[:-1], e2, r1, e1),
       'N3 unexpected arm': (r2 + ext[:1], e2, r1, e1),
       'N4 nan': ([r2[0].replace(' mse=', ' mse=nan x=')] + r2[1:], e2, r1, e1),
       'N5 mixed commits': ([r2[0].replace('sha=dry0000', 'sha=dry1111')] + r2[1:], e2, r1, e1),
       'N6 missing energy row': (r2, e2[:-1], r1, e1),
       'N7 energy identity off': ([r2[0].replace('syn_energy_uJ=', 'syn_energy_uJ=99 x=')] + r2[1:], e2, r1, e1),
       'N8 stage-1 S row missing': (r2, e2, r1[:-1], e1)}
for i, (nm, (rw2, ev2, rw1, ev1)) in enumerate(bad.items()):
    rc, d = run(f'N{i + 1}', rw2, ev2, rw1, ev1)
    check(f'{nm}: exit 2, nothing written', rc == 2 and not os.path.exists(os.path.join(d, 'summary.json'))
          and not os.path.exists(os.path.join(d, 'selected_s2.json')), f'rc {rc}')
for nm in os.listdir(os.path.join(C, 'results')):
    if nm.startswith('dry_sxa2_'):
        shutil.rmtree(os.path.join(C, 'results', nm))
print('ALL PASS' if fails == 0 else f'{fails} FAIL')
sys.exit(0 if fails == 0 else 1)
