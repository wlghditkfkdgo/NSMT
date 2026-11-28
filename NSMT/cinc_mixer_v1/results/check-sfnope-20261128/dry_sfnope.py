"""Dry run of summarize_sfnope.py with planted rows (2026-11-28, exp/f-lif-pop-v3; no training). A temporary results root holds the
real reference suites (symlinks) and fake new suites built from the PE-on rows (tag + _pe0, planted macro-F1 / MSE, one commit),
plus fake order JSONs. Planted readings: (A) MIT-BIH PE off = PE on - 2 %p per seed -> M1 'A worse', M2 'A better'; ETT PE off =
PE on -> E1 'no clear difference'; (B) MIT-BIH + 0.1 %p -> M1 'no clear difference'; ETT x 1.03 -> E1 'A worse'. Stop cases (exit 2):
SKIP line, mixed commits, MIT-BIH order drop 0.2 %p, ETT SSA shuffle 0.1 %, missing energy recount row, a NaN MSE.
usage (from cinc_mixer_v1/): python -B results/check-sfnope-20261128/dry_sfnope.py
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
SUM = os.path.join(HERE, 'summarize_sfnope.py')
REFS = ('flif-v1-mitbih-sfde', 'flif-v1-mitbih-spkmux-tune', 'flif-v1-mitbih-memoff', 'flif-v1-ett-sfde-h1m1', 'flif-v1-ett-spkmux-retune',
        'flif-v1-ett-memoff')
fails = 0


def check(label, c):
    global fails
    fails += not c
    print(f'{"PASS" if c else "FAIL"} {label}', flush=True)


def parse(l):
    return dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)


def build(f1_shift, mse_scale, skip=False, mixed=False, mdrop=0., edrop=0., drop_energy=False, nan=False):
    root = tempfile.mkdtemp(); od = tempfile.mkdtemp()
    for s in REFS:
        os.symlink(os.path.join(C, 'results', s), os.path.join(root, s))
    # MIT-BIH new suite
    md = os.path.join(root, 'flif-v1-mitbih-sfde-nope'); os.makedirs(md)
    src = [l for l in open(os.path.join(C, 'results', 'flif-v1-mitbih-sfde', 'raw.txt')) if l.startswith('RESULT') and parse(l)['tag'] == 'spk_ssa_lr0.01_L2_nalif']
    out, en = [], []
    for i, l in enumerate(src):
        r = parse(l); f1 = float(r['f1']) + f1_shift / 100.
        sha = 'bbbbbbbbb' if (mixed and i == 0) else 'aaaaaaaaa'
        l2 = l.replace(f"tag={r['tag']}", f"tag={r['tag']}_pe0").replace(f" f1={r['f1']} ", f" f1={f1:.6f} ").replace(f"sha={r['sha']}", f'sha={sha}')
        out.append(l2.rstrip('\n'))
        en.append(dict(cond='spk_ssa', tag=r['tag'] + '_pe0', seed=r['seed'], f1_recomputed=f'{f1:.6f}'))
    if skip:
        out.append('SKIP(rc=1, no result) cond=spk_ssa seed=7')
    open(os.path.join(md, 'raw.txt'), 'w').write('\n'.join(out) + '\n')
    with open(os.path.join(md, 'energy_recount.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(en[0])); w.writeheader(); w.writerows(en)
    json.dump(dict(arm='sfdenope', tag='spk_ssa_lr0.01_L2_nalif_pe0',
                   runs=[dict(seed=e['seed'], mixer=dict(drop_f1_pp=mdrop), full=dict(drop_f1_pp=0.)) for e in en]),
              open(os.path.join(od, 'order_sfdenope.json'), 'w'))
    # ETT new suite
    ed = os.path.join(root, 'flif-v1-ett-sfde-nope'); os.makedirs(ed)
    erows = {(e['tag'], e['ds'], e['pl'], e['seed']): e for e in csv.DictReader(open(os.path.join(C, 'results', 'flif-v1-ett-sfde-h1m1', 'energy_recount.csv')))}
    src = [l for l in open(os.path.join(C, 'results', 'flif-v1-ett-sfde-h1m1', 'raw.txt')) if l.startswith('RESULT')]
    out, en, orr = [], [], []
    for i, l in enumerate(src):
        r = parse(l)
        mse = float('nan') if (nan and i == 0) else float(r['mse']) * mse_scale
        l2 = l.replace(f"tag={r['tag']}", f"tag={r['tag']}_pe0").replace(f" mse={r['mse']} ", f" mse={mse:.6f} ").replace(f"sha={r['sha']}", 'sha=ccccccccc')
        out.append(l2.rstrip('\n'))
        e = dict(erows[(r['tag'], r['ds'], r['pl'], r['seed'])]); e['tag'] = r['tag'] + '_pe0'
        if not (drop_energy and i == 0):
            en.append(e)
        orr.append(dict(ds=r['ds'], pl=r['pl'], seed=r['seed'], mixer=dict(d_mse_pct=edrop), head=dict(d_mse_pct=20.), full=dict(d_mse_pct=21.)))
    open(os.path.join(ed, 'raw.txt'), 'w').write('\n'.join(out) + '\n')
    with open(os.path.join(ed, 'energy_recount.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(en[0])); w.writeheader(); w.writerows(en)
    json.dump(dict(arm='sfde1n', runs=orr), open(os.path.join(od, 'order_sfde1n.json'), 'w'))
    return root, od


def run(**kw):
    root, od = build(**kw)
    p = subprocess.run([sys.executable, '-B', SUM, '--root', root, '--order_dir', od, '--out_dir', od], capture_output=True, text=True)
    js = json.load(open(os.path.join(od, 'summary_sfnope.json'))) if os.path.exists(os.path.join(od, 'summary_sfnope.json')) else None
    shutil.rmtree(root); shutil.rmtree(od)
    return p.returncode, p.stdout + p.stderr, js


rc, out, js = run(f1_shift=-2., mse_scale=1.)
mc = {c['kind']: c for c in js['mitbih']['comparisons']} if js else {}
ec = {c['kind']: c for c in js['ett']['comparisons']} if js else {}
check(f'(A) rc {rc}; M1 {mc.get("M1", {}).get("verdict")} (want A worse), mean {mc.get("M1", {}).get("mean_pp")}', rc == 0 and mc['M1']['verdict'] == 'A worse'
      and abs(mc['M1']['mean_pp'] + 2.) < 1e-3)
check(f'(A) M2 {mc["M2 (main)"]["verdict"]} (want A better), mean {mc["M2 (main)"]["mean_pp"]:.3f}', mc['M2 (main)']['verdict'] == 'A better')
check(f'(A) E1 {ec["E1"]["mse"]["verdict"]} (want no clear difference), R {ec["E1"]["mse"]["R_pct"]:.4f}', ec['E1']['mse']['verdict'] == 'no clear difference'
      and abs(ec['E1']['mse']['R_pct']) < 1e-3)
check(f'(A) E1 energy ratio {ec["E1"]["energy_ratio"]:.4f} = 1', abs(ec['E1']['energy_ratio'] - 1.) < 1e-9)
rc, out, js = run(f1_shift=0.1, mse_scale=1.03)
mc = {c['kind']: c for c in js['mitbih']['comparisons']} if js else {}
ec = {c['kind']: c for c in js['ett']['comparisons']} if js else {}
check(f'(B) rc {rc}; M1 {mc["M1"]["verdict"]} (want no clear difference)', rc == 0 and mc['M1']['verdict'] == 'no clear difference')
check(f'(B) E1 {ec["E1"]["mse"]["verdict"]} (want A worse), R {ec["E1"]["mse"]["R_pct"]:.3f}, higher {ec["E1"]["mse"]["higher"]}', ec['E1']['mse']['verdict'] == 'A worse')
for nm, kw, msg in [('SKIP line', dict(skip=True), 'SKIP'), ('mixed commits', dict(mixed=True), 'mixed training commits'),
                    ('MIT-BIH order drop 0.2', dict(mdrop=0.2), 'order-blind'), ('ETT SSA shuffle 0.1 %', dict(edrop=0.1), 'equivariant'),
                    ('missing energy row', dict(drop_energy=True), 'no energy recount row'), ('NaN MSE', dict(nan=True), 'not finite')]:
    rc, out, js = run(f1_shift=-2., mse_scale=1., **kw)
    check(f'stop on {nm}: rc {rc}, "{out.strip().splitlines()[-1] if out.strip() else ""}"', rc == 2 and msg in out and js is None)
print('ALL PASS' if fails == 0 else f'{fails} FAIL')
sys.exit(0 if fails == 0 else 1)
