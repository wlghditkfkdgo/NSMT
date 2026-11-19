"""Dry runs of summarize_pn.py --stage mxett and recheck_mxett.py (2026-11-19 evening, exp/f-lif-pop-v3; PROJECT_LOG mixer-only population ETT
pre-registration). No training. A fake suite flif-v1-ett-spkmux-mxpop is built from the 12 tuned LIF SpkMixer rows (flif-v1-ett-spkmux-tune) put
under the 4 arm names (cond / tag changed; values optionally scaled); the real SF-de suite is copied next to it (PN_DRY folders).
  A identity: every arm = LIF -> B1 = C3 = C5 = R 0, 0/12, energy 1.00; B3 = LIF vs pop LIF all; no arm passes -> LIF stays
  B two passing arms: pop LIF mixer+mean MSE / MAE x 0.995, pop f-LIF mixer x 0.99 -> pop f-LIF mixer (lowest MSE)
  C energy: pop LIF mixer MSE / MAE x 0.99 but logged total energy x 1.2 -> energy A/B > 1.05 -> not adopted -> LIF
  D fail closed (exit 2, no selection written): a SKIP line; 47 rows; an existing selected_structure.json with another choice
Each of A-C: the summarizer runs, recheck_mxett.py agrees (exit 0), and the expected choice / values hold.
usage (from cinc_mixer_v1/): python -B results/check-mxett-20261119/check_dry_mxett.py > results/check-mxett-20261119/check_dry_mxett.txt
"""
import csv
import json
import os
import re
import shutil
import subprocess
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(C, 'results')
PY = sys.executable
LRM = {'ETTh2': '0.0005', 'ETTm2': '0.0001'}
MX = {'pop LIF mixer': ('pop_lif', '_nsmixer'), 'pop LIF mixer+mean': ('pop_lif', '_mpmean_nsmixer'), 'pop f-LIF mixer': ('pop_flif', '_nsmixer'),
      'pop f-LIF mixer+mean': ('pop_flif', '_mpmean_nsmixer')}
fails = 0


def lif_rows():
    rr = [l for l in open(os.path.join(RES, 'flif-v1-ett-spkmux-tune', 'raw.txt')) if l.startswith('RESULT')]
    want = {f'tag=spkmx_lif_muxlin_iand_lr{lr}_L1' for lr in LRM.values()}
    out = []
    for l in rr:
        d = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
        if d['cond'] == 'spkmx_lif_muxlin_iand' and f'tag={d["tag"]}' in want and d['tag'] == f'spkmx_lif_muxlin_iand_lr{LRM[d["ds"]]}_L1' \
                and d['pl'] in ('96', '720') and d['seed'] in ('42', '7', '2026'):
            out.append(l.rstrip('\n'))
    assert len(out) == 12, len(out)
    en = [e for e in csv.DictReader(open(os.path.join(RES, 'flif-v1-ett-spkmux-tune', 'energy_recount.csv')))
          if e.get('tag') == f'spkmx_lif_muxlin_iand_lr{LRM[e["ds"]]}_L1' and e['pl'] in ('96', '720') and e['seed'] in ('42', '7', '2026')]
    assert len(en) == 12, len(en)
    return out, en


def build(name, scale=None, escale=None, drop=0, skip=False, presel=None):
    D = os.path.join(OUT, f'dry_{name}')
    shutil.rmtree(D, ignore_errors=True)
    S, F = os.path.join(D, 'flif-v1-ett-spkmux-mxpop'), os.path.join(D, 'flif-v1-ett-sfde')
    os.makedirs(S); os.makedirs(F)
    for f in ('raw.txt', 'energy_recount.csv', 'selected_lr.json'):
        shutil.copy(os.path.join(RES, 'flif-v1-ett-sfde', f), F)
    raw, en = lif_rows()
    scale, escale = scale or {}, escale or {}
    lines, erows = [], []
    for nm, (mx, suf) in MX.items():
        k, ke = scale.get(nm, 1.0), escale.get(nm, 1.0)
        cond = f'spkmx_{mx}_muxlin_iand'
        for l in raw:
            t = l.split()
            d = dict(x.split('=', 1) for x in t[1:] if '=' in x)
            tag = f'{cond}_lr{LRM[d["ds"]]}_L1{suf}'
            new = []
            for x in t:
                key = x.split('=', 1)[0]
                if key == 'cond':
                    x = f'cond={cond}'
                elif key == 'tag':
                    x = f'tag={tag}'
                elif key in ('mse', 'mae'):
                    x = f'{key}={float(d[key]) * k:.6f}'
                elif key == 'total_fft_uJ':
                    x = f'total_fft_uJ={float(d[key]) * ke:.6f}'
                new.append(x)
            lines.append(' '.join(new))
        for e in en:
            e2 = dict(e, cond=cond, tag=f'{cond}_lr{LRM[e["ds"]]}_L1{suf}')
            for key in ('mse_logged', 'mse_recomputed'):
                e2[key] = f'{float(e[key]) * k:.6f}'
            erows.append(e2)
    lines = lines[:len(lines) - drop]
    with open(os.path.join(S, 'raw.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
        if skip:
            f.write('SKIP(rc=1, no result) ds=ETTh2 pl=96 cond=spkmx_pop_lif_muxlin_iand seed=42 out=-\n')
    with open(os.path.join(S, 'energy_recount.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(erows[0])); w.writeheader(); w.writerows(erows)
    if presel:
        json.dump({'selected': presel}, open(os.path.join(S, 'selected_structure.json'), 'w'))
    return D


def run(D, args):
    env = dict(os.environ, PN_DRY=D, LD_LIBRARY_PATH=os.path.join(os.path.dirname(os.path.dirname(PY)), 'lib'))
    p = subprocess.run([PY, '-B'] + args, cwd=C, env=env, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def check(label, cond, detail=''):
    global fails
    fails += not cond
    print(f'{"PASS" if cond else "FAIL"} {label}{(": " + detail) if detail else ""}')


def scenario(name, want, **kw):
    D = build(name, **kw)
    rc, out = run(D, ['summarize_pn.py', '--stage', 'mxett'])
    check(f'{name}: summarizer exit 0', rc == 0, out.strip().splitlines()[-1] if rc else '')
    if rc:
        return None, None
    rc2, out2 = run(D, ['recheck_mxett.py'])
    check(f'{name}: recheck_mxett agrees', rc2 == 0 and 'ALL CHECKS PASSED' in out2, out2.strip().splitlines()[-1])
    sel = json.load(open(os.path.join(D, 'flif-v1-ett-spkmux-mxpop', 'selected_structure.json')))
    check(f'{name}: adopted {want}', sel['selected'] == want, f'got {sel["selected"]}')
    return sel, open(os.path.join(D, 'flif-v1-ett-spkmux-mxpop', 'summary.md')).read()


def row(summ, name):
    m = re.search(r'^\| ' + re.escape(name) + r' \| ([+-]\d+\.\d+) \| (\d+)/12 \| ([a-z ]+) \| (\d+\.\d+) \|', summ, re.M)
    return m.groups() if m else None


selA, sA = scenario('A', 'LIF')
if sA:
    for q in ('B1 pop LIF mixer vs LIF', 'C3 pop LIF mixer+mean vs pop LIF mixer', 'C5 pop f-LIF mixer+mean vs pop f-LIF mixer'):
        check(f'A: {q} = R 0, 0/12, energy 1.00', row(sA, q) == ('+0.00', '0', 'no clear difference', '1.00'), str(row(sA, q)))
    b3 = row(sA, 'B3 pop LIF mixer vs pop LIF all')
    rep = row(sA, '(report) LIF vs SF-de')
    check('A: B3 (= LIF vs pop LIF all) present', b3 is not None, str(b3))
    check('A: no arm passes', not any(v['passes'] for v in selA['scores'].values()))
    check('A: every arm d_mse = d_mae = 0', all(abs(v['d_mse']) < 1e-12 and abs(v['d_mae']) < 1e-12 for v in selA['scores'].values()))
    check('A: (report) LIF vs SF-de energy (syops) = 0.98 (17:34 re-read 0.9825)', rep is not None and rep[3] == '0.98', str(rep))
selB, _ = scenario('B', 'pop f-LIF mixer', scale={'pop LIF mixer+mean': 0.995, 'pop f-LIF mixer': 0.99})
if selB:
    check('B: both scaled arms pass, the others do not', [n for n, v in selB['scores'].items() if v['passes']] == ['pop LIF mixer+mean', 'pop f-LIF mixer']
          or sorted(n for n, v in selB['scores'].items() if v['passes']) == ['pop LIF mixer+mean', 'pop f-LIF mixer'])
selC, _ = scenario('C', 'LIF', scale={'pop LIF mixer': 0.99}, escale={'pop LIF mixer': 1.2})
if selC:
    v = selC['scores']['pop LIF mixer']
    check('C: pop LIF mixer passes MSE / MAE but energy A/B > 1.05', v['d_mse'] >= 0.001 and v['d_mae'] >= 0.001 and v['energy_ratio_sfde'] > 1.05 and not v['passes'],
          f"d {v['d_mse']:.4f} / {v['d_mae']:.4f}, energy {v['energy_ratio_sfde']:.4f}")
for name, kw in (('D1', dict(skip=True)), ('D2', dict(drop=1)), ('D3', dict(scale={'pop f-LIF mixer': 0.99}, presel='LIF'))):
    D = build(name, **kw)
    rc, out = run(D, ['summarize_pn.py', '--stage', 'mxett'])
    p = os.path.join(D, 'flif-v1-ett-spkmux-mxpop')
    wrote = os.path.exists(os.path.join(p, 'summary.md'))
    kept = json.load(open(os.path.join(p, 'selected_structure.json')))['selected'] == 'LIF' if name == 'D3' else not os.path.exists(os.path.join(p, 'selected_structure.json'))
    why = {'D1': 'SKIP line', 'D2': '47 RESULT rows', 'D3': 'already holds LIF'}[name]                # the stop must be for this reason
    check(f'{name} ({", ".join(kw)}): exit 2 for "{why}", no summary, selection untouched', rc == 2 and why in out and not wrote and kept,
          [l for l in out.splitlines() if 'STOP' in l][:1].__str__())
print('ALL DRY CHECKS PASSED' if fails == 0 else f'{fails} FAILED')
sys.exit(0 if fails == 0 else 1)
