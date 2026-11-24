"""Dry run of summarize_full16.py on a fabricated suite (2026-11-24, exp/f-lif-pop-v3; PROJECT_LOG full-ETT-table pre-registration).
No training. The fabricated suite holds 72 RESULT lines and energy rows built from the real rows of the 4 trained cells:
  new cell (ds, H) of a model <- the model's real row of (ETTh2 if ds is hourly else ETTm2, 96 if H <= 192 else 720) and the same seed,
  with mse + 0.001 * i and mae + 0.0005 * i (i = index of the new cell, 1..12) so that every cell differs; energy fields unchanged.
D1  the summary's 16-cell means, every margin, every pass flag, the energy count and the verdict equal an independent computation from
    the fabricated lines + refs16.json + the SpikF csv files (plain arithmetic here, none of the summarizer's functions)
D2  the 4-cell means in the summary equal the final.json of each chain
N1-N5 (each must exit 2 and write no summary): one row missing; a SKIP line; a duplicated row; a non-finite mse; an energy row whose
    recomputed MSE differs by 1e-3
usage (from cinc_mixer_v1/): python -B results/check-full16-20261124/check_dry_full.py > results/check-full16-20261124/check_dry_full.txt
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUM = os.path.join(C, 'results', 'ett-full16-20261124', 'summarize_full16.py')
DSS, PLS, SEEDS = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'), (96, 192, 336, 720), (42, 7, 2026)
OLD = [('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720)]
NEW = [(d, p) for d in DSS for p in PLS if (d, p) not in OLD]
SRC = {'first': ('flif-v1-ett-spkmux-ettune', {'h': 'spkmx_lif_muxlin_iand_lr0.0005_L1_pt16_wd0.1', 'm': 'spkmx_lif_muxlin_iand_lr0.0001_L1_pt16_mlp2_wd0.1'}),
       'ext': ('flif-v1-ett-spkmux-ettune2', {'h': 'spkmx_lif_muxlin_iand_lr0.0005_L1_pt24_wd0.005', 'm': 'spkmx_lif_muxlin_iand_lr0.0001_L1_pt24_e256_blk2_wd0.02'})}
fails = 0


def ok(c, m):
    global fails
    print(('PASS ' if c else 'FAIL ') + m)
    fails += 0 if c else 1


def fam(ds):
    return 'h' if ds.startswith('ETTh') else 'm'


real, ereal = {}, {}
for mk, (suite, tags) in SRC.items():
    for l in open(os.path.join(C, 'results', suite, 'raw.txt')):
        if l.startswith('RESULT'):
            r = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
            if r['tag'] == tags[fam(r['ds'])] and (r['ds'], int(r['pl'])) in OLD:
                real[(mk, r['ds'], int(r['pl']), int(r['seed']))] = (l.split(), r)
    for e in csv.DictReader(open(os.path.join(C, 'results', suite, 'energy_recount.csv'))):
        if e['tag'] == tags[fam(e['ds'])] and (e['ds'], int(e['pl'])) in OLD:
            ereal[(mk, e['ds'], int(e['pl']), int(e['seed']))] = e
ok(len(real) == 24 and len(ereal) == 24, f'real rows of the 4 trained cells found: {len(real)} result rows, {len(ereal)} energy rows (want 24 / 24)')
lines, erows, V = [], [], {}
for mk in SRC:
    for (d, p) in OLD:
        for s in SEEDS:
            r = real[(mk, d, p, s)][1]
            V[(mk, d, p, s)] = (float(r['mse']), float(r['mae']), float(r['syn_energy_uJ']))
    for i, (d, p) in enumerate(NEW, 1):
        sd, sp = ('ETTh2' if fam(d) == 'h' else 'ETTm2'), (96 if p <= 192 else 720)
        for s in SEEDS:
            toks, r = real[(mk, sd, sp, s)]
            mse, mae = float(r['mse']) + 0.001 * i, float(r['mae']) + 0.0005 * i
            new = []
            for t in toks:
                k = t.split('=', 1)[0]
                new.append({'ds': f'ds={d}', 'pl': f'pl={p}', 'mse': f'mse={mse:.6f}', 'mae': f'mae={mae:.6f}'}.get(k, t))
            lines.append(' '.join(new))
            e = dict(ereal[(mk, sd, sp, s)]); e.update(ds=d, pl=p, mse_logged=f'{mse:.6f}', mse_recomputed=f'{mse:.6f}')
            erows.append(e)
            V[(mk, d, p, s)] = (mse, mae, float(r['syn_energy_uJ']))


def write(folder, L, E):
    os.makedirs(folder, exist_ok=True)
    open(os.path.join(folder, 'raw.txt'), 'w').write('\n'.join(L) + '\n')
    with open(os.path.join(folder, 'energy_recount.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(E[0].keys())); w.writeheader(); [w.writerow(x) for x in E]


def run(folder):
    for f in ('summary.md', 'summary.json'):
        if os.path.exists(os.path.join(folder, f)):
            os.remove(os.path.join(folder, f))
    p = subprocess.run([sys.executable, '-B', SUM], cwd=C, env=dict(os.environ, FULL_DIR=folder), capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip().splitlines()[-1:] if p.returncode else [''], os.path.exists(os.path.join(folder, 'summary.json'))


T = tempfile.mkdtemp()
try:
    d0 = os.path.join(T, 'ok'); write(d0, lines, erows)
    rc, last, has = run(d0)
    ok(rc == 0 and has, f'summarizer runs on the fabricated suite (rc {rc})')
    if rc == 0:
        S = json.load(open(os.path.join(d0, 'summary.json')))
        REF = json.load(open(os.path.join(C, 'results', 'ett-full16-20261124', 'refs16.json')))['models']
        SP = {}
        for ds in DSS:
            for r in csv.DictReader(open(os.path.join(C, 'results', 'ett-full16-20261124', 'spikf16', f'spikf_syops_{ds}.csv'))):
                SP[(r['ds'], int(r['pl']))] = float(r['energy_uJ'])
        bad = []
        for mk, (suite, _) in SRC.items():
            cm = {(d, p): [np.mean([V[(mk, d, p, s)][j] for s in SEEDS]) for j in range(3)] for d in DSS for p in PLS}
            om, oa = np.mean([cm[c][0] for c in cm]), np.mean([cm[c][1] for c in cm])
            M = S['models'][mk]
            if abs(M['mse16'] - om) > 1e-9 or abs(M['mae16'] - oa) > 1e-9:
                bad.append(f'{mk} 16-cell means {M["mse16"]} / {M["mae16"]} vs {om} / {oa}')
            acc = True
            for m, v in REF.items():
                rm = np.mean([v['rows'][f'{d} {p}'][0] for d in DSS for p in PLS]); ra = np.mean([v['rows'][f'{d} {p}'][1] for d in DSS for p in PLS])
                ps = (rm - om >= 0.001 - 1e-12) and (ra - oa >= 0.001 - 1e-12)
                g = M['rows'][m]
                if abs(g['margin_mse'] - (rm - om)) > 1e-9 or abs(g['margin_mae'] - (ra - oa)) > 1e-9 or g['passes'] != bool(ps):
                    bad.append(f'{mk} {m}: {g} vs margins {rm - om} / {ra - oa} pass {ps}')
                if v['role'] != 'own':
                    acc = acc and ps
            below = sum(cm[c][2] < SP[c] for c in cm)
            verdict = 'PASS' if acc and below == 16 else 'NOT PASSED'
            if M['energy_cells_below_spikf'] != below or M['verdict'] != verdict:
                bad.append(f'{mk}: energy cells {M["energy_cells_below_spikf"]} vs {below}, verdict {M["verdict"]} vs {verdict}')
            fj = json.load(open(os.path.join(C, 'results', suite, 'final.json')))['tuned']
            ok(abs(M['mse4'] - fj['mse']) < 1e-9 and abs(M['mae4'] - fj['mae']) < 1e-9, f'D2 {mk}: 4-cell means {M["mse4"]:.6f} / {M["mae4"]:.6f} = final.json')
            print(f'  {mk}: fabricated 16-cell means {om:.6f} / {oa:.6f}, verdict {verdict}, energy cells below SpikF {below}/16')
        ok(not bad, 'D1 summary = independent computation (16-cell means, 18 margins and pass flags, energy count, verdict)' + ('' if not bad else ': ' + '; '.join(bad)))
    neg = {'N1 one row missing': (lines[:-1], erows),
           'N2 a SKIP line': (lines + ['SKIP(rc=1, no result) ds=ETTh1 pl=96 cond=x seed=42 out=x'], erows),
           'N3 a duplicated row': (lines + [lines[0]], erows),
           'N4 a non-finite mse': ([' '.join('mse=nan' if t.startswith('mse=') else t for t in lines[0].split())] + lines[1:], erows),
           'N5 recomputed MSE off by 1e-3': (lines, [dict(erows[0], mse_recomputed=f'{float(erows[0]["mse_recomputed"]) + 1e-3:.6f}')] + erows[1:])}
    for nm, (Lx, Ex) in neg.items():
        d = os.path.join(T, nm.split()[0]); write(d, Lx, Ex)
        rc, last, has = run(d)
        ok(rc == 2 and not has, f'{nm}: exit {rc}, no summary written ({last[0][:110] if last else ""})')
finally:
    shutil.rmtree(T)
print(f'{"ALL PASS" if fails == 0 else str(fails) + " FAIL"}')
sys.exit(0 if fails == 0 else 1)
