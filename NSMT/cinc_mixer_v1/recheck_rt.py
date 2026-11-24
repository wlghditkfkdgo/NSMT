"""Independent recheck of the ETTh1 / ETTm1 retuning (2026-11-25, exp/f-lif-pop-v3; PROJECT_LOG pre-registration of 2026-11-25).
Imports nothing from this repository (no rt_plan, no summarize_rt): the plan (start values, candidate lists, order, tag rule) is
written out again here. From raw.txt alone it rebuilds the selection chain (it never reads a selected_*.json to decide the next
stage's configuration), then compares every selection it reached with the selected_*.json on disk.
  --stage <s>   checks every stage up to and including <s> (lr layers patch dmodel mlp blocks wd)
  --stage judge also rebuilds both 16-cell models (ETTh2 / ETTm2 rows from the old suites + full16 raw files) and recomputes the
                16-cell means, the 8 margins, pass flags, energy cells below SpikF and the verdict; compares with judge.json (1e-9)
  --sha a,b     every retune row was trained at commit a; commits after a (b, ...) change no .py / .sh file under NSMT/
Prints PASS / FAIL lines and "ALL CHECKS PASSED"; exit 1 on any FAIL.
usage (from cinc_mixer_v1/): python -B recheck_rt.py --stage <s>|judge [--sha a,b]
"""
import argparse
import csv
import json
import os
import subprocess
import sys

C = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(C, 'results')
SU = 'flif-v1-ett-spkmux-retune'
OUT = os.path.join(os.environ['RT_DRY'], SU) if os.environ.get('RT_DRY') else os.path.join(R, SU)
DS2, SEEDS = ('ETTh1', 'ETTm1'), (42, 7, 2026)
KNOBS = ['LR', 'LAYERS', 'PATCH', 'EMB', 'MLPR', 'BLOCKS', 'WD']
FLD = ['lr', 'layers', 'patch', 'emb', 'mlpr', 'blocks', 'wd']
START = {'ETTh1': ['0.0005', '2', '8', '32', '1', '1', '0.06'], 'ETTm1': ['0.001', '2', '8', '64', '1', '1', '0.06']}
MLP_DEFAULT = {'ETTh1': '0.5', 'ETTm1': '1'}
SH = [('lr', 0, ['0.0001', '0.00025', '0.0005', '0.001', '0.002']), ('layers', 1, ['1', '2', '3'])]
CHA = {'a': [('patch', 2, ['4', '8', '16']), ('dmodel', 3, ['32', '64', '128', '256']), ('mlp', 4, ['0.5', '1', '2', '4']),
             ('blocks', 5, ['1', '2', '3']), ('wd', 6, ['0', '0.01', '0.06', '0.1'])],
       'x': [('patch', 2, ['4', '8', '16', '24', '32']), ('dmodel', 3, ['32', '64', '128', '256']), ('mlp', 4, ['0.5', '1', '2', '4']),
             ('blocks', 5, ['1', '2', '3']), ('wd', 6, ['0', '0.005', '0.01', '0.02', '0.05', '0.06'])]}
ORDER = ['lr', 'layers', 'patch', 'dmodel', 'mlp', 'blocks', 'wd']
fails = 0


def ok(c, m):
    global fails
    print(('PASS ' if c else 'FAIL ') + m)
    fails += 0 if c else 1


def tagof(ds, v):
    t = f'spkmx_lif_muxlin_iand_lr{v[0]}_L{v[1]}'
    for i, (suf, dflt) in enumerate([('_pt', '8'), ('_e', '64'), ('_mlp', MLP_DEFAULT[ds]), ('_blk', '1'), ('_wd', '0.06')]):
        if v[2 + i] != dflt:
            t += f'{suf}{v[2 + i]}'
    return t


def result_lines(path):
    out, skips = [], 0
    for l in open(path):
        skips += l.startswith('SKIP')
        if l.startswith('RESULT'):
            out.append(dict(t.split('=', 1) for t in l.split()[1:] if '=' in t))
    return out, skips


ap = argparse.ArgumentParser()
ap.add_argument('--stage', required=True, choices=ORDER + ['judge'])
ap.add_argument('--sha', default='')
a = ap.parse_args()
rows, skips = result_lines(os.path.join(OUT, 'raw.txt'))
ok(skips == 0, f'no SKIP line in the retune suite ({skips})')
V = {}
bad = []
for r in rows:
    v = tuple(r[f] for f in FLD)
    k = (r['ds'], int(r['pl']), int(r['seed']), v)
    if k in V:
        bad.append(f'duplicate {k}')
    if r['tag'] != tagof(r['ds'], v):
        bad.append(f'tag {r["tag"]} != {tagof(r["ds"], v)}')
    if '+epoch+50+' not in r['dir'] or r['cond'] != 'spkmx_lif_muxlin_iand':
        bad.append(f'epochs / cond of {r["tag"]}')
    V[k] = r
ok(not bad, f'{len(rows)} rows: tags from fields, 50 epochs, no duplicate' + ('' if not bad else ': ' + '; '.join(bad[:5])))


def valmean(ds, pl, v):
    x = [V.get((ds, pl, s, tuple(v))) for s in SEEDS]
    if any(y is None for y in x):
        return None
    return sum(float(y['best_epoch_val_epochs'].split(':')[1]) for y in x) / 3


def choose(ds, base, idx, vals):
    """-> (selected value, scores) or (None, missing list); the rule of rt_plan.py written again."""
    cur = base[idx]
    m = {}
    for val in vals:
        w = list(base); w[idx] = val
        m[val] = [valmean(ds, pl, w) for pl in (96, 720)]
    miss = [val for val in vals if None in m[val]]
    if miss:
        return None, miss
    sc = {val: (m[val][0] / m[cur][0] + m[val][1] / m[cur][1]) / 2 for val in vals}
    lo = min(sc.values())
    if abs(sc[cur] - lo) < 1e-12:
        return cur, sc
    return [val for val in vals if sc[val] == lo][0], sc


asked = {ds: set() for ds in DS2}
upto = ORDER.index(a.stage) if a.stage != 'judge' else len(ORDER) - 1
base = {ds: list(START[ds]) for ds in DS2}
final = None
for name, idx, vals in SH:
    if ORDER.index(name) > upto:
        break
    f = os.path.join(OUT, f'selected_{name}.json')
    js = json.load(open(f)) if os.path.exists(f) else None
    for ds in DS2:
        for val in vals:
            w = list(base[ds]); w[idx] = val; asked[ds].add(tuple(w))
        s, sc = choose(ds, base[ds], idx, vals)
        ok(s is not None and js is not None and js['selected'][ds] == s and abs(js['scores'][ds][s] - sc[s]) < 1e-9,
           f'{name} {ds}: recomputed {s} (score {sc[s] if s else sc}) = selected_{name}.json {js["selected"][ds] if js else None}')
        if s is not None:
            base[ds][idx] = s
cb = {c: {ds: list(base[ds]) for ds in DS2} for c in CHA}
for c in CHA:
    for name, idx, vals in CHA[c]:
        if ORDER.index(name) > upto:
            break
        f = os.path.join(OUT, f'selected_{c}_{name}.json')
        js = json.load(open(f)) if os.path.exists(f) else None
        for ds in DS2:
            for val in vals:
                w = list(cb[c][ds]); w[idx] = val; asked[ds].add(tuple(w))
            s, sc = choose(ds, cb[c][ds], idx, vals)
            ok(s is not None and js is not None and js['selected'][ds] == s and abs(js['scores'][ds][s] - sc[s]) < 1e-9,
               f'chain {c} {name} {ds}: recomputed {s} = selected_{c}_{name}.json {js["selected"][ds] if js else None}')
            if s is not None:
                cb[c][ds][idx] = s
extra = [k for k in V if k[1] in (96, 720) and k[3] not in asked[k[0]]]
if a.stage == 'judge':
    final = cb
    fin = {(ds, tuple(cb[c][ds])) for c in CHA for ds in DS2}
    extra += [k for k in V if k[1] in (192, 336) and (k[0], k[3]) not in fin]
    ok(not extra, f'every row belongs to a requested configuration ({len(extra)} extra)')
else:
    ok(not extra and not any(k[1] in (192, 336) for k in V), f'every H96 / H720 row belongs to a stage up to {a.stage}, no H192 / H336 row yet ({len(extra)} extra)')

if a.sha:
    shas = a.sha.split(',')
    ok(all(r['sha'] == shas[0] for r in rows), f'all {len(rows)} rows trained at {shas[0]} ({sorted({r["sha"] for r in rows})})')
    for s in shas[1:]:
        ch = subprocess.run(['git', 'diff', '--name-only', shas[0], s, '--', '.'], cwd=os.path.dirname(C), capture_output=True, text=True).stdout.split()
        code = [x for x in ch if x.endswith(('.py', '.sh'))]
        ok(not code, f'commit {s} changes no .py / .sh under NSMT/ since {shas[0]} ({code})')

if a.stage == 'judge':
    J = json.load(open(os.path.join(OUT, 'judge.json')))
    OLDT = {'first': ('flif-v1-ett-spkmux-ettune', {'ETTh2': 'spkmx_lif_muxlin_iand_lr0.0005_L1_pt16_wd0.1', 'ETTm2': 'spkmx_lif_muxlin_iand_lr0.0001_L1_pt16_mlp2_wd0.1'}),
            'ext': ('flif-v1-ett-spkmux-ettune2', {'ETTh2': 'spkmx_lif_muxlin_iand_lr0.0005_L1_pt24_wd0.005', 'ETTm2': 'spkmx_lif_muxlin_iand_lr0.0001_L1_pt24_e256_blk2_wd0.02'})}
    REF = json.load(open(os.path.join(R, 'ett-full16-20261124', 'refs16.json')))['models']
    SP = {}
    for ds in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
        for r in csv.DictReader(open(os.path.join(R, 'ett-full16-20261124', 'spikf16', f'spikf_syops_{ds}.csv'))):
            SP[(r['ds'], int(r['pl']))] = float(r['energy_uJ'])
    cells = [(d, p) for d in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2') for p in (96, 192, 336, 720)]
    for mk, ch in (('first', 'a'), ('ext', 'x')):
        M = {}
        for suite, pls in ((OLDT[mk][0], (96, 720)), ('flif-v1-ett-spkmux-full16', (192, 336))):
            for r in result_lines(os.path.join(R, suite, 'raw.txt'))[0]:
                if r['ds'] in OLDT[mk][1] and r['tag'] == OLDT[mk][1][r['ds']] and int(r['pl']) in pls:
                    M[(r['ds'], int(r['pl']), int(r['seed']))] = [float(r[x]) for x in ('mse', 'mae', 'syn_energy_uJ')]
        for ds in DS2:
            for pl in (96, 192, 336, 720):
                for s in SEEDS:
                    r = V.get((ds, pl, s, tuple(final[ch][ds])))
                    if r is not None:
                        M[(ds, pl, s)] = [float(r[x]) for x in ('mse', 'mae', 'syn_energy_uJ')]
        ok(len(M) == 48, f'{mk}: 48 rows (16 cells x 3 seeds) ({len(M)})')
        if len(M) != 48:
            continue
        cmn = {c: [sum(M[(c[0], c[1], s)][j] for s in SEEDS) / 3 for j in range(3)] for c in cells}
        om = sum(cmn[c][0] for c in cells) / 16; oa = sum(cmn[c][1] for c in cells) / 16
        G = J['models'][mk]
        ok(abs(G['mse16'] - om) < 1e-9 and abs(G['mae16'] - oa) < 1e-9, f'{mk}: 16-cell MSE / MAE {om:.6f} / {oa:.6f} = judge.json')
        acc = True
        names = [m for m in REF if REF[m]['role'] in ('SNN', 'DNN')]
        ok(sorted(names) == sorted(G['rows']) and len(names) == 8, f'{mk}: the 8 reference rows (no NSMT) = judge.json rows')
        for m in names:
            rm = sum(REF[m]['rows'][f'{d} {p}'][0] for d, p in cells) / 16
            ra = sum(REF[m]['rows'][f'{d} {p}'][1] for d, p in cells) / 16
            ps = rm - om >= 0.001 - 1e-12 and ra - oa >= 0.001 - 1e-12
            g = G['rows'].get(m, {})
            ok(abs(g.get('margin_mse', 9) - (rm - om)) < 1e-9 and abs(g.get('margin_mae', 9) - (ra - oa)) < 1e-9 and g.get('passes') == ps,
               f'{mk} vs {m}: margin {rm - om:+.6f} / {ra - oa:+.6f}, passes {ps} = judge.json')
            acc = acc and ps
        below = sum(cmn[c][2] < SP[c] for c in cells)
        verdict = 'PASS' if acc and below == 16 else 'NOT PASSED'
        ok(G['energy_cells_below_spikf'] == below and G['verdict'] == verdict, f'{mk}: energy below SpikF {below}/16, verdict {verdict} = judge.json')
        print(f'  {mk}: final ETTh1 {final[ch]["ETTh1"]} ETTm1 {final[ch]["ETTm1"]} (lr, M, patch, d_model, MLP, blocks, wd)')
print('ALL CHECKS PASSED' if fails == 0 else f'{fails} CHECKS FAILED')
sys.exit(0 if fails == 0 else 1)
