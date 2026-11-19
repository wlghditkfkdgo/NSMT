"""Dry runs of the ETT tuning chain (2026-11-20, exp/f-lif-pop-v3; PROJECT_LOG ETT tuning pre-registration). No training.
For each stage: scripts/make_queue_ettune.sh (ET_DRY) builds the queue from the earlier selections; every queue line becomes a fake
RESULT row (the 11-01 default row of the same dataset / horizon / seed with the tag, the knob fields and the validation L1 / test MSE /
MAE multiplied by a factor of the configuration); summarize_ettune.py --stage and recheck_ettune.py --stage run; after wd a fake
energy_recount.csv (the default rows' recount rows, tag changed, recomputed MSE scaled) and the final stage run.
  A identity (every factor 1): every stage keeps the current value; final = default; tuned vs untuned R 0; pass result = the 21:09 LIF one
  B a chosen path: ETTh2 PATCH 16, EMB 128, MLPR 0.5 (kept), BLOCKS 2, WD 0.1; ETTm2 PATCH 4, EMB 64 (kept), MLPR 2, BLOCKS 1 (kept), WD 0;
    the queue of each stage carries the earlier winners; final 4-cell tuned MSE = the factor-weighted value
  C tie between two other values (ETTh2 PATCH 4 and 16 both 0.99): the earlier candidate (4)
  D fail closed: D1 SKIP line, D2 a missing row, D3 an existing selection with another value, D4 a row of a configuration no stage asked
    for, D5 a tag that disagrees with its fields -> summarizer exit 2 for that reason and no selected_patch.json;
    D6 make_queue dmodel without selected_patch.json -> exit 1; D7 final with one recount row missing -> exit 2, no final.json
usage (from cinc_mixer_v1/): python -B results/check-ettune-20261120/check_dry_ettune.py > results/check-ettune-20261120/check_dry_ettune.txt
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
LIB = os.path.join(os.path.dirname(os.path.dirname(PY)), 'lib')
LR = {'ETTh2': '0.0005', 'ETTm2': '0.0001'}
K = ('PATCH', 'EMB', 'MLPR', 'BLOCKS', 'WD')
FLD = dict(zip(K, ('patch', 'emb', 'mlpr', 'blocks', 'wd')))
SUF = dict(zip(K, ('_pt', '_e', '_mlp', '_blk', '_wd')))
D0 = {'ETTh2': dict(zip(K, ('8', '64', '0.5', '1', '0.06'))), 'ETTm2': dict(zip(K, ('8', '64', '1', '1', '0.06')))}
STAGES = ('patch', 'dmodel', 'mlp', 'blocks', 'wd')
fails = 0


def check(label, cond, detail=''):
    global fails
    fails += not cond
    print(f'{"PASS" if cond else "FAIL"} {label}{(": " + detail) if detail else ""}')


def tag(ds, cfg):
    return f'spkmx_lif_muxlin_iand_lr{LR[ds]}_L1' + ''.join(SUF[k] + cfg[k] for k in K if cfg[k] != D0[ds][k])


def defaults():
    rows, en = {}, {}
    for l in open(os.path.join(RES, 'flif-v1-ett-spkmux-tune', 'raw.txt')):
        if not l.startswith('RESULT'):
            continue
        d = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
        if d['cond'] == 'spkmx_lif_muxlin_iand' and d['ds'] in LR and d['tag'] == tag(d['ds'], D0[d['ds']]) and d['pl'] in ('96', '720') \
                and d['seed'] in ('42', '7', '2026'):
            rows[(d['ds'], d['pl'], d['seed'])] = l.rstrip('\n')
    for e in csv.DictReader(open(os.path.join(RES, 'flif-v1-ett-spkmux-tune', 'energy_recount.csv'))):
        if e['ds'] in LR and e['tag'] == tag(e['ds'], D0[e['ds']]) and e['pl'] in ('96', '720') and e['seed'] in ('42', '7', '2026'):
            en[(e['ds'], e['pl'], e['seed'])] = e
    assert len(rows) == 12 and len(en) == 12, (len(rows), len(en))
    return rows, en


ROWS, EN = defaults()


def fake(ds, pl, seed, cfg, f):
    t = ROWS[(ds, pl, seed)].split()
    d = dict(x.split('=', 1) for x in t[1:] if '=' in x)
    out = []
    for x in t:
        k = x.split('=', 1)[0]
        if k == 'tag':
            x = f'tag={tag(ds, cfg)}'
        elif k == 'layers':
            x = 'layers=1 ' + ' '.join(f'{FLD[q]}={cfg[q]}' for q in K)
        elif k in ('mse', 'mae'):
            x = f'{k}={float(d[k]) * f:.6f}'
        elif k == 'best_epoch_val_epochs':
            a, v, b = d[k].split(':'); x = f'{k}={a}:{float(v) * f:.6f}:{b}'
        elif k == 'val_loss_best':
            x = f'{k}={float(d[k]) * f:.6f}'
        elif k == 'sha':
            x = 'sha=dry'
        out.append(x)
    return ' '.join(out)


def parse_line(l):
    env = dict(t.split('=', 1) for t in l.split() if re.match(r'^[A-Z_]+=', t))
    ds, pl, seed = l.split()[-3:]
    return ds, pl, seed, dict(D0[ds], **{k: env[k] for k in K if k in env})


def run(D, args, sh=False):
    """-> (rc, stdout + stderr); for the shell queue maker (sh) the conda library path is not set and stdout comes LAST, so the last
    line is the queue file it printed."""
    env = dict(os.environ, ET_DRY=D, TMPDIR=D)
    if not sh:
        env['LD_LIBRARY_PATH'] = LIB
    else:
        env.pop('LD_LIBRARY_PATH', None)
    cmd = ['bash'] + args if sh else [PY, '-B'] + args
    p = subprocess.run(cmd, cwd=C, env=env, capture_output=True, text=True)
    return p.returncode, (p.stderr + p.stdout) if sh else (p.stdout + p.stderr)


def factor(F, ds, cfg):
    f = 1.0
    for k in K:
        if cfg[k] != D0[ds][k]:
            f *= F.get(ds, {}).get(k, {}).get(cfg[k], 1.0 if F.get('identity') else 1.01)
    return f


def chain(name, F, stages=STAGES, do_final=True):
    D = os.path.join(OUT, f'dry_{name}')
    shutil.rmtree(D, ignore_errors=True)
    S = os.path.join(D, 'flif-v1-ett-spkmux-ettune'); os.makedirs(S)
    raw, sel = [], {}
    for st in stages:
        rc, out = run(D, [os.path.join(C, 'scripts', 'make_queue_ettune.sh'), st], sh=True)
        if rc:
            check(f'{name} {st}: make_queue', False, out.strip()[-200:]); return D, None
        qf = out.strip().splitlines()[-1].rsplit(':', 1)[0]
        lines = [l for l in open(qf).read().splitlines() if l.strip()]
        for l in lines:
            ds, pl, seed, cfg = parse_line(l)
            raw.append(fake(ds, pl, seed, cfg, factor(F, ds, cfg)))
        open(os.path.join(S, 'raw.txt'), 'w').write('\n'.join(raw) + '\n')
        rc, out = run(D, ['summarize_ettune.py', '--stage', st])
        if rc:
            check(f'{name} {st}: summarizer exit 0', False, out.strip().splitlines()[-1]); return D, None
        rc2, out2 = run(D, ['recheck_ettune.py', '--stage', st])
        check(f'{name} {st}: {len(lines)} queue lines, summarizer ok, recheck agrees', rc2 == 0 and 'ALL CHECKS PASSED' in out2, out2.strip().splitlines()[-1])
        sel[st] = json.load(open(os.path.join(S, f'selected_{st}.json')))
    if not do_final:
        return D, sel
    with open(os.path.join(S, 'energy_recount.csv'), 'w', newline='') as fo:
        w = None
        for l in raw:
            d = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
            e = dict(EN[(d['ds'], d['pl'], d['seed'])], tag=d['tag'])
            f = float(d['mse']) / float(dict(t.split('=', 1) for t in ROWS[(d['ds'], d['pl'], d['seed'])].split()[1:] if '=' in t)['mse'])
            e['mse_logged'] = d['mse']; e['mse_recomputed'] = f'{float(EN[(d["ds"], d["pl"], d["seed"])]["mse_recomputed"]) * f:.8f}'
            if w is None:
                w = csv.DictWriter(fo, fieldnames=list(e)); w.writeheader()
            w.writerow(e)
    rc, out = run(D, ['summarize_ettune.py', '--stage', 'final'])
    check(f'{name} final: summarizer exit 0', rc == 0, '' if rc == 0 else out.strip().splitlines()[-1])
    if rc:
        return D, sel
    rc2, out2 = run(D, ['recheck_ettune.py', '--stage', 'final'])
    check(f'{name} final: recheck agrees', rc2 == 0 and 'ALL CHECKS PASSED' in out2, out2.strip().splitlines()[-1])
    sel['final'] = json.load(open(os.path.join(S, 'final.json')))
    return D, sel


# A identity
_, A = chain('A', {'identity': True})
if A:
    check('A: every stage keeps the current value', all(A[s]['selected'] == A[s]['current'] for s in STAGES))
    fa = A.get('final')
    if fa:
        check('A: final config = default', fa['config'] == D0)
        c = fa['comparisons']['tuned vs untuned LIF SpkMixer']
        check('A: tuned vs untuned R 0, 0/12', abs(c['R']) < 1e-12 and c['lo'] == 0, f'{c["R"]} {c["lo"]}')
        check('A: tuned 4-cell MSE / MAE = 0.324497 / 0.356692 (mxett 23:51), NOT PASSED',
              abs(fa['tuned']['mse'] - 0.32449658333) < 1e-9 and abs(fa['tuned']['mae'] - 0.35669175) < 1e-9 and not fa['pass']['tuned LIF SpkMixer']['pass'],
              f"{fa['tuned']['mse']:.9f} / {fa['tuned']['mae']:.9f}")
        c2 = fa['comparisons']['tuned LIF SpkMixer vs SF-de']
        check('A: tuned vs SF-de = (report) LIF vs SF-de of 23:51 (R +0.30, 5/12, energy 0.98)', f'{c2["R"]:+.2f}' == '+0.30' and c2['lo'] == 5
              and f'{c2["Es"]:.2f}' == '0.98', f'{c2["R"]:+.4f} {c2["lo"]} {c2["Es"]:.4f}')
# B chosen path
FB = {'ETTh2': {'PATCH': {'16': 0.99}, 'EMB': {'128': 0.98}, 'BLOCKS': {'2': 0.995}, 'WD': {'0.1': 0.997}},
      'ETTm2': {'PATCH': {'4': 0.99}, 'MLPR': {'2': 0.98}, 'WD': {'0': 0.999}}}
DB, B = chain('B', FB)
WANTB = {'ETTh2': dict(zip(K, ('16', '128', '0.5', '2', '0.1'))), 'ETTm2': dict(zip(K, ('4', '64', '2', '1', '0')))}
if B:
    got = {ds: {k: B[s]['selected'][ds] for s, k in zip(STAGES, K)} for ds in LR}
    check('B: selections follow the factors', got == WANTB, str(got))
    q = open(sorted(f for f in [os.path.join(DB, 'queues', x) for x in os.listdir(os.path.join(DB, 'queues'))] if '-wd-' in f)[0]).read()
    check('B: the wd queue carries the earlier winners (ETTh2 PATCH=16 EMB=128 BLOCKS=2; ETTm2 PATCH=4 MLPR=2)',
          'PATCH=16 EMB=128 BLOCKS=2 WD=' in q and 'PATCH=4 MLPR=2 WD=' in q and 'MLPR=0.5' not in q)
    fb = B.get('final')
    if fb:
        check('B: final config', fb['config'] == WANTB, str(fb['config']))
        exp = 0
        for ds in LR:
            f = factor(FB, ds, WANTB[ds])
            for pl in ('96', '720'):
                exp += sum(float(dict(t.split('=', 1) for t in ROWS[(ds, pl, s)].split()[1:] if '=' in t)['mse']) * f for s in ('42', '7', '2026')) / 3
        exp /= 4
        check('B: final tuned 4-cell MSE = factor-weighted default rows (1e-5, 6-decimal rounding)', abs(fb['tuned']['mse'] - exp) < 1e-5,
              f"{fb['tuned']['mse']:.6f} vs {exp:.6f}")
# C tie between two non-current values
_, Cc = chain('C', {'ETTh2': {'PATCH': {'4': 0.99, '16': 0.99}}}, stages=('patch',), do_final=False)
if Cc:
    check('C: ETTh2 PATCH 4 and 16 tie -> 4 (earlier candidate); ETTm2 keeps 8', Cc['patch']['selected'] == {'ETTh2': '4', 'ETTm2': '8'},
          str(Cc['patch']['selected']))


# D fail closed
def dcase(name, mutate, why, stage='patch', rc_want=2):
    D = os.path.join(OUT, f'dry_{name}')
    shutil.rmtree(D, ignore_errors=True)
    S = os.path.join(D, 'flif-v1-ett-spkmux-ettune'); os.makedirs(S)
    rc, out = run(D, [os.path.join(C, 'scripts', 'make_queue_ettune.sh'), 'patch'], sh=True)
    lines = [l for l in open(out.strip().splitlines()[-1].rsplit(':', 1)[0]).read().splitlines() if l.strip()]
    raw = [fake(*parse_line(l), 1.0) for l in lines]
    raw = mutate(raw, S)
    open(os.path.join(S, 'raw.txt'), 'w').write('\n'.join(raw) + '\n')
    rc, out = run(D, ['summarize_ettune.py', '--stage', stage])
    p = os.path.join(S, 'selected_patch.json')
    kept = (json.load(open(p))['selected'] == {'ETTh2': '16', 'ETTm2': '8'}) if name == 'D3' else not os.path.exists(p)
    check(f'{name}: exit {rc_want} for "{why}", selection untouched', rc == rc_want and why in out and kept, out.strip().splitlines()[-1][:160])


dcase('D1', lambda r, S: r + ['SKIP(rc=1, no result) ds=ETTh2 pl=96 cond=spkmx_lif_muxlin_iand seed=42 out=-'], 'SKIP line')
dcase('D2', lambda r, S: r[:-1], 'missing runs')


def d3(r, S):
    json.dump({'selected': {'ETTh2': '16', 'ETTm2': '8'}}, open(os.path.join(S, 'selected_patch.json'), 'w')); return r


dcase('D3', d3, 'not overwritten')
dcase('D4', lambda r, S: r + [fake('ETTh2', '96', '42', dict(D0['ETTh2'], EMB='128'), 1.0)], 'no stage asked for')
dcase('D5', lambda r, S: r[:1] + [r[1].replace('patch=4', 'patch=16', 1)] + r[2:], 'tag')
D = os.path.join(OUT, 'dry_D6'); shutil.rmtree(D, ignore_errors=True); os.makedirs(os.path.join(D, 'flif-v1-ett-spkmux-ettune'))
rc, out = run(D, [os.path.join(C, 'scripts', 'make_queue_ettune.sh'), 'dmodel'], sh=True)
check('D6: make_queue dmodel without selected_patch.json -> exit 1, STOP', rc == 1 and 'STOP' in out and 'selected_patch.json missing' in out, out.strip()[-120:])
# D7: chain A again, one recount row removed before final
DA = os.path.join(OUT, 'dry_A'); S = os.path.join(DA, 'flif-v1-ett-spkmux-ettune')
D7 = os.path.join(OUT, 'dry_D7'); shutil.rmtree(D7, ignore_errors=True); shutil.copytree(DA, D7)
S7 = os.path.join(D7, 'flif-v1-ett-spkmux-ettune')
os.remove(os.path.join(S7, 'final.json')); os.remove(os.path.join(S7, 'summary_final.md'))
L = open(os.path.join(S7, 'energy_recount.csv')).read().splitlines()
open(os.path.join(S7, 'energy_recount.csv'), 'w').write('\n'.join(L[:-1]) + '\n')
rc, out = run(D7, ['summarize_ettune.py', '--stage', 'final'])
check('D7: final with a recount row missing -> exit 2, no final.json', rc == 2 and 'no energy recount row' in out and not os.path.exists(os.path.join(S7, 'final.json')),
      out.strip().splitlines()[-1][:160])
print('ALL DRY CHECKS PASSED' if fails == 0 else f'{fails} FAILED')
sys.exit(0 if fails == 0 else 1)
