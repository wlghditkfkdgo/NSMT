"""Dry run of the whole AD tuning flow on planted rows (2026-11-29, before training): adtune.py --queue / --select for the seven stages,
then summarize_adtune2.py and recheck_adtune2.py, in a temporary ADT_ROOT holding copies of the real reference suites.
Planted AUC-PR of a new row = the real start (G5) row of the same dataset and seed + a bonus fixed per (dataset, setting value), so the
selected path is known in advance:
  MSL  norm window (kept), epochs 3, width 32, depth kept, patch (20, 1), lr kept, res add
  SMAP norm mean, epochs kept, width 128 ties the current exactly -> kept 64 (tie rule), depth kept, patch kept, lr 0.004, res iand
  PSM  norm none, epochs 1, width kept, depth (1, 2) and (2, 1) tie -> (1, 2) (earlier in the list), patch kept, lr 0.001, res kept iand
Checks: queue sizes (stage norm 18 runs; later stages only the missing runs), the selections above, the final tags, the summary runs and
the independent recheck passes; stop cases (exit 2): SKIP line, a missing seed, a selection file with a different selection, a duplicate
row across suites; the recheck fails on a tampered selection.
usage (from cinc_mixer_v1/): python -B results/check-adtune2-20261129/dry_adtune2.py --tmp <empty folder>
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.dirname(os.path.dirname(HERE))
RS = os.path.join(C, 'results')
sys.path.insert(0, C)
ap = argparse.ArgumentParser(); ap.add_argument('--tmp', required=True); a = ap.parse_args()
R = os.path.join(a.tmp, 'results')
shutil.rmtree(R, ignore_errors=True)
for su in ('flif-v1-ad-energy', 'flif-v1-ad-iand', 'flif-v1-ad-sfde'):
    os.makedirs(os.path.join(R, su))
    for f in ('raw.txt', 'selected_lr.json'):
        if os.path.exists(os.path.join(RS, su, f)):
            shutil.copy(os.path.join(RS, su, f), os.path.join(R, su, f))
os.makedirs(os.path.join(R, 'flif-v1-ad-tune2'))
ENV = dict(os.environ, ADT_ROOT=R, ADT_OUT=os.path.join(a.tmp, 'out'))
os.environ['ADT_ROOT'] = R
import adtune as T                                                       # noqa: E402
bad = 0


def ok(c, m):
    global bad
    bad += not c
    print(('PASS ' if c else 'FAIL ') + m, flush=True)


def run(*args, script='adtune.py'):
    p = subprocess.run([sys.executable, '-B', os.path.join(C, script) if not script.startswith('/') else script, *args], cwd=C, env=ENV, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


BON = {'MSL': {('norm', 'mean'): -0.002, ('norm', 'none'): -0.003, ('ep', 3): 0.004, ('ep', 1): 0.001, ('ep', 30): -0.001, ('emb', 32): 0.002, ('emb', 128): 0.001,
               ('pt', 20): 0.001, ('pt', 4): 0.0005, ('lr', '0.004'): -0.001, ('res', 'add'): 0.0005, ('res', 'iand'): -0.0005},
       'SMAP': {('norm', 'mean'): 0.01, ('norm', 'none'): 0.002, ('ep', 30): -0.002, ('emb', 128): 0.0, ('emb', 32): -0.001, ('bl', 2): -0.001, ('ml', 2): -0.001,
                ('lr', '0.004'): 0.001, ('res', 'iand'): 0.0007, ('res', 'add'): -0.0003},
       'PSM': {('norm', 'none'): 0.05, ('norm', 'mean'): 0.01, ('ep', 1): 0.003, ('ep', 3): 0.001, ('bl', 2): 0.002, ('ml', 2): 0.002, ('pt', 20): -0.002, ('pt', 4): -0.001,
               ('lr', '0.001'): 0.001, ('res', 'or'): -0.001, ('res', 'add'): -0.0002}}
WANT = {'norm': {'MSL': ['window'], 'SMAP': ['mean'], 'PSM': ['none']}, 'epochs': {'MSL': [3], 'SMAP': [10], 'PSM': [1]},
        'width': {'MSL': [32], 'SMAP': [64], 'PSM': [64]}, 'depth': {'MSL': [1, 1], 'SMAP': [1, 1], 'PSM': [1, 2]},
        'patch': {'MSL': [20, '1'], 'SMAP': [8, '1'], 'PSM': [8, '1']}, 'lr': {'MSL': ['0.002'], 'SMAP': ['0.004'], 'PSM': ['0.001']},
        'res': {'MSL': ['add'], 'SMAP': ['iand'], 'PSM': ['iand']}}
base = {}
for l in open(os.path.join(R, 'flif-v1-ad-energy', 'raw.txt')):
    if l.startswith('RESULT'):
        r = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
        if r['holdout'] == '0' and r['tag'] == T.tag(T.BASE[r['ds']]):
            base[(r['ds'], r['seed'])] = l.strip()
assert len(base) == 9, len(base)


def setk(l, **ch):
    return ' '.join((f'{t.split("=", 1)[0]}={ch[t.split("=", 1)[0]]}' if '=' in t and t.split('=', 1)[0] in ch else t) for t in l.split())


def planted(qline):
    env = dict(x.split('=', 1) for x in qline.split() if '=' in x and x.split('=', 1)[0].isupper())
    ds, seed = qline.split()[-2], qline.split()[-1]
    c = dict(res=env['COND'].split('_')[-1], lr=env['LR'], ml=int(env['MLAYERS']), bl=int(env['BLOCKS']), emb=int(env['EMB']), pt=int(env['PATCH']),
             mlp=env['MLP'], norm=env['ADNORM'], ep=int(env['EPOCHS']))
    b = sum(v for (k, val), v in BON[ds].items() if c[k] == val)
    tmpl = base[(ds, seed)]
    au = float(dict(t.split('=', 1) for t in tmpl.split()[1:] if '=' in t)['aucpr']) + b
    return setk(tmpl, cond=env['COND'], tag=T.tag(c), lr=c['lr'], layers=str(c['ml']), aucpr=f'{au:.6f}', sha='dryTUNE0')


raw = os.path.join(R, 'flif-v1-ad-tune2', 'raw.txt')
open(raw, 'w').close()
sizes = {}
for st in T.NAMES:
    q = os.path.join(a.tmp, f'q_{st}.txt')
    rc, txt = run('--stage', st, '--queue', q)
    L = [l.strip() for l in open(q) if l.strip()] if rc == 0 else []
    sizes[st] = len(L)
    with open(raw, 'a') as f:
        for l in L:
            f.write(planted(l) + '\n')
    rc, txt = run('--stage', st, '--select')
    sel = json.load(open(os.path.join(R, 'flif-v1-ad-tune2', f'selected_{st}.json')))['selected'] if rc == 0 else None
    ok(rc == 0 and sel == WANT[st], f'stage {st}: {len(L)} queued runs, selected {sel} (want {WANT[st]})')
ok(sizes['norm'] == 18, f'stage norm queue = 18 runs ({sizes["norm"]}); later stages: {sizes}')
rc, txt = run('--stage', 'lr', '--select')
ok(rc == 0, 're-selecting an existing stage with the same result is allowed')
rc, txt = run(script=os.path.join(HERE, 'summarize_adtune2.py'))
ok(rc == 0, 'summarize_adtune2.py on the planted flow' + ('' if rc == 0 else ' ' + txt[-400:]))
S = json.load(open(os.path.join(a.tmp, 'out', 'summary_adtune2.json')))
want_tags = {'MSL': 'spkmx_lif_muxlin_add_lr0.002_L1_e32_p20_ep3', 'SMAP': 'spkmx_lif_muxlin_iand_lr0.004_L1_nmmean', 'PSM': 'spkmx_lif_muxlin_iand_lr0.001_L2_nmnone_ep1'}
ok(all(S['final_config'][ds]['tag'] == want_tags[ds] for ds in want_tags), f"final tags {[S['final_config'][ds]['tag'] for ds in ('MSL', 'SMAP', 'PSM')]}")
c = {(x['metric'], x['B']): x for x in S['comparisons']}
ok(c[('aucpr', 'SpkMixer start (11-19 G5)')]['d_mean'] > 0, f"tuned - start AUC-PR mean {c[('aucpr', 'SpkMixer start (11-19 G5)')]['d_mean']:+.4f} > 0 (planted bonuses)")
rc, txt = run(script=os.path.join(HERE, 'recheck_adtune2.py'))
ok(rc == 0 and 'ALL CHECKS PASSED' in txt, 'recheck on the planted flow: ' + txt.strip().splitlines()[-1])
p = os.path.join(R, 'flif-v1-ad-tune2', 'selected_width.json'); js = json.load(open(p)); keep = json.dumps(js)
js['selected']['SMAP'] = [128]; json.dump(js, open(p, 'w'))
rc, txt = run(script=os.path.join(HERE, 'recheck_adtune2.py'))
ok('FAIL' in txt and 'ALL CHECKS PASSED' not in txt, 'recheck on a tampered selection: ' + txt.strip().splitlines()[-1])
rc, txt = run('--stage', 'width', '--select')
ok(rc == 2 and 'not overwritten' in txt, 'selection file with a different selection is not overwritten: exit %d' % rc)
open(p, 'w').write(keep)
L = open(raw).read().splitlines()
open(raw, 'w').write('\n'.join(L + ['SKIP(rc=1, no result) ds=MSL']) + '\n')
rc, txt = run('--stage', 'res', '--select'); ok(rc == 2 and 'SKIP' in txt, f'SKIP line stops: exit {rc}')
open(raw, 'w').write('\n'.join([l for l in L if not ('ds=PSM' in l and 'seed=7 ' in l and 'nmnone_ep1' in l and 'L2' in l and 'lr0.001' in l)]) + '\n')
rc, txt = run('--stage', 'res', '--select'); ok(rc == 2 and 'missing seed' in txt, f'missing seed stops: exit {rc}')
open(raw, 'w').write('\n'.join(L + [base[('MSL', '42')]]) + '\n')
rc, txt = run('--stage', 'res', '--select'); ok(rc == 2 and 'duplicate row' in txt, f'duplicate row across suites stops: exit {rc}')
open(raw, 'w').write('\n'.join(L) + '\n')
print('ALL DRY CHECKS PASSED' if bad == 0 else f'{bad} FAIL')
sys.exit(0 if bad == 0 else 1)
