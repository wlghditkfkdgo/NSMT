"""Checks of summarize_ade.py on dry suites (2026-11-19, exp/f-lif-pop-v3; PROJECT_LOG AD energy pre-registration). Fail closed (exit 1).
  d1 stage lr on the relabelled 11-03 lr-stage rows selects the 11-03 muxlin lr (MSL 0.002, SMAP 0.002, PSM 0.001)
  d2 stage res equals a direct recomputation from the raw validation numbers (res = argmin of OR / IAND / ADD, spike = argmin of OR / IAND,
     ties keep OR)
  d3 stage final: G1 = G3 = E1 of results/flif-v1-ad-sfde/summary.md (R +1.368 %, 6/9) with energy 1.1937 (syops) / 1.0499 (current, as the
     01:16 entry); G2 = G4 = R 0, energy 1.0000; G5 present
  d4 fail closed: a SKIP line, a missing run, a different selection over an existing selected_lr.json -> exit 2, no summary written
usage (from cinc_mixer_v1/): python -B results/check-ade-20261119/check_dry_ade.py <empty scratch folder>
"""
import json
import os
import re
import shutil
import subprocess
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
out = sys.argv[1]
BAD = []


def ok(c, m):
    print(('PASS ' if c else 'FAIL ') + m, flush=True)
    if not c:
        BAD.append(m)


def summ(folder, stage):
    env = dict(os.environ, ADE_DRY=folder)
    return subprocess.run([PY, '-B', os.path.join(C, 'summarize_ade.py'), '--stage', stage], cwd=C, env=env, capture_output=True, text=True)


subprocess.run([PY, '-B', os.path.join(HERE, 'make_dry_ade.py'), out], cwd=C, check=True)
S, F = os.path.join(out, 'sel'), os.path.join(out, 'final')
p = summ(S, 'lr'); ok(p.returncode == 0, 'd1 stage lr runs')
ok(json.load(open(os.path.join(S, 'selected_lr.json')))['selected'] == {'MSL': '0.002', 'SMAP': '0.002', 'PSM': '0.001'}, 'd1 lr = the 11-03 muxlin selection')
p = summ(S, 'res'); ok(p.returncode == 0, 'd2 stage res runs')
sel = json.load(open(os.path.join(S, 'selected_res.json')))['selected']
lr = json.load(open(os.path.join(S, 'selected_lr.json')))['selected']
v = {}
for l in open(os.path.join(S, 'raw.txt')):
    g = lambda k: re.search(rf' {k}=(\S+)', l).group(1)
    if float(g('lr')) == float(lr[g('ds')]):
        v.setdefault((g('ds'), g('cond').split('_')[3]), []).append(float(g('val_mse_best')))
exp = {'res': {}, 'spike': {}}
for ds in ('MSL', 'SMAP', 'PSM'):
    m = {r: sum(v[(ds, r)]) / 3 for r in ('or', 'iand', 'add')}
    assert all(len(v[(ds, r)]) == 3 for r in m)
    exp['res'][ds] = min(m, key=lambda r: (m[r], r != 'or'))
    exp['spike'][ds] = min(('or', 'iand'), key=lambda r: (m[r], r != 'or'))
ok(sel == exp, f'd2 res selection {sel} = direct recomputation {exp}')
p = summ(F, 'final'); ok(p.returncode == 0, 'd3 stage final runs')
t = open(os.path.join(F, 'summary.md')).read()
row = lambda q: next((l for l in t.splitlines() if l.startswith(f'| {q}')), '')
for q in ('G1 (B) M1(tuned) vs SF-de(tuned)', 'G3 (A) mean vs SF-de(tuned)'):
    ok('| +1.368 | 6/9 | no clear difference | 1.1937 | not met | 1.0499 | not met |' in row(q), f'd3 {q} = E1 ({row(q)})')
for q in ('G2 (B) M1(tuned) vs muxlin(tuned, 11-03)', 'G4 (A) mean vs muxlin(tuned, 11-03)'):
    ok('| +0.000 | 0/9 | no clear difference | 1.0000 |  | 1.0000 |  |' in row(q), f'd3 {q} = 0 ({row(q)})')
ok(row('G5') != '', 'd3 G5 present')
# d4 fail closed
for name, edit in (('SKIP line', lambda d: open(os.path.join(d, 'raw.txt'), 'a').write('SKIP(rc=1, no result) ds=MSL\n')),
                   ('missing run', lambda d: open(os.path.join(d, 'raw.txt'), 'w').writelines(open(os.path.join(S, 'raw.txt')).readlines()[1:]))):
    d = os.path.join(out, 'd4_' + name.split()[0]); os.makedirs(d)
    shutil.copy(os.path.join(S, 'raw.txt'), d); edit(d)
    p = summ(d, 'lr')
    ok(p.returncode == 2 and not os.path.exists(os.path.join(d, 'selection_lr.md')) and not os.path.exists(os.path.join(d, 'selected_lr.json')),
       f'd4 {name} -> exit 2, nothing written ({p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""})')
d = os.path.join(out, 'd4_overwrite'); os.makedirs(d); shutil.copy(os.path.join(S, 'raw.txt'), d)
json.dump({'selected': {'MSL': '0.0001', 'SMAP': '0.002', 'PSM': '0.001'}}, open(os.path.join(d, 'selected_lr.json'), 'w'))
p = summ(d, 'lr')
ok(p.returncode == 2 and json.load(open(os.path.join(d, 'selected_lr.json')))['selected']['MSL'] == '0.0001', 'd4 different selection -> exit 2, file kept')
print(f'{len(BAD)} FAILED' if BAD else 'all dry checks passed')
sys.exit(1 if BAD else 0)
