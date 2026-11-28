"""Dry run of summarize_memoffi.py and recheck_memoffi.py on planted suites (2026-11-28, before training).
  m1: the ADD memoryless rows renamed to the IAND memoryless tag (+ their energy recount, order_muxoff.json with the tag replaced)
      -> U3 = 0; U1 = ADD memoryless - IAND base per seed; U2 = IAND shuffled - ADD memoryless (= T6 of summary_tshufi with the sign
      reversed); decomposition adds up; recheck ALL CHECKS PASSED, and FAIL on a tampered verdict
  m2 stop cases (exit 2, nothing written): SKIP line, missing seed, mixed commits, recount != logged, nan, order json drop 0.1 %p,
      order json unshuffled reading != logged
usage (from cinc_mixer_v1/): python -B results/check-iandmem-20261128/dry_memoffi.py --tmp <empty folder>
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = os.path.dirname(os.path.abspath(__file__))
RS = os.path.join(C, 'results')
ap = argparse.ArgumentParser(); ap.add_argument('--tmp', required=True); a = ap.parse_args()
os.makedirs(a.tmp, exist_ok=True)
TAG = 'spkmx_lif_muxlin_iand_lr0.01_L2_tmoff'
bad = 0


def ok(c, m):
    global bad
    bad += not c
    print(('PASS ' if c else 'FAIL ') + m, flush=True)


def kv(l):
    return dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)


def setk(l, **ch):
    return ' '.join((f'{t.split("=", 1)[0]}={ch[t.split("=", 1)[0]]}' if '=' in t and t.split('=', 1)[0] in ch else t) for t in l.split())


def suite(name, L, mut=None):
    d = os.path.join(a.tmp, name); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    open(os.path.join(d, 'raw.txt'), 'w').write('\n'.join(L) + '\n')
    rows = list(csv.DictReader(open(os.path.join(RS, 'flif-v1-mitbih-memoff', 'energy_recount.csv'))))
    for r in rows:
        r['tag'] = TAG; r['cond'] = 'spkmx_lif_muxlin_iand'
    if mut:
        mut(rows)
    w = csv.DictWriter(open(os.path.join(d, 'energy_recount.csv'), 'w', newline=''), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return d


def run(script, *args):
    p = subprocess.run([sys.executable, '-B', os.path.join(H, script), *args], cwd=C, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


src = [l.strip() for l in open(os.path.join(RS, 'flif-v1-mitbih-memoff', 'raw.txt')) if l.startswith('RESULT')]
fake = [setk(l, cond='spkmx_lif_muxlin_iand', tag=TAG, sha='dryMEM00') for l in src]
oj = json.load(open(os.path.join(RS, 'check-mitorder-20261128', 'order_muxoff.json'))); oj['tag'] = TAG
of = os.path.join(a.tmp, 'order.json'); json.dump(oj, open(of, 'w'))
nd = suite('m1', fake); out = os.path.join(a.tmp, 'out_m1'); shutil.rmtree(out, ignore_errors=True)
rc, txt = run('summarize_memoffi.py', '--new', nd, '--order', of, '--out_dir', out)
ok(rc == 0, f'm1 planted run exit {rc}' + ('' if rc == 0 else ' ' + txt[-300:]))
S = json.load(open(os.path.join(out, 'summary_memoffi.json')))
c = {x['kind'].split(' ')[0]: x for x in S['comparisons']}
T6 = {x['kind'].split(' ')[0]: x for x in json.load(open(os.path.join(RS, 'check-iand-20261128', 'summary_tshufi.json')))['comparisons']}['T6']
base = {kv(l)['seed']: float(kv(l)['f1']) for l in open(os.path.join(RS, 'flif-v1-mitbih-res', 'raw.txt')) if l.startswith('RESULT') and kv(l)['tag'] == 'spkmx_lif_muxlin_iand_lr0.01_L2'}
addoff = {kv(l)['seed']: float(kv(l)['f1']) for l in src}
ok(all(abs(t) < 1e-12 for t in c['U3']['d_pp']), f'm1 U3 = 0 ({c["U3"]["mean_pp"]:+.3g})')
ok(all(abs(x - 100 * (addoff[s] - base[s])) < 1e-9 for x, s in zip(c['U1']['d_pp'], ('42', '7', '2026'))), f'm1 U1 = ADD memoryless - IAND base ({c["U1"]["mean_pp"]:+.3f})')
ok(all(abs(x - y) < 1e-9 for x, y in zip(c['U2']['d_pp'], T6['d_pp'])), f'm1 U2 = T6 of summary_tshufi ({c["U2"]["mean_pp"]:+.3f} = {T6["mean_pp"]:+.3f})')
ok(all(abs(t - (x + y)) < 1e-9 for t, x, y in zip(S['memoryless_drop_pp'], S['order_share_pp'], S['interaction_share_pp'])), 'm1 decomposition adds up per seed')
rd = os.path.join(a.tmp, 'out_m1')
rc, txt = run('recheck_memoffi.py', '--new', nd, '--dir', rd, '--order', of)
ok(rc == 0 and 'ALL CHECKS PASSED' in txt, f'm1 recheck: {txt.strip().splitlines()[-1]}')
S['comparisons'][0]['verdict'] = 'A better'; json.dump(S, open(os.path.join(rd, 'summary_memoffi.json'), 'w'))
rc, txt = run('recheck_memoffi.py', '--new', nd, '--dir', rd, '--order', of)
ok('FAIL' in txt and 'ALL CHECKS PASSED' not in txt, f'm1 recheck on a tampered verdict: {txt.strip().splitlines()[-1]}')
ob = dict(oj, runs=[dict(x, mixer=dict(x['mixer'], drop_f1_pp=0.1)) for x in oj['runs']]); obf = os.path.join(a.tmp, 'order_drop.json'); json.dump(ob, open(obf, 'w'))
ou = dict(oj, runs=[dict(x, f1=x['f1'] + 1) for x in oj['runs']]); ouf = os.path.join(a.tmp, 'order_f1.json'); json.dump(ou, open(ouf, 'w'))
cases = {'SKIP line': (fake + ['SKIP(rc=1) x'], None, of), 'missing seed': (fake[:2], None, of), 'mixed commits': ([setk(fake[0], sha='x')] + fake[1:], None, of),
         'recount != logged': (fake, lambda rows: rows[0].__setitem__('f1_recomputed', '0.1'), of), 'nan': ([setk(fake[0], f1='nan')] + fake[1:], None, of),
         'order drop 0.1 %p': (fake, None, obf), 'order unshuffled != logged': (fake, None, ouf)}
for name, (L, mut, o_) in cases.items():
    od = os.path.join(a.tmp, 'out_stop'); shutil.rmtree(od, ignore_errors=True)
    rc, txt = run('summarize_memoffi.py', '--new', suite('m2', L, mut), '--order', o_, '--out_dir', od)
    ok(rc == 2 and not os.path.exists(od), f'm2 stop "{name}": exit {rc}, ' + txt.strip().splitlines()[-1][-100:])
print('ALL DRY CHECKS PASSED' if bad == 0 else f'{bad} FAIL')
sys.exit(0 if bad == 0 else 1)
