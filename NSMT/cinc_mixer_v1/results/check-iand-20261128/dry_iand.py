"""Dry run of summarize_adiand.py, summarize_tshufi.py and recheck_iand.py on planted suites (2026-11-28, before training).
Fake suites are built from logged rows in a temporary folder (--tmp), so the expected readings are known:
  d1 AD: the 11-19 G5 OR rows of MSL / SMAP renamed to the IAND tag -> J2 R = 0, 0/6, no clear difference, ΔAUC-PR 0, energy 1;
         J1 = J0 (= the 11-19 G5 reading: R +1.452 %, 8/9, energy syops + neuron 0.9880); reproduction "same"
  d2 AD: the same with MSL and SMAP AUC-PR x 0.9 -> J2 R -10 %, 'A worse', recommendation 'keep G5' (v1 of this check lowered MSL
         only and wanted 'A worse'; the rule correctly said 'no clear difference' (3 of 6 lower < 4), my expectation was wrong)
  d3 AD stop cases (exit 2, nothing written): SKIP line, 5 rows, duplicate seed, nan AUC-PR, mixed commits, energy_check fail, other tag
  d4 MIT-BIH: the ADD shuffled rows renamed to the IAND shuffled tag -> T5 = 0, T3 = S3 and T6 = S2 of summary_tshuf.json,
         T2 = IAND base - ADD base per seed (= R1 of the residual test with the sign of IAND - ADD)
  d5 MIT-BIH stop cases: SKIP line, missing seed, mixed commits, recount != logged, spread default != logged, order reading != logged, nan
  d6 recheck_iand.py on d1 + d4 -> ALL CHECKS PASSED; on a tampered summary -> FAIL
usage (from cinc_mixer_v1/): python -B results/check-iand-20261128/dry_iand.py --tmp <empty folder>
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
RS = os.path.join(C, 'results')
PY = sys.executable
ap = argparse.ArgumentParser()
ap.add_argument('--tmp', required=True)
a = ap.parse_args()
os.makedirs(a.tmp, exist_ok=True)
bad = 0


def ok(c, m):
    global bad
    bad += not c
    print(('PASS ' if c else 'FAIL ') + m, flush=True)


def lines(path):
    return [l.rstrip('\n') for l in open(os.path.join(path, 'raw.txt'))]


def kv(l):
    return dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)


def setk(l, **ch):
    out = []
    for t in l.split():
        k = t.split('=', 1)[0]
        out.append(f'{k}={ch[k]}' if '=' in t and k in ch else t)
    return ' '.join(out)


def write(d, L):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    open(os.path.join(d, 'raw.txt'), 'w').write('\n'.join(L) + '\n')
    return d


def run(script, *args):
    p = subprocess.run([PY, '-B', os.path.join(HERE, script), *args], cwd=C, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


# ---- AD
E = [l for l in lines(os.path.join(RS, 'flif-v1-ad-energy')) if l.startswith('RESULT')]
g5or = [l for l in E if kv(l)['ds'] in ('MSL', 'SMAP') and kv(l)['tag'] == 'spkmx_lif_muxlin_or_lr0.002_L1' and kv(l)['holdout'] == '0']
psm42 = [l for l in E if kv(l)['ds'] == 'PSM' and kv(l)['tag'] == 'spkmx_lif_muxlin_iand_lr0.002_L1' and kv(l)['holdout'] == '0' and kv(l)['seed'] == '42']
assert len(g5or) == 6 and len(psm42) == 1
fake = [setk(l, cond='spkmx_lif_muxlin_iand', tag='spkmx_lif_muxlin_iand_lr0.002_L1', sha='dryAD000') for l in g5or]
rep = write(os.path.join(a.tmp, 'ad_repro'), psm42)
new = write(os.path.join(a.tmp, 'ad_d1'), fake)
out1 = os.path.join(a.tmp, 'out_d1')
rc, txt = run('summarize_adiand.py', '--new', new, '--repro', rep, '--out_dir', out1)
ok(rc == 0, f'd1 AD planted run exit {rc}')
S = json.load(open(os.path.join(out1, 'summary_adiand.json')))
cj = {c['kind']: c for c in S['comparisons']}
J0, J1, J2 = cj['J0 (report)'], cj['J1'], cj['J2']
ok(abs(J2['R']) < 1e-12 and J2['hi'] == 0 and J2['n'] == 6 and J2['verdict'] == 'no clear difference' and abs(J2['d_aucpr']) < 1e-15
   and abs(J2['energy_syn'] - 1) < 1e-12, f'd1 J2 = 0: R {J2["R"]:+.3g}, {J2["hi"]}/{J2["n"]}, {J2["verdict"]}, energy {J2["energy_syn"]:.6f}')
ok(all(abs(J1[k] - J0[k]) < 1e-12 for k in ('R', 'd_aucpr', 'energy_syn', 'energy_total')) and J1['hi'] == J0['hi'] and J1['verdict'] == J0['verdict'],
   f'd1 J1 = J0: R {J1["R"]:+.3f}, {J1["hi"]}/9, {J1["verdict"]}, ΔAUC-PR {J1["d_aucpr"]:+.4f}, energy syn {J1["energy_syn"]:.4f}, + neuron {J1["energy_total"]:.4f}')
ok(abs(J0['R'] - 1.452) < 5e-4 and J0['hi'] == 8 and abs(J0['energy_total'] - 0.9880) < 5e-5,
   f'd1 J0 = 11-19 G5 reading (R +1.452 %, 8/9, 0.9880): {J0["R"]:+.4f}, {J0["hi"]}/9, {J0["energy_total"]:.5f}')
ok(S['reproduction']['ok'] is True, 'd1 reproduction reading "same" on the 11-19 row itself')
ok(S['recommendation'] == ('IAND for AD' if J1['adoption_bar'] else 'keep G5'), f'd1 recommendation {S["recommendation"]} (J1 adoption bar {J1["adoption_bar"]})')
fake2 = [setk(l, aucpr=f'{0.9 * float(kv(l)["aucpr"]):.6f}') for l in fake]
rc, _ = run('summarize_adiand.py', '--new', write(os.path.join(a.tmp, 'ad_d2'), fake2), '--repro', rep, '--out_dir', os.path.join(a.tmp, 'out_d2'))
S2 = json.load(open(os.path.join(a.tmp, 'out_d2', 'summary_adiand.json')))
j2 = {c['kind']: c for c in S2['comparisons']}['J2']
ok(rc == 0 and j2['verdict'] == 'A worse' and abs(j2['R'] + 10) < 1e-3 and j2['hi'] == 0 and S2['recommendation'] == 'keep G5',
   f'd2 MSL, SMAP x 0.9: J2 R {j2["R"]:+.3f} % (want -10.000), {j2["verdict"]}, recommendation {S2["recommendation"]}')
cases = {'SKIP line': fake + ['SKIP(rc=1, no result) ds=MSL'], '5 rows': fake[:5], 'duplicate seed': fake + [fake[0]],
         'nan AUC-PR': [setk(fake[0], aucpr='nan')] + fake[1:], 'mixed commits': [setk(fake[0], sha='other')] + fake[1:],
         'energy_check fail': [setk(fake[0], energy_check='fail')] + fake[1:], 'other tag': [setk(fake[0], tag='spkmx_lif_muxlin_or_lr0.002_L1')] + fake[1:]}
for name, L in cases.items():
    od = os.path.join(a.tmp, 'out_stop')
    shutil.rmtree(od, ignore_errors=True)
    rc, txt = run('summarize_adiand.py', '--new', write(os.path.join(a.tmp, 'ad_stop'), L), '--repro', rep, '--out_dir', od)
    ok(rc == 2 and not os.path.exists(od), f'd3 AD stop "{name}": exit {rc}, ' + txt.strip().splitlines()[-1][:110])

# ---- MIT-BIH
T = [l for l in lines(os.path.join(RS, 'flif-v1-mitbih-tshuf')) if l.startswith('RESULT')]
TAGI = 'spkmx_lif_muxlin_iand_lr0.01_L2_tshuf'
fm = [setk(l, cond='spkmx_lif_muxlin_iand', tag=TAGI, sha='dryMB000') for l in T]


def mb_suite(name, L, rec_mut=None):
    d = write(os.path.join(a.tmp, name), L)
    rows = list(csv.DictReader(open(os.path.join(RS, 'flif-v1-mitbih-tshuf', 'energy_recount.csv'))))
    for r in rows:
        r['tag'] = TAGI; r['cond'] = 'spkmx_lif_muxlin_iand'
    if rec_mut:
        rec_mut(rows)
    w = csv.DictWriter(open(os.path.join(d, 'energy_recount.csv'), 'w', newline=''), fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
    return d


sp = json.load(open(os.path.join(RS, 'check-tshuf-20261128', 'tshuf_spread.json')))
sp['tag'] = TAGI
spf = os.path.join(a.tmp, 'spread.json'); json.dump(sp, open(spf, 'w'))
om = json.load(open(os.path.join(RS, 'check-mitorder-20261128', 'order_mux.json')))
base = {kv(l)['seed']: float(kv(l)['f1']) for l in lines(os.path.join(RS, 'flif-v1-mitbih-res')) if l.startswith('RESULT') and kv(l)['tag'] == 'spkmx_lif_muxlin_iand_lr0.01_L2'}
oi = dict(om, tag='spkmx_lif_muxlin_iand_lr0.01_L2', runs=[dict(x, f1=100 * base[x['seed']]) for x in om['runs']])
orf = os.path.join(a.tmp, 'order.json'); json.dump(oi, open(orf, 'w'))
out4 = os.path.join(a.tmp, 'out_d4')
nm = mb_suite('mb_d4', fm)
rc, txt = run('summarize_tshufi.py', '--new', nm, '--spread', spf, '--order', orf, '--out_dir', out4)
ok(rc == 0, f'd4 MIT-BIH planted run exit {rc}' + ('' if rc == 0 else ' ' + txt[-300:]))
M = json.load(open(os.path.join(out4, 'summary_tshufi.json')))
mc = {c['kind'].split(' ')[0]: c for c in M['comparisons']}
OLD = {c['kind']: c for c in json.load(open(os.path.join(RS, 'check-tshuf-20261128', 'summary_tshuf.json')))['comparisons']}
ok(all(abs(t) < 1e-12 for t in mc['T5']['d_pp']) and mc['T5']['verdict'] == 'no clear difference', f'd4 T5 = 0 ({mc["T5"]["mean_pp"]:+.3g})')
ok(all(abs(x - y) < 1e-9 for x, y in zip(mc['T3']['d_pp'], OLD['S3']['d_pp'])) and mc['T3']['verdict'] == OLD['S3']['verdict'],
   f'd4 T3 = S3 ({mc["T3"]["mean_pp"]:+.3f} = {OLD["S3"]["mean_pp"]:+.3f})')
ok(all(abs(x - y) < 1e-9 for x, y in zip(mc['T6']['d_pp'], OLD['S2']['d_pp'])), f'd4 T6 = S2 ({mc["T6"]["mean_pp"]:+.3f} = {OLD["S2"]["mean_pp"]:+.3f})')
addb = {kv(l)['seed']: float(kv(l)['f1']) for l in lines(os.path.join(RS, 'flif-v1-mitbih-spkmux-tune')) if l.startswith('RESULT') and kv(l).get('tag') == 'spkmx_lif_muxlin_add_lr0.02_L2'}
exp2 = [100 * (base[s] - addb[s]) for s in ('42', '7', '2026')]
ok(all(abs(x - y) < 1e-9 for x, y in zip(mc['T2']['d_pp'], exp2)), f'd4 T2 = IAND base - ADD base ({sum(exp2) / 3:+.3f}; residual test R1 +0.23)')
ok(all(abs(x - y) < 1e-9 for x, y in zip(M['eval_shuffle']['iand_mixer_drop'], M['eval_shuffle']['add_mixer_drop'])), 'd4 planted order json read through')
mstop = {'SKIP line': (fm + ['SKIP(rc=1) x'], None, spf, orf), 'missing seed': (fm[:2], None, spf, orf),
         'mixed commits': ([setk(fm[0], sha='other')] + fm[1:], None, spf, orf),
         'recount != logged': (fm, lambda rows: rows[0].__setitem__('f1_recomputed', '0.1'), spf, orf),
         'nan f1': ([setk(fm[0], f1='nan')] + fm[1:], None, spf, orf)}
spb = dict(sp, runs=[dict(x, default=x['default'] + 1) for x in sp['runs']]); spbf = os.path.join(a.tmp, 'spread_bad.json'); json.dump(spb, open(spbf, 'w'))
orb = dict(oi, runs=[dict(x, f1=x['f1'] + 1) for x in oi['runs']]); orbf = os.path.join(a.tmp, 'order_bad.json'); json.dump(orb, open(orbf, 'w'))
mstop['spread default != logged'] = (fm, None, spbf, orf)
mstop['order reading != logged'] = (fm, None, spf, orbf)
for name, (L, mut, s_, o_) in mstop.items():
    od = os.path.join(a.tmp, 'out_mstop')
    shutil.rmtree(od, ignore_errors=True)
    rc, txt = run('summarize_tshufi.py', '--new', mb_suite('mb_stop', L, mut), '--spread', s_, '--order', o_, '--out_dir', od)
    ok(rc == 2 and not os.path.exists(od), f'd5 MIT-BIH stop "{name}": exit {rc}, ' + txt.strip().splitlines()[-1][:110])

# ---- recheck
rd = os.path.join(a.tmp, 'out_both'); shutil.rmtree(rd, ignore_errors=True); os.makedirs(rd)
shutil.copy(os.path.join(out1, 'summary_adiand.json'), rd); shutil.copy(os.path.join(out4, 'summary_tshufi.json'), rd)
rc, txt = run('recheck_iand.py', '--new_ad', new, '--new_mb', nm, '--repro', rep, '--dir', rd)
ok(rc == 0 and 'ALL CHECKS PASSED' in txt, f'd6 recheck on planted summaries: {txt.strip().splitlines()[-1]}')
T_ = json.load(open(os.path.join(rd, 'summary_adiand.json')))
T_['comparisons'][0]['verdict'] = 'A better'
json.dump(T_, open(os.path.join(rd, 'summary_adiand.json'), 'w'))
rc, txt = run('recheck_iand.py', '--new_ad', new, '--new_mb', nm, '--repro', rep, '--dir', rd)
ok('ALL CHECKS PASSED' not in txt and 'FAIL' in txt, f'd6 recheck on a tampered J1 verdict: {txt.strip().splitlines()[-1]}')
print('ALL DRY CHECKS PASSED' if bad == 0 else f'{bad} FAIL')
sys.exit(0 if bad == 0 else 1)
