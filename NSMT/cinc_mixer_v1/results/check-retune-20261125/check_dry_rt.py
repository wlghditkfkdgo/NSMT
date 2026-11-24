"""Dry run of the ETTh1 / ETTm1 retuning pipeline (2026-11-25, exp/f-lif-pop-v3; PROJECT_LOG pre-registration of 2026-11-25). No training.
Simulates scripts/retune_driver.sh in a temporary folder (RT_DRY): for every stage it runs make_queue_rt.py, answers each queue line
with a fabricated RESULT row, then runs summarize_rt.py and recheck_rt.py; at the end fabricated energy rows, the judge and its recheck.
The fabricated best validation L1 is base(pl) x product of one factor per knob value (FACT below), so every stage's winner is known
by construction (the argmin of the factors of that knob; the current value on a tie; the earlier value on a tie of two others).
  A  chains a and x diverge on both datasets: expected path and queue sizes 60 / 24 / 48 / 72 / 72 / 48 / 96 / 24 (444 rows);
     fabricated test values = SpikF - 0.01 on the 8 new cells -> judge PASS expected for both models (energy 10 µJ)
  B  ETTm1 chain x picks the same patch as chain a (shared runs are queued once): 60 / 24 / 48 / 54 / 54 / 36 / 84 / 18 (378 rows);
     test values = SpikF + 0.01 -> NOT PASSED
  J  judge.json = an independent computation (16-cell means, 8 margins, pass flags, energy cells, verdict; ETTh2 / ETTm2 cells
     from the 11-24 summary.json)
  N  fail-closed cases (each exits non-zero and writes no new selection / judge file): SKIP line, missing row, duplicate row, row of a
     configuration no stage asked for, tag not matching the fields, 100-epoch row, non-finite mse, an existing different selection,
     chain x rows missing while chain a is complete (neither file written), queue of a stage whose earlier selection is missing,
     a missing energy row, a recomputed MSE off by 1e-3; recheck fails on a tampered selection
  Q  resume: with half of a stage's rows present the queue holds exactly the other half
usage (from cinc_mixer_v1/): python -B results/check-retune-20261125/check_dry_rt.py > results/check-retune-20261125/check_dry_rt.txt
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable
SU = 'flif-v1-ett-spkmux-retune'
KN = ['LR', 'LAYERS', 'PATCH', 'EMB', 'MLPR', 'BLOCKS', 'WD']
FLD = dict(LR='lr', LAYERS='layers', PATCH='patch', EMB='emb', MLPR='mlpr', BLOCKS='blocks', WD='wd')
STAGES = ['lr', 'layers', 'patch', 'dmodel', 'mlp', 'blocks', 'wd', 'final']
CELLS16 = [(d, p) for d in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2') for p in (96, 192, 336, 720)]
fails = 0


def ok(c, m):
    global fails
    print(('PASS ' if c else 'FAIL ') + m)
    fails += 0 if c else 1


def tagof(ds, e):
    t = f'spkmx_lif_muxlin_iand_lr{e["LR"]}_L{e["LAYERS"]}'
    for k, suf, d in (('PATCH', '_pt', '8'), ('EMB', '_e', '64'), ('MLPR', '_mlp', '0.5' if ds == 'ETTh1' else '1'), ('BLOCKS', '_blk', '1'), ('WD', '_wd', '0.06')):
        if e[k] != d:
            t += f'{suf}{e[k]}'
    return t


# factor per (ds, knob, value); a value not listed: 1.0 for the start value, 1.01 otherwise
START = {'ETTh1': dict(LR='0.0005', LAYERS='2', PATCH='8', EMB='32', MLPR='1', BLOCKS='1', WD='0.06'),
         'ETTm1': dict(LR='0.001', LAYERS='2', PATCH='8', EMB='64', MLPR='1', BLOCKS='1', WD='0.06')}
FACT_A = {'ETTh1': {('LR', '0.00025'): 0.99, ('LAYERS', '1'): 0.995, ('PATCH', '16'): 0.98, ('PATCH', '24'): 0.97, ('EMB', '128'): 0.99,
                    ('MLPR', '2'): 0.995, ('WD', '0.01'): 0.999, ('WD', '0.005'): 0.998},
          'ETTm1': {('LAYERS', '3'): 0.99, ('PATCH', '4'): 0.98, ('PATCH', '16'): 0.98, ('PATCH', '24'): 0.97, ('MLPR', '0.5'): 0.99,
                    ('BLOCKS', '2'): 0.995, ('WD', '0'): 1.0}}
EXP_A = {'a': {'ETTh1': ['0.00025', '1', '16', '128', '2', '1', '0.01'], 'ETTm1': ['0.001', '3', '4', '64', '0.5', '2', '0.06']},
         'x': {'ETTh1': ['0.00025', '1', '24', '128', '2', '1', '0.005'], 'ETTm1': ['0.001', '3', '24', '64', '0.5', '2', '0.06']}}
FACT_B = {'ETTh1': dict(FACT_A['ETTh1']), 'ETTm1': {**FACT_A['ETTm1'], ('PATCH', '24'): 1.01}}
EXP_B = {'a': EXP_A['a'], 'x': {'ETTh1': EXP_A['x']['ETTh1'], 'ETTm1': ['0.001', '3', '4', '64', '0.5', '2', '0.06']}}
NQ_A = [60, 24, 48, 72, 72, 48, 96, 24]
NQ_B = [60, 24, 48, 54, 54, 36, 84, 18]
REF = json.load(open(os.path.join(C, 'results', 'ett-full16-20261124', 'refs16.json')))['models']
S16 = json.load(open(os.path.join(C, 'results', 'flif-v1-ett-spkmux-full16', 'summary.json')))
SP = {}
for ds in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
    for r in csv.DictReader(open(os.path.join(C, 'results', 'ett-full16-20261124', 'spikf16', f'spikf_syops_{ds}.csv'))):
        SP[(r['ds'], int(r['pl']))] = float(r['energy_uJ'])


def val(fact, ds, pl, e):
    v = {96: 0.4, 720: 0.6}.get(pl, 0.5)
    for k in KN:
        v *= fact[ds].get((k, e[k]), 1.0 if e[k] == START[ds][k] else 1.01)
    return v


def parse_q(line):
    t = line.split()
    env = dict(x.split('=', 1) for x in t if '=' in x and x.split('=', 1)[0].isupper())
    return env, t[-3], int(t[-2]), int(t[-1])


def fake(env, ds, pl, seed, fact, dtest, epochs='50'):
    v = val(fact, ds, pl, env)
    mse = REF['SpikF']['rows'][f'{ds} {pl}'][0] + dtest + 0.0001 * (seed % 5)
    mae = REF['SpikF']['rows'][f'{ds} {pl}'][1] + dtest + 0.0001 * (seed % 3)
    tg = tagof(ds, env)
    f = dict(exp='cinc_mixer_v1_ett_spkmux', ds=ds, pl=pl, cond='spkmx_lif_muxlin_iand', tag=tg, **{FLD[k]: env[k] for k in KN},
             seed=seed, mse=f'{mse:.6f}', mae=f'{mae:.6f}', val_loss_best=f'{v:.6f}', syn_energy_uJ='10.000000', total_fft_uJ='12.000000',
             best_epoch_val_epochs=f'3:{v:.6f}:14', params=1000, time_s=1, sha='dry',
             dir=f'/dry/{ds}/p{pl}/x+{tg}+seed{seed}+model+spkMixer+epoch+{epochs}+lr+{env["LR"]}/seed{seed}_x')
    return 'RESULT ' + ' '.join(f'{k}={v_}' for k, v_ in f.items())


def run(args, T, env_extra=None):
    p = subprocess.run([PY, '-B'] + args, cwd=C, env=dict(os.environ, RT_DRY=T, **(env_extra or {})), capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip().splitlines()


def rawp(T):
    return os.path.join(T, SU, 'raw.txt')


def append(T, lines):
    os.makedirs(os.path.join(T, SU), exist_ok=True)
    with open(rawp(T), 'a') as f:
        f.write(''.join(l + '\n' for l in lines))


def queue(T, st):
    rc, out = run(['scripts/make_queue_rt.py', st], T)
    if rc != 0:
        return rc, None, out
    path, n = out[-1].rsplit(':', 1)
    L = [l for l in open(path).read().splitlines() if l.strip()]
    os.remove(path)                                                              # the pool would drain it
    return rc, L, out


def energy_rows(T):
    rows = []
    for l in open(rawp(T)):
        if l.startswith('RESULT'):
            r = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
            rows.append(dict(tag=r['tag'], ds=r['ds'], pl=r['pl'], seed=r['seed'], mse_logged=r['mse'], mse_recomputed=r['mse'],
                             body_syn_uJ=f'{10.0 - 1000 * 4.6e-6:.9f}', head_MAC='1000', total_fft_uJ_corr='12.0'))
    with open(os.path.join(T, SU, 'energy_recount.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(x) for x in rows]


def scenario(nm, fact, exp, nq, dtest, keep=None):
    T = tempfile.mkdtemp()
    try:
        sizes = []
        for st in STAGES:
            rc, L, out = queue(T, st)
            if rc != 0:
                ok(False, f'{nm} queue {st}: rc {rc} {out[-1:]}'); return
            sizes.append(len(L))
            append(T, [fake(*parse_q(l), fact, dtest) for l in L])
            if st != 'final':
                rc, out = run(['summarize_rt.py', '--stage', st], T)
                rc2, out2 = run(['recheck_rt.py', '--stage', st], T)
                ok(rc == 0 and rc2 == 0 and out2[-1] == 'ALL CHECKS PASSED', f'{nm} stage {st}: summarizer rc {rc}, recheck "{out2[-1]}"')
            if keep is not None and st == keep:
                shutil.copytree(T, keep_dir[nm])
        ok(sizes == nq, f'{nm} queue sizes {sizes} = {nq} (total {sum(sizes)})')
        got = {}
        for c in ('a', 'x'):
            got[c] = {}
            for ds in ('ETTh1', 'ETTm1'):
                cfg = [json.load(open(os.path.join(T, SU, 'selected_lr.json')))['selected'][ds],
                       json.load(open(os.path.join(T, SU, 'selected_layers.json')))['selected'][ds]]
                cfg += [json.load(open(os.path.join(T, SU, f'selected_{c}_{st}.json')))['selected'][ds] for st in ('patch', 'dmodel', 'mlp', 'blocks', 'wd')]
                got[c][ds] = cfg
        ok(got == exp, f'{nm} selected path = expected by construction: {got}')
        energy_rows(T)
        rc, out = run(['summarize_rt.py', '--stage', 'judge'], T)
        rc2, out2 = run(['recheck_rt.py', '--stage', 'judge'], T)
        ok(rc == 0 and rc2 == 0 and out2[-1] == 'ALL CHECKS PASSED', f'{nm} judge rc {rc} {out[-1:] if rc else ""}, recheck "{out2[-1]}"')
        if rc != 0:
            return
        J = json.load(open(os.path.join(T, SU, 'judge.json')))
        bad = []
        for mk, c in (('first', 'a'), ('ext', 'x')):
            cm = {}
            for d, p in CELLS16:
                if d in ('ETTh1', 'ETTm1'):
                    cm[(d, p)] = (sum(REF['SpikF']['rows'][f'{d} {p}'][0] + dtest + 0.0001 * (s % 5) for s in (42, 7, 2026)) / 3,
                                  sum(REF['SpikF']['rows'][f'{d} {p}'][1] + dtest + 0.0001 * (s % 3) for s in (42, 7, 2026)) / 3, 10.0)
                else:
                    x = S16['models'][mk]['cells'][f'{d} {p}']; cm[(d, p)] = (x['mse'], x['mae'], x['syn_energy_uJ'])
            om = sum(v[0] for v in cm.values()) / 16; oa = sum(v[1] for v in cm.values()) / 16
            acc = True
            for m in ('SpikF', 'SpikeRNN', 'iSpikformer', 'Informer', 'FEDformer', 'PatchTST', 'ETSformer', 'Autoformer'):
                rm = sum(REF[m]['rows'][f'{d} {p}'][0] for d, p in CELLS16) / 16; ra = sum(REF[m]['rows'][f'{d} {p}'][1] for d, p in CELLS16) / 16
                ps = rm - om >= 0.001 - 1e-12 and ra - oa >= 0.001 - 1e-12
                acc = acc and ps
                g = J['models'][mk]['rows'][m]
                if abs(g['margin_mse'] - (rm - om)) > 1e-9 or abs(g['margin_mae'] - (ra - oa)) > 1e-9 or g['passes'] != ps:
                    bad.append(f'{mk} {m}')
            below = sum(cm[c_][2] < SP[c_] for c_ in CELLS16)
            vd = 'PASS' if acc and below == 16 else 'NOT PASSED'
            if abs(J['models'][mk]['mse16'] - om) > 1e-9 or abs(J['models'][mk]['mae16'] - oa) > 1e-9 or J['models'][mk]['verdict'] != vd \
                    or J['models'][mk]['energy_cells_below_spikf'] != below:
                bad.append(f'{mk} means / verdict')
            if [J['final_config'][c][ds][k] for ds in ('ETTh1',) for k in KN] != exp[c]['ETTh1']:
                bad.append(f'{mk} final_config')
            print(f'  {nm} {mk}: 16-cell {om:.6f} / {oa:.6f}, verdict {vd} (judge.json {J["models"][mk]["verdict"]})')
        ok(not bad, f'{nm} J: judge.json = independent computation' + ('' if not bad else ': ' + ', '.join(bad)))
        if nm == 'A':
            ok(all(J['models'][m]['verdict'] == 'PASS' for m in J['models']), 'A: both models PASS (fabricated values below SpikF by 0.01)')
        else:
            ok(all(J['models'][m]['verdict'] == 'NOT PASSED' for m in J['models']), 'B: both models NOT PASSED (fabricated values above SpikF)')
        # N9 / N10 on the finished folder
        for tnm, mut in (('N9 a missing energy row', lambda rows: rows[1:]),
                         ('N10 recomputed MSE off by 1e-3', lambda rows: [dict(rows[0], mse_recomputed=f'{float(rows[0]["mse_recomputed"]) + 1e-3:.6f}')] + rows[1:])):
            ep = os.path.join(T, SU, 'energy_recount.csv')
            orig = list(csv.DictReader(open(ep)))
            fin_tags = {(J['final_tags'][c][ds], ds) for c in ('a', 'x') for ds in ('ETTh1', 'ETTm1')}
            orig.sort(key=lambda r: (r['tag'], r['ds']) not in fin_tags)                # a row the judge needs first
            with open(ep, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(orig[0])); w.writeheader(); [w.writerow(x) for x in mut(orig)]
            if os.path.exists(os.path.join(T, SU, 'judge.json')):
                os.remove(os.path.join(T, SU, 'judge.json'))
            rc, out = run(['summarize_rt.py', '--stage', 'judge'], T)
            ok(rc == 2 and not os.path.exists(os.path.join(T, SU, 'judge.json')), f'{nm} {tnm}: exit {rc}, no judge.json ({out[-1][:100] if out else ""})')
            with open(ep, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(orig[0])); w.writeheader(); [w.writerow(x) for x in orig]
    finally:
        shutil.rmtree(T)


keep_dir = {'A': tempfile.mkdtemp() + '/k'}
scenario('A', FACT_A, EXP_A, NQ_A, -0.01, keep='patch')
scenario('B', FACT_B, EXP_B, NQ_B, +0.01)

# N cases on a copy of scenario A taken right after the patch stage (selections lr, layers, a/x patch exist)
K = keep_dir['A']


def fresh():
    T = tempfile.mkdtemp(); shutil.rmtree(T); shutil.copytree(K, T)
    return T


def dmodel_rows(T):
    rc, L, _ = queue(T, 'dmodel')
    return [fake(*parse_q(l), FACT_A, -0.01) for l in L], L


cases = {}
T = fresh(); R_, L = dmodel_rows(T)
cases['N1 SKIP line'] = (T, R_ + ['SKIP(rc=1, no result) ds=ETTh1 pl=96 cond=x seed=42 out=x'])
T = fresh(); R_, L = dmodel_rows(T); cases['N2 missing row'] = (T, R_[1:])
T = fresh(); R_, L = dmodel_rows(T); cases['N3 duplicate row'] = (T, R_ + [R_[0]])
T = fresh(); R_, L = dmodel_rows(T)
e0, d0, p0, s0 = parse_q(L[0]); cases['N4 unrequested configuration'] = (T, R_ + [fake(dict(e0, LR='0.0003'), d0, p0, s0, FACT_A, -0.01)])
T = fresh(); R_, L = dmodel_rows(T); cases['N5 tag not matching fields'] = (T, [R_[0].replace('tag=spkmx_lif', 'tag=spkmx_lif_x')] + R_[1:])
T = fresh(); R_, L = dmodel_rows(T); cases['N6 100-epoch row'] = (T, [fake(e0, d0, p0, s0, FACT_A, -0.01, epochs='100')] + R_[1:])
T = fresh(); R_, L = dmodel_rows(T); cases['N7 non-finite mse'] = (T, [' '.join('mse=nan' if t.startswith('mse=') else t for t in R_[0].split())] + R_[1:])
T = fresh(); R_, L = dmodel_rows(T)
cases['N11 chain x rows missing'] = (T, [r for r, l in zip(R_, L) if not ('PATCH=24' in l and ' ETTh1 ' in l)])
for nm, (T, rows_) in cases.items():
    append(T, rows_)
    before = sorted(os.listdir(os.path.join(T, SU)))
    rc, out = run(['summarize_rt.py', '--stage', 'dmodel'], T)
    after = sorted(os.listdir(os.path.join(T, SU)))
    ok(rc == 2 and before == after, f'{nm}: exit {rc}, no file written ({out[-1][:110] if out else ""})')
    shutil.rmtree(T)
# N8: an existing different selection is not overwritten
T = fresh(); R_, L = dmodel_rows(T); append(T, R_)
f = os.path.join(T, SU, 'selected_a_dmodel.json')
json.dump({'stage': 'dmodel', 'chain': 'a', 'knob': 'EMB', 'selected': {'ETTh1': '256', 'ETTm1': '64'}, 'current': {}, 'scores': {}}, open(f, 'w'))
txt = open(f).read()
rc, out = run(['summarize_rt.py', '--stage', 'dmodel'], T)
ok(rc == 2 and open(f).read() == txt and not os.path.exists(os.path.join(T, SU, 'selected_x_dmodel.json')),
   f'N8 different selection on disk: exit {rc}, file unchanged, chain x not written ({out[-1][:100]})')
shutil.rmtree(T)
# N12: queue of a stage whose earlier selection is missing
T = fresh()
rc, L, out = queue(T, 'mlp')
ok(rc == 1 and L is None, f'N12 queue mlp before dmodel is selected: exit {rc} ({out[-1][:100]})')
shutil.rmtree(T)
# N13: recheck fails on a tampered selection
T = fresh()
f = os.path.join(T, SU, 'selected_x_patch.json'); s = json.load(open(f)); s['selected']['ETTh1'] = '32'; json.dump(s, open(f, 'w'))
rc, out = run(['recheck_rt.py', '--stage', 'patch'], T)
ok(rc == 1 and out[-1].endswith('CHECKS FAILED'), f'N13 recheck on a tampered selection: exit {rc}, "{out[-1]}"')
shutil.rmtree(T)
# Q: resume -- half of the dmodel rows present
T = fresh(); R_, L = dmodel_rows(T)
append(T, R_[::2])
rc, L2, _ = queue(T, 'dmodel')
ok(rc == 0 and sorted(L2) == sorted(L[1::2]), f'Q resume: queue holds the {len(L2)} missing lines of {len(L)} (want {len(L[1::2])})')
shutil.rmtree(T)
shutil.rmtree(os.path.dirname(K))
print(f'{"ALL PASS" if fails == 0 else str(fails) + " FAIL"}')
sys.exit(0 if fails == 0 else 1)
