"""MODI 2026-11-27: test of summarize_rd.py -- synthetic new suites (flif-v1-mitbih-rd, flif-v1-sleep5, flif-v1-ett-rd) built from the
real reference rows plus planted offsets, in a temporary copy of the results tree; the planted verdicts / adoption readings / candidate
must come out, and 9 broken inputs must stop. usage (classification folder): python test_summarize_rd.py"""
import contextlib
import io
import os
import re
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import select_sleep as SS
import summarize_rd as SR

SLEEP_REF = SR.SLEEP_REF
FAILS = []


def ok(name, cond, info=''):
    print(f'{"PASS" if cond else "FAIL"} {name} {info}', flush=True)
    if not cond:
        FAILS.append(name)


def line(d):
    return 'RESULT ' + ' '.join(f'{k}={v}' for k, v in d.items())


def ref_rows(path):
    return [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in open(path) if l.startswith('RESULT')]


# planted models: P_ALL passes everywhere (candidate); P_EN passes on accuracy but its ETT energy is 1.7x; P_E (E test) is better than
# its own spk-off base in MIT-BIH
P_ALL, P_EN = ('flif', 'mem', 1), ('lif', 'memd', 1)


def build(tmp, mutate=None):
    for sub, suite in (('classification', 'flif-v1-mitbih-r1'), ('forecasting', 'flif-v1-ett-r1')) + tuple(('classification', s) for s in SLEEP_REF):
        os.makedirs(os.path.join(tmp, sub, 'results', suite), exist_ok=True)
        shutil.copy(os.path.join(ROOT, sub, 'results', suite, 'raw.txt'), os.path.join(tmp, sub, 'results', suite, 'raw.txt'))
    # MIT-BIH
    r1 = ref_rows(os.path.join(ROOT, 'classification', 'results', 'flif-v1-mitbih-r1', 'raw.txt'))
    sf = {r['seed']: r for r in r1 if r['cond'] == 'spkall_lif'}
    L = []
    for m in SR.MODELS:
        if m == SR.BASE_REF:
            continue
        for s in SR.SEEDS:
            f1 = float(sf[s]['f1']) + {P_ALL: 0.010, P_EN: 0.006}.get(m, -0.002) + (0.008 if m == ('lif', 'spk', 1) else 0)
            if m == ('lif', 'spk', 0):                                   # lif spk+acc (+0.6 %p) - lif spk (-0.4 %p) = +1.0 %p
                f1 = float(sf[s]['f1']) - 0.004
            e = float(sf[s]['syn_energy_uJ']) * (1.2 if m in (P_ALL, P_EN) else 1.1)
            L.append(line(dict(exp='x', cond=SR.cond_rd(m), dataset='MITBIH', seed=s, lr='0.01', val='0', overall_acc='0.98', f1=f'{f1:.6f}',
                               syn_energy_uJ=f'{e:.4f}', firing_rate='0.4', epochs='30', pe='0', wdx='1', sha='aaa1111',
                               aux='0.1' if m[2] else '0')))
    mb = L
    # Sleep fill-in
    sf_sl = {}
    for s in SLEEP_REF:
        for r in ref_rows(os.path.join(ROOT, 'classification', 'results', s, 'raw.txt')):
            if r['cond'] == 'spkall_lif' and r['lr'] == '0.003':
                sf_sl[int(r['fold'])] = r
    L = []
    for m in sorted(SR.FILL):
        for f in range(5):
            v = float(sf_sl[f]['macro_f1']) + {P_ALL: 0.012, P_EN: 0.008}.get(m, -0.01)
            e = float(sf_sl[f]['syn_energy_uJ']) * (1.2 if m in (P_ALL, P_EN) else 1.1)
            L.append(line(dict(exp='x', data='SLEEPEDF', cond=SR.cond_sleep(m), fold=f, seed='42', lr='0.003', bs='64', epochs='30', score=f'{v:.6f}',
                               acc='0.8', macro_f1=f'{v:.6f}', syn_energy_uJ=f'{e:.4f}', syops_firing_rate='0.4', best_epoch_val='7:0.7', sha='bbb2222')))
    sl = L
    # ETT
    r1e = ref_rows(os.path.join(ROOT, 'forecasting', 'results', 'flif-v1-ett-r1', 'raw.txt'))
    sfe = {(r['ds'], r['pl'], r['seed']): r for r in r1e if r['cond'] == 'spkall_lif'}
    L = []
    for m in SR.MODELS:
        if m == SR.BASE_REF:
            continue
        for d, p in SR.CELLS:
            for s in SR.SEEDS:
                b = sfe[(d, p, s)]
                k = {P_ALL: 0.97, P_EN: 0.97}.get(m, 1.005)
                mse, mae = float(b['mse']) * k, float(b['mae']) - (0.004 if m in (P_ALL, P_EN) else -0.001)
                e = float(b['syn_energy_uJ']) * (1.2 if m == P_ALL else 1.7 if m == P_EN else 1.1)
                L.append(line(dict(exp='x', ds=d, pl=p, cond=SR.cond_rd(m), seed=s, mse=f'{mse:.6f}', mae=f'{mae:.6f}', syops_firing_rate='0.4',
                                   syn_energy_uJ=f'{e:.4f}', seq_len='96', wdx='1', vpe='4', pe='0', sha='ccc3333')))
    et = L
    if mutate:
        mb, sl, et = mutate(mb, sl, et)
    for (sub, suite), rows in (((('classification', 'flif-v1-mitbih-rd')), mb), (('classification', 'flif-v1-sleep5'), sl),
                               (('forecasting', 'flif-v1-ett-rd'), et)):
        os.makedirs(os.path.join(tmp, sub, 'results', suite), exist_ok=True)
        open(os.path.join(tmp, sub, 'results', suite, 'raw.txt'), 'w').write('\n'.join(rows) + '\n')


def run(task, mutate=None):
    tmp = tempfile.mkdtemp()
    try:
        build(tmp, mutate)
        with contextlib.redirect_stdout(io.StringIO()):
            return SR.main(['--task', task, '--root', tmp])
    except SystemExit as e:
        return f'STOP {e}'
    finally:
        SS.HERE = HERE
        shutil.rmtree(tmp)


def main():
    d = run('all')
    if isinstance(d, str):
        ok('good input runs', False, d); print('FAILED'); sys.exit(1)
    T = d['tasks']
    nm = SR.name
    # the spk-off pop LIF model is a real reference run (r1 / Sleep suites), not planted: it may pass on its own real numbers
    ok('a1 candidates = the planted all-pass model (+ only real-reference models)', nm(P_ALL) in d['candidates'] and nm(P_EN) not in d['candidates']
       and all(c in (nm(P_ALL), nm(SR.BASE_REF)) for c in d['candidates']), str(d['candidates']))
    ok('a2 the planted model passes on each task; the other plant fails ETT on energy',
       all(T[k]['S'][f'{nm(P_ALL)} − Spikformer(DE)']['adoption']['adoption'] == 'pass' for k in ('mitbih', 'sleep', 'ett'))
       and T['ett']['S'][f'{nm(P_EN)} − Spikformer(DE)']['adoption']['adoption'] == 'energy over'
       and T['mitbih']['S'][f'{nm(P_EN)} − Spikformer(DE)']['adoption']['adoption'] == 'pass')
    ok('a3 MIT-BIH S verdict of the planted model: better (+1.0 %p, 3/3)', T['mitbih']['S'][f'{nm(P_ALL)} − Spikformer(DE)']['verdict'] == 'better')
    ok('a4 ETT S verdict of the planted model: better (R -3 %, 12/12 lower)', T['ett']['S'][f'{nm(P_ALL)} − Spikformer(DE)']['verdict'] == 'better'
       and abs(T['ett']['S'][f'{nm(P_ALL)} − Spikformer(DE)']['stat'] + 3.0) < 0.01, str(T['ett']['S'][f'{nm(P_ALL)} − Spikformer(DE)']))
    e = T['mitbih']['E'][f'{nm(("lif", "spk", 1))} − {nm(("lif", "spk", 0))}']
    ok('a5 MIT-BIH E: lif spk+acc − lif spk = +1.0 %p, better', e['verdict'] == 'better' and abs(e['stat'] - 1.0) < 1e-6, str(e))
    ok('a6 below-bar models read "below"', T['sleep']['S'][f'{nm(("pop_lif", "memd", 0))} − Spikformer(DE)']['adoption']['adoption'] == 'below')
    f = T['ett']['F'][f'{nm(P_ALL)} vs SpikF (paper)']
    ok('a7 SpikF reading computed from the branch refs', f['adoption'] in ('pass', 'not passed') and isinstance(f['margin_mse'], float), str(f))
    ok('a8 N comparison present (lif / flif vs pop_lif at 6 readout x aux)', len(T['ett']['N']) == 12 and len(T['mitbih']['N']) == 12)
    bad = {'missing ETT row': lambda a, b, c: (a, b, c[1:]),
           'duplicate MIT-BIH row': lambda a, b, c: (a + a[:1], b, c),
           'nan Sleep value': lambda a, b, c: (a, [re.sub(r' macro_f1=[0-9.]+', ' macro_f1=nan', b[0])] + b[1:], c),
           'SKIP line': lambda a, b, c: (a + ['SKIP(rc=1) cond=rd_lif_spk seed=42'], b, c),
           'unexpected row': lambda a, b, c: (a, b, c + [c[0].replace('seed=42', 'seed=9')]),
           'two shas': lambda a, b, c: (a, b, [c[0].replace('sha=ccc3333', 'sha=ddd4444')] + c[1:]),
           'ETT vpe 1': lambda a, b, c: (a, b, [c[0].replace('vpe=4', 'vpe=1')] + c[1:]),
           'MIT-BIH aux mismatch': lambda a, b, c: ([x.replace('aux=0.1', 'aux=0') if 'rd_lif_spk_acc' in x else x for x in a], b, c),
           'Sleep epochs 29': lambda a, b, c: (a, [b[0].replace('epochs=30', 'epochs=29')] + b[1:], c)}
    for k, fn in bad.items():
        r = run('all', fn)
        ok(f'b stops: {k}', isinstance(r, str) and r.startswith('STOP'), r[:90] if isinstance(r, str) else 'did not stop')
    print('ALL PASS' if not FAILS else f'FAILED: {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
