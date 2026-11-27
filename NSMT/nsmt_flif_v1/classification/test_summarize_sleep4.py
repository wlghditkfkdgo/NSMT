"""MODI 2026-11-27: test of summarize_sleep4.py -- planted readings on synthetic flif-v1-sleep4 rows (offsets added to the real reference
rows) and 7 cases that must stop. usage (classification folder): python test_summarize_sleep4.py"""
import io, os, re, shutil, sys, contextlib, tempfile
C = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, C)
import select_sleep as SS
import summarize_sleep4 as S4
REFS = ('flif-v1-sleep-lr', 'flif-v1-sleep', 'flif-v1-sleep2', 'flif-v1-sleep3')
ref = {}
for s in REFS:
    for r in SS.rows(s):
        ref[(r['cond'], r['lr'], int(r['fold']))] = float(r['macro_f1'])
SF = lambda lr, f: ref[('spkall_lif', lr, f)]
a, b = '0.001', '0.003'
plan = {  # (cond, lr) -> (base fn, offsets in %p per fold)
    ('q_lif', b): (lambda f: SF(b, f), [1.5] * 5),                  # L@0.003 better
    ('q_lif', a): (lambda f: SF(a, f), [0.5] * 5),                  # L@0.001 no clear -> "depends on lr"
    ('kv_lif', a): (lambda f: SF(a, f), [-1.2] * 5), ('kv_lif', b): (lambda f: SF(b, f), [0.0] * 5),
    ('q_lif_cshuf', b): (lambda f: SF(b, f), [-0.5] * 5),          # O = +2.0 better -> q_lif candidate
    ('q_lif_nml', b): (lambda f: SF(b, f), [1.5] * 5),
    ('q_lif_mem', b): (lambda f: SF(b, f), [-1.2] * 5),            # S worse
    ('q_lif_mem_cshuf', b): (lambda f: SF(b, f), [-1.2] * 5), ('q_lif_mem_nml', b): (lambda f: SF(b, f), [-1.0] * 5),  # -> neither
    ('q_lif_acc', b): (lambda f: SF(b, f), [0.2] * 5),
    ('q_lif_acc_cshuf', b): (lambda f: SF(b, f), [-1.5] * 5),      # O acc +1.7 better, S no -> "order / memory now counts..."
    ('q_lif_mem_acc', b): (lambda f: SF(b, f), [2.0] * 5),         # beats (no own control)
    ('q_flif', b): (lambda f: SF(b, f), [1.1] * 5), ('q_flif_mem', b): (lambda f: SF(b, f), [-0.5] * 5)}
VAL = {('q_lif', a): 0.70, ('q_lif', b): 0.71, ('kv_lif', a): 0.70, ('kv_lif', b): 0.70}   # q -> 0.003, kv tie -> 0.001
def line(c, lr, f, mf, sha='abc1234', **ov):
    d = dict(exp='x', data='SLEEPEDF', cond=c, fold=f, seed=42, lr=lr, bs=64, epochs=30, score=mf, acc=0.8, macro_f1=mf, f1_1=0.4,
             syn_energy_uJ=20.0, syops_firing_rate=0.3, best_epoch_val=f'7:{VAL.get((c, lr), 0.7)}', sha=sha)
    d.update(ov)
    return 'RESULT ' + ' '.join(f'{k}={v}' for k, v in d.items())
def good():
    L = []
    for (c, lr), (base, off) in plan.items():
        for f in range(5):
            L.append(line(c, lr, f, base(f) + off[f] / 100))
    return L
def run(lines, name):
    T = tempfile.mkdtemp()
    for s in REFS:
        os.makedirs(f'{T}/results/{s}'); shutil.copy(f'{C}/results/{s}/raw.txt', f'{T}/results/{s}/raw.txt')
    os.makedirs(f'{T}/results/flif-v1-sleep4'); open(f'{T}/results/flif-v1-sleep4/raw.txt', 'w').write('\n'.join(lines) + '\n')
    SS.HERE = T
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return S4.main(out_dir=f'{T}/out')
    except SystemExit as e:
        return f'STOP {e}'
    finally:
        SS.HERE = C
        shutil.rmtree(T)
L0 = good(); assert len(L0) == 70, len(L0)
d = run(L0, 'good')
exp = {'q_lif': 'candidate for the base (beats Spikformer(DE) and the Neocortex order / memory counts)', 'q_lif_mem': 'neither',
       'q_lif_acc': 'the Neocortex order / memory now counts, but Spikformer(DE) is not beaten',
       'q_lif_mem_acc': 'beats Spikformer(DE) (no own order / memory control)', 'q_flif': 'beats Spikformer(DE) (no own order / memory control)',
       'q_flif_mem': 'does not beat Spikformer(DE)'}
res = [('readings', d['reading'] == exp), ('L reading', d['L reading'] == 'depends on lr'),
       ('lr choice', d['lr']['q_lif']['lr'] == b and d['lr']['kv_lif']['lr'] == a),
       ('Q2 query better', d['Q2 q_lif − kv_lif (chosen lrs)']['verdict'] == 'query better'),   # q@0.003 (+1.5 vs SF0.003) - kv@0.001 (-1.2 vs SF0.001)
       ('E mem worse', d['E q_lif_mem − q_lif']['verdict'] == 'worse')]
q2 = [d['Q2 q_lif − kv_lif (chosen lrs)']['mean']]
bad_cases = {'missing row': L0[1:], 'duplicate': L0 + L0[:1], 'nan': [re.sub(r' macro_f1=[0-9.]+', ' macro_f1=nan', L0[0])] + L0[1:],
             'SKIP line': L0 + ['SKIP(rc=1, no result) cond=q_lif fold=0'], 'unexpected row': L0 + [line('q_lif', '0.01', 0, 0.7)],
             'two shas': [L0[0].replace('sha=abc1234', 'sha=def5678')] + L0[1:], 'epochs 29': [L0[0].replace('epochs=30', 'epochs=29')] + L0[1:]}
for k, Lb in bad_cases.items():
    r = run(Lb, k)
    res.append((f'stops: {k}', isinstance(r, str) and r.startswith('STOP')))
for k, v in res:
    print('PASS' if v else 'FAIL', k)
print('ALL PASS' if all(v for _, v in res) else 'FAILED')
sys.exit(0 if all(v for _, v in res) else 1)
