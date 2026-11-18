# recheck_pn.py (2026-11-18, exp/f-lif-pop-v3; PROJECT_LOG neuron implementation / placement): independent recomputation of the pre-registered
# selection and comparisons. Reads raw.txt and energy_recount.csv only; does not import summarize_pn / summarize_l2 / summarize_ms. Fails closed on
# missing, ambiguous or non-finite rows and on a recount that does not reproduce the logged metric.
# sf336 (user decision 11-18 21:4x): D1 LIF 336 vs SF-de 336 + reports, without pop LIF mixer-only 336; paper bar = mean better (R < 0) and energy A/B < 1
# usage (from cinc_mixer_v1/): python -B recheck_pn.py select|sf_ett|sf_mit|ett|mitbih|len336|sf336 [--sha <pre-reg commit>[,<later commit>...] (rows of the new suites)]
#        env PN_DRY=<folder>: read the new suites from <folder>/<suite> (dry runs)
import collections, csv, json, math, os, sys
ST = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ('select', 'sf_ett', 'sf_mit', 'ett', 'mitbih', 'len336', 'sf336') else sys.exit('usage: recheck_pn.py <stage> [--sha c]')
# --sha a[,b,...]: the pre-registration commit and any later commits; a later commit is accepted only if it changes no .py / .sh file against the first
SHA = sys.argv[sys.argv.index('--sha') + 1].split(',') if '--sha' in sys.argv else None
if SHA and len(SHA) > 1:
    import subprocess
    for c in SHA[1:]:
        d = subprocess.run(['git', 'diff', '--name-only', SHA[0], c, '--', '*.py', '*.sh'], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        if d.returncode != 0 or d.stdout.strip():
            sys.exit(f'STOP: commit {c} changes code against {SHA[0]}: {d.stdout.strip() or d.stderr.strip()}')
        print(f'commit {c}: no .py / .sh change against {SHA[0]}')
R = os.path.dirname(os.path.abspath(__file__)) + '/results/'
DRY = os.environ.get('PN_DRY')
NEWS = ('flif-v1-ett-sfde', 'flif-v1-mitbih-sfde', 'flif-v1-ett-spkmux-pn', 'flif-v1-mitbih-spkmux-pn', 'flif-v1-ett-spkmux-pn336')
d_ = lambda s: (DRY + '/' + s) if DRY and s in NEWS else R + s
SEEDS = ['42', '7', '2026']
CELLS = [('ETTh2', '96'), ('ETTh2', '720'), ('ETTm2', '96'), ('ETTm2', '720')]
LRM = {'ETTh2': '0.0005', 'ETTm2': '0.0001'}; LRS = {'ETTh2': '0.0005', 'ETTm2': '0.00025'}; LYS = {'ETTh2': '1', 'ETTm2': '2'}
bad = 0


def rows(p):
    out = []
    for ln in open(p):
        if ln.startswith('SKIP'):
            print('SKIP line in', p, ln[:120]); globals()['bad'] += 1
        if ln.startswith('RESULT'):
            out.append(dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t))
    return out


def shas(suite):
    rr = rows(d_(suite) + '/raw.txt')
    c = collections.Counter(r['sha'] for r in rr)
    print(f'{suite}: {len(rr)} rows, sha {dict(c)}')
    if SHA is not None and not set(c) <= set(SHA):
        print('SHA MISMATCH', suite); globals()['bad'] += 1


def ett_arm(suite, tagf, rawname='raw.txt', enname='energy_recount.csv', cells=None):
    """(ds, pl, seed) -> (mse, mae, val, E) for the rows whose tag is tagf(ds)."""
    p = d_(suite)
    rr = rows(p + '/' + rawname)
    en = collections.defaultdict(list)
    for e in csv.DictReader(open(p + '/' + enname)):
        en[(e['ds'], e['pl'], e.get('tag', ''), e['seed'])].append(e)
    T = {}
    for ds, pl in (cells or CELLS):
        for s in SEEDS:
            m = [r for r in rr if r['ds'] == ds and r['pl'] == pl and r['seed'] == s and r.get('tag', '') == tagf(ds) and r.get('cond') == (r.get('cond') if tagf(ds) else 'spk_ssa')]
            e = en.get((ds, pl, tagf(ds), s), [])
            if len(m) != 1 or len(e) != 1:
                print('MISSING/AMBIG', suite, tagf(ds), ds, pl, s, len(m), len(e)); globals()['bad'] += 1; continue
            v = (float(m[0]['mse']), float(m[0]['mae']), float(m[0]['val_loss_best']), float(e[0]['total_fft_uJ_corr']))
            if not all(math.isfinite(x) for x in v) or abs(float(e[0]['mse_recomputed']) - v[0]) > 1e-5 or e[0]['binary'] != 'True':
                print('BAD', suite, tagf(ds), ds, pl, s, v); globals()['bad'] += 1; continue
            T[(ds, pl, s)] = v
    return T


def sf_sj_ett():
    """the tuned spikingjelly Spikformer-style rows: ETTh2 lr 5e-4 = the reused 10-31 baseline rows, ETTm2 lr 2.5e-4 = the 11-01 tuning rows."""
    a = ett_arm('flif-v1-ett-spkmux', lambda ds: '', 'baseline_spk_ssa_raw.txt', 'baseline_spk_ssa_energy.csv', [c for c in CELLS if c[0] == 'ETTh2'])
    b = ett_arm('flif-v1-ett-spkmux-tune', lambda ds: f'spk_ssa_lr{LRS[ds]}_L{LYS[ds]}', cells=[c for c in CELLS if c[0] == 'ETTm2'])
    a.update(b)
    return a


def mit_arm(suite, tag):
    p = d_(suite)
    rr = [r for r in rows(p + '/raw.txt') if r['tag'] == tag and r.get('val_fold', '-1') == '-1']
    en = collections.defaultdict(list)
    for e in csv.DictReader(open(p + '/energy_recount.csv')):
        if e.get('val_fold', '-1') == '-1':
            en[(e['tag'], e['seed'])].append(e)
    T = {}
    for s in SEEDS:
        m = [r for r in rr if r['seed'] == s]
        e = en.get((tag, s), [])
        if len(m) != 1 or len(e) != 1:
            print('MISSING/AMBIG', suite, tag, s, len(m), len(e)); globals()['bad'] += 1; continue
        f1 = float(m[0]['f1'])
        if not math.isfinite(f1) or abs(float(e[0]['f1_recomputed']) - f1) > 1e-5:
            print('BAD', suite, tag, s, f1); globals()['bad'] += 1; continue
        T[s] = (100 * f1, float(e[0]['total_fft_uJ_corr']))
    return T


def cmp_ett(name, A, B):
    cell = []; lo = hi = 0
    for ds, pl in CELLS:
        rs = [100 * (A[(ds, pl, s)][0] - B[(ds, pl, s)][0]) / B[(ds, pl, s)][0] for s in SEEDS]
        lo += sum(x < 0 for x in rs); hi += sum(x > 0 for x in rs); cell.append(sum(rs) / 3)
    Rm = sum(cell) / 4
    v = 'better' if Rm <= -1 and lo >= 8 else 'worse' if Rm >= 1 and hi >= 8 else 'no clear difference'
    er = sum(sum(A[(ds, pl, s)][3] for s in SEEDS) / sum(B[(ds, pl, s)][3] for s in SEEDS) for ds, pl in CELLS) / 4
    print(f'{name}: R {Rm:+.2f} lower {lo}/12 {v} energy {er:.2f} cells ' + ', '.join(f'{c:+.2f}' for c in cell))
    return Rm, er


def cmp_mit(name, A, B):
    d = [A[s][0] - B[s][0] for s in SEEDS]; m = sum(d) / 3; hi = sum(x > 0 for x in d); lo = sum(x < 0 for x in d)
    v = 'better' if m >= .5 and hi >= 2 else 'worse' if m <= -.5 and lo >= 2 else 'no clear difference'
    er = sum(A[s][1] for s in SEEDS) / sum(B[s][1] for s in SEEDS)
    print(f'{name}: {m:+.2f} higher {hi}/3 ' + ', '.join(f'{x:+.2f}' for x in d) + f' {v} energy {er:.2f}')


ARMS = {'LIF': ('lif', ''), 'LIF+mean': ('lif', '_mpmean'), 'f-LIF': ('flif', ''), 'f-LIF+mean': ('flif', '_mpmean'), 'pop LIF all': ('pop_lif', ''),
        'pop LIF all+mean': ('pop_lif', '_mpmean'), 'pop LIF mixer': ('pop_lif', '_nsmixer'), 'pop LIF mixer+mean': ('pop_lif', '_mpmean_nsmixer'),
        'pop f-LIF all': ('pop_flif', ''), 'pop f-LIF all+mean': ('pop_flif', '_mpmean'), 'pop f-LIF mixer': ('pop_flif', '_nsmixer'),
        'pop f-LIF mixer+mean': ('pop_flif', '_mpmean_nsmixer')}


def suite_of(arm, task):
    old = {'LIF': ('flif-v1-ett-spkmux-tune', 'flif-v1-mitbih-spkmux-tune'), 'LIF+mean': ('flif-v1-ett-spkmux-mx3', 'flif-v1-mitbih-spkmux-mx3'),
           'pop LIF all': ('flif-v1-ett-spkmux-stft', 'flif-v1-mitbih-spkmux-stft')}
    if arm in old:
        return old[arm][0 if task == 'ett' else 1]
    return 'flif-v1-ett-spkmux-pn' if task == 'ett' else 'flif-v1-mitbih-spkmux-pn'


def sel():
    e = json.load(open(d_('flif-v1-ett-sfde') + '/selected_lr.json'))['selected']
    m = json.load(open(d_('flif-v1-mitbih-sfde') + '/selected_lr.json'))['selected']
    return e, m


QS = [('B1 pop LIF mixer vs LIF', 'pop LIF mixer', 'LIF'), ('B2 pop LIF all vs LIF', 'pop LIF all', 'LIF'), ('B3 pop LIF mixer vs pop LIF all', 'pop LIF mixer', 'pop LIF all'),
      ('B4 pop f-LIF mixer vs f-LIF', 'pop f-LIF mixer', 'f-LIF'), ('B5 pop f-LIF all vs f-LIF', 'pop f-LIF all', 'f-LIF'),
      ('B6 pop f-LIF mixer vs pop f-LIF all', 'pop f-LIF mixer', 'pop f-LIF all')] + \
     [(f'C{i} {n}+mean vs {n}', f'{n}+mean', n) for i, n in enumerate(('f-LIF', 'pop LIF all', 'pop LIF mixer', 'pop f-LIF all', 'pop f-LIF mixer'), 1)] + \
     [('(report) f-LIF vs LIF', 'f-LIF', 'LIF'), ('(report) LIF+mean vs LIF', 'LIF+mean', 'LIF')]

if ST == 'select':
    shas('flif-v1-ett-sfde'); shas('flif-v1-mitbih-sfde')
    rr = [r for r in rows(d_('flif-v1-ett-sfde') + '/raw.txt') if r['cond'] == 'spk_ssa']
    out = {}
    for ds in ('ETTh2', 'ETTm2'):
        V = {}
        for lr in ('0.0001', '0.00025', '0.0005', '0.001', '0.002'):
            for pl in ('96', '720'):
                v = [float(r['val_loss_best']) for r in rr if r['ds'] == ds and r['pl'] == pl and r['lr'] == lr and r['tag'] == f'spk_ssa_lr{lr}_L{LYS[ds]}_nalif']
                if len(v) != 3:
                    print('MISSING', ds, pl, lr, len(v)); bad += 1; continue
                V[(lr, pl)] = sum(v) / 3
        if bad: break
        sc = {lr: sum(V[(lr, pl)] / V[('0.0005', pl)] for pl in ('96', '720')) / 2 for lr in ('0.0001', '0.00025', '0.0005', '0.001', '0.002')}
        b = min(sc, key=lambda k: sc[k]); b = b if sc[b] < sc['0.0005'] - 1e-12 else '0.0005'
        out[ds] = b
        print(ds, 'selected', b, ' '.join(f'{k} {v:.5f}' for k, v in sc.items()))
    mm = [r for r in rows(d_('flif-v1-mitbih-sfde') + '/raw.txt') if r.get('val_fold') == '0']
    F = {}
    for lr in ('0.0025', '0.005', '0.01', '0.02'):
        v = [100 * float(r['val_f1_last']) for r in mm if r['lr'] == lr and r['tag'] == f'spk_ssa_lr{lr}_L2_v0_nalif']
        if len(v) != 3:
            print('MISSING MIT-BIH', lr, len(v)); bad += 1; continue
        F[lr] = sum(v) / 3
    if not bad:
        b = max(F, key=lambda k: F[k]); b = b if F[b] > F['0.01'] + 1e-9 else '0.01'
        print('MITBIH selected', b, ' '.join(f'{k} {v:.3f}' for k, v in F.items()))
        e, m = sel()
        print('selected_lr.json agrees:', e == out and m == {'MITBIH': b})
        if not (e == out and m == {'MITBIH': b}): bad += 1
elif ST in ('sf_ett', 'ett', 'len336', 'sf336'):
    e, _ = sel()
    if ST == 'sf336':
        shas('flif-v1-ett-spkmux-pn336')
        T = {'LIF 336': ett_arm('flif-v1-ett-spkmux-len336', lambda ds: f'spkmx_lif_muxlin_iand_lr{LRM[ds]}_L1_s336'),
             'pop LIF all 336': ett_arm('flif-v1-ett-spkmux-len336', lambda ds: f'spkmx_pop_lif_muxlin_iand_lr{LRM[ds]}_L1_s336'),
             'SF-sj 336': ett_arm('flif-v1-ett-spkmux-len336', lambda ds: f'spk_ssa_lr{LRS[ds]}_L{LYS[ds]}_s336'),
             'SF-de 336': ett_arm('flif-v1-ett-spkmux-pn336', lambda ds: f'spk_ssa_lr{e[ds]}_L{LYS[ds]}_s336_nalif')}
        Q = [('D1 LIF 336 vs SF-de 336', 'LIF 336', 'SF-de 336'), ('(report) pop LIF all 336 vs SF-de 336', 'pop LIF all 336', 'SF-de 336'),
             ('(report) SF-de 336 vs SF-sj 336', 'SF-de 336', 'SF-sj 336'), ('(report) LIF 336 vs SF-sj 336', 'LIF 336', 'SF-sj 336')]
    elif ST == 'len336':
        shas('flif-v1-ett-spkmux-pn336')
        T = {'LIF 336': ett_arm('flif-v1-ett-spkmux-len336', lambda ds: f'spkmx_lif_muxlin_iand_lr{LRM[ds]}_L1_s336'),
             'pop LIF all 336': ett_arm('flif-v1-ett-spkmux-len336', lambda ds: f'spkmx_pop_lif_muxlin_iand_lr{LRM[ds]}_L1_s336'),
             'SF-sj 336': ett_arm('flif-v1-ett-spkmux-len336', lambda ds: f'spk_ssa_lr{LRS[ds]}_L{LYS[ds]}_s336'),
             'pop LIF mixer 336': ett_arm('flif-v1-ett-spkmux-pn336', lambda ds: f'spkmx_pop_lif_muxlin_iand_lr{LRM[ds]}_L1_nsmixer_s336'),
             'SF-de 336': ett_arm('flif-v1-ett-spkmux-pn336', lambda ds: f'spk_ssa_lr{e[ds]}_L{LYS[ds]}_s336_nalif')}
        Q = [('D1 LIF 336 vs SF-de 336', 'LIF 336', 'SF-de 336'), ('D2 pop LIF mixer 336 vs LIF 336', 'pop LIF mixer 336', 'LIF 336'),
             ('D3 pop LIF mixer 336 vs pop LIF all 336', 'pop LIF mixer 336', 'pop LIF all 336'), ('(report) pop LIF mixer 336 vs SF-de 336', 'pop LIF mixer 336', 'SF-de 336'),
             ('(report) SF-de 336 vs SF-sj 336', 'SF-de 336', 'SF-sj 336'), ('(report) LIF 336 vs SF-sj 336', 'LIF 336', 'SF-sj 336')]
    else:
        shas('flif-v1-ett-sfde')
        T = {'SF-de': ett_arm('flif-v1-ett-sfde', lambda ds: f'spk_ssa_lr{e[ds]}_L{LYS[ds]}_nalif'), 'SF-sj': sf_sj_ett()}
        names = ['LIF'] if ST == 'sf_ett' else list(ARMS)
        if ST == 'ett':
            shas('flif-v1-ett-spkmux-pn')
        for n in names:
            mx, suf = ARMS[n]
            T[n] = ett_arm(suite_of(n, 'ett'), lambda ds, mx=mx, suf=suf: f'spkmx_{mx}_muxlin_iand_lr{LRM[ds]}_L1{suf}')
        Q = [('A1 LIF SpkMixer vs SF-de', 'LIF', 'SF-de'), ('(report) SF-de vs SF-sj', 'SF-de', 'SF-sj'), ('(report) LIF SpkMixer vs SF-sj', 'LIF', 'SF-sj')]
        if ST == 'ett':
            Q = QS + [(f'(report) {n} vs SF-de', n, 'SF-de') for n in ARMS]
    if bad: sys.exit('STOP: %d problems' % bad)
    for name, a, b in Q:
        Rm, er = cmp_ett(name, T[a], T[b])
        if ST == 'sf336' and b == 'SF-de 336':
            print(f'   paper bar: mean better {Rm < 0}, energy < 1 {er < 1} -> {"met" if Rm < 0 and er < 1 else "not met"} (R {Rm:+.3f}, energy {er:.4f})')
    for n, X in T.items():
        ms = [sum(X[(ds, pl, s)][0] for s in SEEDS) / 3 for ds, pl in CELLS]
        print(f'{n:22s}', ' '.join(f'{z:.4f}' for z in ms), f'mean {sum(ms) / 4:.4f}', 'energy %.2f' % (sum(sum(X[(ds, pl, s)][3] for s in SEEDS) / 3 for ds, pl in CELLS) / 4))
else:
    _, m = sel()
    shas('flif-v1-mitbih-sfde')
    T = {'SF-de': mit_arm('flif-v1-mitbih-sfde', f'spk_ssa_lr{m["MITBIH"]}_L2_nalif'), 'SF-sj': mit_arm('flif-v1-mitbih-spkmux-tune', 'spk_ssa_lr0.005_L2')}
    names = ['LIF'] if ST == 'sf_mit' else list(ARMS)
    if ST == 'mitbih':
        shas('flif-v1-mitbih-spkmux-pn')
    for n in names:
        mx, suf = ARMS[n]
        T[n] = mit_arm(suite_of(n, 'mitbih'), f'spkmx_{mx}_muxlin_add_lr0.02_L2{suf}')
    Q = [('A1 LIF SpkMixer vs SF-de', 'LIF', 'SF-de'), ('(report) SF-de vs SF-sj', 'SF-de', 'SF-sj'), ('(report) LIF SpkMixer vs SF-sj', 'LIF', 'SF-sj')]
    if ST == 'mitbih':
        Q = QS + [(f'(report) {n} vs SF-de', n, 'SF-de') for n in ARMS]
    if bad: sys.exit('STOP: %d problems' % bad)
    for name, a, b in Q:
        cmp_mit(name, T[a], T[b])
    for n, X in T.items():
        print(f'{n:22s} F1 {sum(X[s][0] for s in SEEDS) / 3:.2f} energy {sum(X[s][1] for s in SEEDS) / 3:.2f}')
if bad: sys.exit('STOP: %d problems' % bad)
