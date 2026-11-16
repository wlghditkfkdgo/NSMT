# recheck_stft.py (2026-11-17, exp/f-lif-pop-v3; PROJECT_LOG STFT band results): independent recomputation of the pre-registered comparisons of the STFT band test.
# Reads raw.txt and energy_recount.csv only; does not import summarize_stft / summarize_l2 / summarize_ms. Fails closed on missing, ambiguous or non-finite rows.
# usage (from cinc_mixer_v1/): python -B recheck_stft.py ett|mitbih
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) + '/results/'
TASK = sys.argv[1] if len(sys.argv) > 1 else sys.exit('usage: recheck_stft.py ett|mitbih')
if TASK == 'ett':
    import csv, math, sys, collections
    R = HERE
    def rows(p):
        out = []
        for ln in open(p):
            if ln.startswith('RESULT'):
                d = dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t); out.append(d)
        return out
    new = rows(R + 'flif-v1-ett-spkmux-stft/raw.txt'); old = rows(R + 'flif-v1-ett-spkmux-tune/raw.txt')
    LR = {'ETTh2': '0.0005', 'ETTm2': '0.0001'}
    CELLS = [('ETTh2', '96'), ('ETTh2', '720'), ('ETTm2', '96'), ('ETTm2', '720')]; SEEDS = ['42', '7', '2026']
    def tag(ds, nrn, band, tx):
        return f'spkmx_{nrn}_muxlin_iand_lr{LR[ds]}_L1' + ('' if band == 'none' else '_bd' + band) + ('_txtoken' if tx == 'token' else '')
    FORMS = {'LIF rep': ('lif', 'none', 'rep'), 'LIF token': ('lif', 'none', 'token'), 'A rep': ('lif', 'group', 'rep'), 'A token': ('lif', 'group', 'token'),
             'POP rep': ('pop_lif', 'none', 'rep'), 'POP token': ('pop_lif', 'none', 'token'), 'B rep': ('pop_lif', 'member', 'rep'), 'B token': ('pop_lif', 'member', 'token')}
    en = {}
    for f in ('flif-v1-ett-spkmux-stft', 'flif-v1-ett-spkmux-tune'):
        for r in csv.DictReader(open(R + f + '/energy_recount.csv')):
            k = (r['ds'], r['pl'], r['tag'], r['seed'])
            en.setdefault(k, []).append(float(r['total_fft_uJ_corr']))
    T = {}; bad = 0
    keys = collections.Counter((r['ds'], r['pl'], r['tag'], r['seed']) for r in new)
    print('new rows', len(new), 'unique', len(keys), 'dups', sum(v > 1 for v in keys.values()), 'sha', collections.Counter(r['sha'] for r in new))
    for name, (nrn, band, tx) in FORMS.items():
        src = old if name == 'LIF rep' else new
        for ds, pl in CELLS:
            for s in SEEDS:
                t = tag(ds, nrn, band, tx)
                m = [r for r in src if r['ds'] == ds and r['pl'] == pl and r['tag'] == t and r['seed'] == s]
                # tune suite may hold the same key more than once; require identical mse
                if not m or len({r['mse'] for r in m}) != 1: print('MISSING/AMBIG', name, ds, pl, s, len(m)); bad += 1; continue
                e = en.get((ds, pl, t, s))
                if not e or len({round(x, 6) for x in e}) != 1: print('ENERGY MISSING/AMBIG', name, ds, pl, s, e); bad += 1; continue
                v = dict(mse=float(m[0]['mse']), mae=float(m[0]['mae']), val=float(m[0]['val_loss_best']), e=e[0])
                if not all(math.isfinite(x) for x in v.values()): print('NONFINITE', name, ds, pl, s); bad += 1; continue
                T[(name, ds, pl, s)] = v
    if bad: sys.exit('STOP: %d problems' % bad)
    def cmpr(a, b):
        cell = []; low = 0
        for ds, pl in CELLS:
            rs = [100 * (T[(a, ds, pl, s)]['mse'] - T[(b, ds, pl, s)]['mse']) / T[(b, ds, pl, s)]['mse'] for s in SEEDS]
            low += sum(x < 0 for x in rs); cell.append(sum(rs) / 3)
        Rm = sum(cell) / 4
        hi = sum(T[(a, ds, pl, s)]['mse'] > T[(b, ds, pl, s)]['mse'] for ds, pl in CELLS for s in SEEDS)
        v = 'better' if Rm <= -1 and low >= 8 else 'worse' if Rm >= 1 and hi >= 8 else 'no clear difference'
        ea = sum(T[(a, ds, pl, s)]['e'] for ds, pl in CELLS for s in SEEDS) / 12; eb = sum(T[(b, ds, pl, s)]['e'] for ds, pl in CELLS for s in SEEDS) / 12
        return f'{a} vs {b}: R {Rm:+.2f} lower {low}/12 higher {hi}/12 {v} energy {ea/eb:.2f} cells ' + ', '.join(f'{c:+.2f}' for c in cell)
    for a, b in [('A rep', 'LIF rep'), ('B rep', 'POP rep'), ('A rep', 'B rep'), ('A token', 'LIF token'), ('B token', 'POP token'), ('A token', 'B token'),
                 ('LIF token', 'LIF rep'), ('A token', 'A rep'), ('POP token', 'POP rep'), ('B token', 'B rep'), ('POP rep', 'LIF rep'), ('POP token', 'LIF token')]:
        print(cmpr(a, b))
    for f in FORMS:
        ms = [sum(T[(f, ds, pl, s)]['mse'] for s in SEEDS) / 3 for ds, pl in CELLS]
        print(f'{f:10s}', ' '.join(f'{m:.4f}' for m in ms), f'mean {sum(ms)/4:.4f}', 'energy %.2f' % (sum(T[(f, ds, pl, s)]['e'] for ds, pl in CELLS for s in SEEDS) / 12),
              'val %.5f' % (sum(T[(f, ds, pl, s)]['val'] for ds, pl in CELLS for s in SEEDS) / 12))
elif TASK == 'mitbih':
    import csv, math, sys, collections
    R = HERE
    def rows(p):
        return [dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t) for ln in open(p) if ln.startswith('RESULT')]
    new = rows(R + 'flif-v1-mitbih-spkmux-stft/raw.txt'); old = rows(R + 'flif-v1-mitbih-spkmux-tune/raw.txt')
    SEEDS = ['42', '7', '2026']
    def tag(nrn, band, tx): return f'spkmx_{nrn}_muxlin_add_lr0.02_L2' + ('' if band == 'none' else '_bd' + band) + ('_txtoken' if tx == 'token' else '')
    FORMS = {'LIF rep': ('lif', 'none', 'rep'), 'LIF token': ('lif', 'none', 'token'), 'A rep': ('lif', 'group', 'rep'), 'A token': ('lif', 'group', 'token'),
             'POP rep': ('pop_lif', 'none', 'rep'), 'POP token': ('pop_lif', 'none', 'token'), 'B rep': ('pop_lif', 'member', 'rep'), 'B token': ('pop_lif', 'member', 'token')}
    en = collections.defaultdict(list)
    for f in ('flif-v1-mitbih-spkmux-stft', 'flif-v1-mitbih-spkmux-tune'):
        for r in csv.DictReader(open(R + f + '/energy_recount.csv')):
            if r['val_fold'] == '-1': en[(r['tag'], r['seed'])].append((float(r['total_fft_uJ_corr']), float(r['f1_logged']), float(r['f1_recomputed'])))
    keys = collections.Counter((r['tag'], r['seed']) for r in new)
    print('new rows', len(new), 'unique', len(keys), 'sha', collections.Counter(r['sha'] for r in new), 'epochs_run', collections.Counter(r['epochs_run'] for r in new), 'val_fold', collections.Counter(r['val_fold'] for r in new), 'n_test', collections.Counter(r['n_test'] for r in new))
    T = {}; bad = 0
    for name, (nrn, band, tx) in FORMS.items():
        src = old if name == 'LIF rep' else new
        for s in SEEDS:
            t = tag(nrn, band, tx)
            m = [r for r in src if r['tag'] == t and r['seed'] == s and r['val_fold'] == '-1']
            e = en.get((t, s))
            if len(m) != 1 or not e or len(e) != 1: print('MISSING/AMBIG', name, s, len(m), e); bad += 1; continue
            f1 = float(m[0]['f1'])
            if not (math.isfinite(f1) and math.isfinite(e[0][0])) or abs(e[0][1] - f1) > 1e-6 or abs(e[0][2] - f1) > 5e-7: print('BAD', name, s, f1, e); bad += 1; continue
            T[(name, s)] = (100 * f1, e[0][0])
    if bad: sys.exit('STOP: %d problems' % bad)
    def cmpr(a, b):
        d = [T[(a, s)][0] - T[(b, s)][0] for s in SEEDS]; m = sum(d) / 3; hi = sum(x > 0 for x in d); lo = sum(x < 0 for x in d)
        v = 'better' if m >= 0.5 and hi >= 2 else 'worse' if m <= -0.5 and lo >= 2 else 'no clear difference'
        ea = sum(T[(a, s)][1] for s in SEEDS) / 3; eb = sum(T[(b, s)][1] for s in SEEDS) / 3
        return f'{a} vs {b}: {m:+.2f} higher {hi}/3 ' + ', '.join(f'{x:+.2f}' for x in d) + f' {v} energy {ea/eb:.2f}'
    for a, b in [('A rep', 'LIF rep'), ('B rep', 'POP rep'), ('A rep', 'B rep'), ('A token', 'LIF token'), ('B token', 'POP token'), ('A token', 'B token'),
                 ('LIF token', 'LIF rep'), ('A token', 'A rep'), ('POP token', 'POP rep'), ('B token', 'B rep'), ('POP rep', 'LIF rep'), ('POP token', 'LIF token')]:
        print(cmpr(a, b))
    for f in FORMS: print(f'{f:10s} F1 {sum(T[(f, s)][0] for s in SEEDS)/3:.2f} energy {sum(T[(f, s)][1] for s in SEEDS)/3:.2f}')
else: sys.exit('usage: recheck_stft.py ett|mitbih')
