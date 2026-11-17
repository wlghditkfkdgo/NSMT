# recheck_mx3.py (2026-11-17, exp/f-lif-pop-v3; PROJECT_LOG mixer variants): independent recomputation of the pre-registered comparisons.
# Reads raw.txt and energy_recount.csv only; does not import summarize_mx3 / summarize_l2 / summarize_ms. Fails closed on missing, ambiguous or non-finite rows.
# usage (from cinc_mixer_v1/): python -B recheck_mx3.py ett|mitbih [--sha <commit>] [--dir <folder with the new raw.txt / energy_recount.csv> (dry runs)] [--res <results root>]
import collections, csv, math, os, sys
TASK = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ('ett', 'mitbih') else sys.exit('usage: recheck_mx3.py ett|mitbih [--sha c] [--dir d] [--res r]')
opt = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
SHA, R = opt('--sha'), (opt('--res') or os.path.dirname(os.path.abspath(__file__)) + '/results') + '/'
NEW = opt('--dir') or R + {'ett': 'flif-v1-ett-spkmux-mx3', 'mitbih': 'flif-v1-mitbih-spkmux-mx3'}[TASK]
SEEDS = ['42', '7', '2026']


def rows(p):
    return [dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t) for ln in open(p) if ln.startswith('RESULT')]


new = rows(NEW + '/raw.txt'); skip = sum(ln.startswith('SKIP') for ln in open(NEW + '/raw.txt'))
want = 72 if TASK == 'ett' else 15
keys = collections.Counter((r.get('ds', ''), r.get('pl', ''), r['tag'], r['seed']) for r in new)
print('new rows', len(new), 'unique', len(keys), 'SKIP', skip, 'sha', dict(collections.Counter(r['sha'] for r in new)))
bad = int(len(new) != want or len(keys) != want or skip != 0 or (SHA is not None and any(r['sha'] != SHA for r in new)))
T = {}
if TASK == 'ett':
    LR = {'ETTh2': '0.0005', 'ETTm2': '0.0001'}; LRS = {'ETTh2': '0.0005', 'ETTm2': '0.00025'}; LYS = {'ETTh2': '1', 'ETTm2': '2'}
    CELLS = [('ETTh2', '96'), ('ETTh2', '720'), ('ETTm2', '96'), ('ETTm2', '720')]
    ARMS = {'base': '_L1', 'max': '_L1_mpmax', 'mean': '_L1_mpmean', 'L2': '_L2', 'MS': '_L2_mms', 'subtract': '_L1_mrsubtract', 'zero': '_L1_mrzero'}
    def files(arm):
        if arm == 'SF':
            return [(R + 'flif-v1-ett-spkmux-tune/raw.txt', R + 'flif-v1-ett-spkmux-tune/energy_recount.csv'),
                    (R + 'flif-v1-ett-spkmux/baseline_spk_ssa_raw.txt', R + 'flif-v1-ett-spkmux/baseline_spk_ssa_energy.csv')]
        d = R + 'flif-v1-ett-spkmux-tune' if arm == 'base' else NEW
        return [(d + '/raw.txt', d + '/energy_recount.csv')]
    for arm in list(ARMS) + ['SF']:
        for ds, pl in CELLS:
            for s in SEEDS:
                hit = []
                for rp, ep in files(arm):
                    en = collections.defaultdict(list)
                    for e in csv.DictReader(open(ep)):
                        en[(e['ds'], e['pl'], e.get('tag', ''), e['seed'])].append(e)
                    for r in rows(rp):
                        if r['ds'] != ds or r['pl'] != pl or r['seed'] != s:
                            continue
                        if arm == 'SF':
                            if r.get('cond') != 'spk_ssa': continue
                            if 'tag' in r:
                                if r['tag'] != f'spk_ssa_lr{LRS[ds]}_L{LYS[ds]}' or ds != 'ETTm2': continue
                            elif ds != 'ETTh2': continue
                            es = en.get((ds, pl, r.get('tag', ''), s), [])
                        else:
                            if r.get('tag') != f'spkmx_lif_muxlin_iand_lr{LR[ds]}{ARMS[arm]}': continue
                            es = en.get((ds, pl, r['tag'], s), [])
                        hit.append((r, es))
                if len(hit) != 1 or len(hit[0][1]) != 1:
                    print('MISSING/AMBIG', arm, ds, pl, s, len(hit)); bad += 1; continue
                r, e = hit[0][0], hit[0][1][0]
                v = dict(m=float(r['mse']), mae=float(r['mae']), e=float(e['total_fft_uJ_corr']))
                if not all(math.isfinite(x) for x in v.values()) or abs(float(e['mse_recomputed']) - v['m']) > 1e-5:
                    print('BAD', arm, ds, pl, s, v); bad += 1; continue
                T[(arm, ds, pl, s)] = v
    if bad: sys.exit('STOP: %d problems' % bad)
    def cmpr(a, b):
        cell = []; lo = hi = 0
        for ds, pl in CELLS:
            rs = [100 * (T[(a, ds, pl, s)]['m'] - T[(b, ds, pl, s)]['m']) / T[(b, ds, pl, s)]['m'] for s in SEEDS]
            lo += sum(x < 0 for x in rs); hi += sum(x > 0 for x in rs); cell.append(sum(rs) / 3)
        Rm = sum(cell) / 4
        v = 'better' if Rm <= -1 and lo >= 8 else 'worse' if Rm >= 1 and hi >= 8 else 'no clear difference'
        er = sum(sum(T[(a, ds, pl, s)]['e'] for s in SEEDS) / sum(T[(b, ds, pl, s)]['e'] for s in SEEDS) for ds, pl in CELLS) / 4
        return f'{a} vs {b}: R {Rm:+.2f} lower {lo}/12 {v} energy {er:.2f} cells ' + ', '.join(f'{c:+.2f}' for c in cell)
    for a, b in [('max', 'base'), ('mean', 'base'), ('MS', 'base'), ('MS', 'L2'), ('subtract', 'base'), ('zero', 'base'), ('max', 'mean'), ('L2', 'base'), ('subtract', 'zero')] + \
                [(x, 'SF') for x in ARMS]:
        print(cmpr(a, b))
    for arm in list(ARMS) + ['SF']:
        ms = [sum(T[(arm, ds, pl, s)]['m'] for s in SEEDS) / 3 for ds, pl in CELLS]
        print(f'{arm:9s}', ' '.join(f'{z:.4f}' for z in ms), f'mean {sum(ms) / 4:.4f}', 'energy %.2f' % (sum(sum(T[(arm, ds, pl, s)]['e'] for s in SEEDS) / 3 for ds, pl in CELLS) / 4))
else:
    ARMS = {'base': '', 'max': '_mpmax', 'mean': '_mpmean', 'MS': '_mms', 'subtract': '_mrsubtract', 'zero': '_mrzero', 'SF': None}
    TU = R + 'flif-v1-mitbih-spkmux-tune'
    for arm, suf in ARMS.items():
        d = TU if arm in ('base', 'SF') else NEW
        tag = 'spk_ssa_lr0.005_L2' if arm == 'SF' else f'spkmx_lif_muxlin_add_lr0.02_L2{suf}'
        en = collections.defaultdict(list)
        for e in csv.DictReader(open(d + '/energy_recount.csv')):
            if e['val_fold'] == '-1': en[(e['tag'], e['seed'])].append(e)
        for s in SEEDS:
            m = [r for r in rows(d + '/raw.txt') if r['tag'] == tag and r['seed'] == s and r['val_fold'] == '-1']
            e = en.get((tag, s), [])
            if len(m) != 1 or len(e) != 1:
                print('MISSING/AMBIG', arm, s, len(m), len(e)); bad += 1; continue
            f1 = float(m[0]['f1'])
            if not math.isfinite(f1) or abs(float(e[0]['f1_recomputed']) - f1) > 1e-5:
                print('BAD', arm, s, f1); bad += 1; continue
            T[(arm, s)] = (100 * f1, float(e[0]['total_fft_uJ_corr']))
    if bad: sys.exit('STOP: %d problems' % bad)
    def cmpr(a, b):
        d = [T[(a, s)][0] - T[(b, s)][0] for s in SEEDS]; m = sum(d) / 3; hi = sum(x > 0 for x in d); lo = sum(x < 0 for x in d)
        v = 'better' if m >= .5 and hi >= 2 else 'worse' if m <= -.5 and lo >= 2 else 'no clear difference'
        er = sum(T[(a, s)][1] for s in SEEDS) / sum(T[(b, s)][1] for s in SEEDS)
        return f'{a} vs {b}: {m:+.2f} higher {hi}/3 ' + ', '.join(f'{x:+.2f}' for x in d) + f' {v} energy {er:.2f}'
    for a, b in [('max', 'base'), ('mean', 'base'), ('MS', 'base'), ('subtract', 'base'), ('zero', 'base'), ('max', 'mean'), ('subtract', 'zero')] + \
                [(x, 'SF') for x in ARMS if x != 'SF']:
        print(cmpr(a, b))
    for arm in ARMS:
        print(f'{arm:9s} F1 {sum(T[(arm, s)][0] for s in SEEDS) / 3:.2f} energy {sum(T[(arm, s)][1] for s in SEEDS) / 3:.2f}')
