# recheck_len.py (2026-11-17, exp/f-lif-pop-v3; PROJECT_LOG input-length test): independent recomputation of the pre-registered comparisons.
# Reads raw.txt and energy_recount.csv only; does not import summarize_len / summarize_l2 / summarize_ms. Fails closed on missing, ambiguous or non-finite rows.
# usage (from cinc_mixer_v1/): python -B recheck_len.py 336 [--sha <pre-registration commit>] [--dir <folder with raw.txt / energy_recount.csv of the new runs> (dry runs)]
import collections, csv, math, os, sys
L = sys.argv[1] if len(sys.argv) > 1 else sys.exit('usage: recheck_len.py <L> [--sha <commit>]')
SHA = sys.argv[sys.argv.index('--sha') + 1] if '--sha' in sys.argv else None
R = os.path.dirname(os.path.abspath(__file__)) + '/results/'
NEW = f'flif-v1-ett-spkmux-len{L}'
def rows(p):
    return [dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t) for ln in open(p) if ln.startswith('RESULT')]
LRM = {'ETTh2': '0.0005', 'ETTm2': '0.0001'}; LRS = {'ETTh2': '0.0005', 'ETTm2': '0.00025'}; LYS = {'ETTh2': '1', 'ETTm2': '2'}
CELLS = [('ETTh2', '96'), ('ETTh2', '720'), ('ETTm2', '96'), ('ETTm2', '720')]; SEEDS = ['42', '7', '2026']
def tag(m, ds, LL):
    t = {'LIF': f'spkmx_lif_muxlin_iand_lr{LRM[ds]}_L1', 'POP': f'spkmx_pop_lif_muxlin_iand_lr{LRM[ds]}_L1', 'SF': f'spk_ssa_lr{LRS[ds]}_L{LYS[ds]}'}[m]
    return t + ('' if LL == '96' else f'_s{LL}')
SRC = {('LIF', '96'): ['flif-v1-ett-spkmux-tune'], ('POP', '96'): ['flif-v1-ett-spkmux-stft'], ('SF', '96'): ['flif-v1-ett-spkmux-tune', 'flif-v1-ett-spkmux'],
       ('LIF', L): [NEW], ('POP', L): [NEW], ('SF', L): [NEW]}
NEWDIR = sys.argv[sys.argv.index('--dir') + 1] if '--dir' in sys.argv else R + NEW
new = rows(NEWDIR + '/raw.txt')
skip = sum(ln.startswith('SKIP') for ln in open(NEWDIR + '/raw.txt'))
keys = collections.Counter((r['ds'], r['pl'], r['tag'], r['seed']) for r in new)
print('new rows', len(new), 'unique', len(keys), 'SKIP', skip, 'sha', dict(collections.Counter(r['sha'] for r in new)))
bad = int(len(new) != 36 or len(keys) != 36 or skip != 0 or (SHA is not None and any(r['sha'] != SHA for r in new)))
T = {}
for (m, LL), suites in SRC.items():
    for ds, pl in CELLS:
        for s in SEEDS:
            t = tag(m, ds, LL); hit = []
            for su in suites:
                # the ETTh2 input-96 Spikformer-style baseline has its own files (no tag field)
                files = [((NEWDIR if su == NEW else R + su) + '/raw.txt', (NEWDIR if su == NEW else R + su) + '/energy_recount.csv')]
                if su == 'flif-v1-ett-spkmux': files = [(R + su + '/baseline_spk_ssa_raw.txt', R + su + '/baseline_spk_ssa_energy.csv')]
                for rp, ep in files:
                    en = collections.defaultdict(list)
                    for e in csv.DictReader(open(ep)):
                        en[(e['ds'], e['pl'], e.get('tag', ''), e['seed'])].append(e)
                    for r in rows(rp):
                        if r['ds'] != ds or r['pl'] != pl or r['seed'] != s: continue
                        if 'tag' in r:
                            if r['tag'] != t: continue
                            es = en.get((ds, pl, t, s), [])
                        else:
                            if not (m == 'SF' and r['cond'] == 'spk_ssa' and ds == 'ETTh2'): continue
                            es = en.get((ds, pl, '', s), [])
                        hit.append((r, es))
            if len(hit) != 1 or len(hit[0][1]) != 1: print('MISSING/AMBIG', m, LL, ds, pl, s, len(hit)); bad += 1; continue
            r, e = hit[0][0], hit[0][1][0]
            v = dict(mse=float(r['mse']), mae=float(r['mae']), val=float(r['val_loss_best']), e=float(e['total_fft_uJ_corr']))
            if not all(math.isfinite(x) for x in v.values()) or abs(float(e['mse_recomputed']) - v['mse']) > 1e-5: print('BAD', m, LL, ds, pl, s, v); bad += 1; continue
            T[(m, LL, ds, pl, s)] = v
if bad: sys.exit('STOP: %d problems' % bad)
def r_(a, b, ds, pl, s): return 100 * (T[a + (ds, pl, s)]['mse'] - T[b + (ds, pl, s)]['mse']) / T[b + (ds, pl, s)]['mse']
def cmpr(a, b):
    cell = []; low = hi = 0
    for ds, pl in CELLS:
        rs = [r_(a, b, ds, pl, s) for s in SEEDS]; low += sum(x < 0 for x in rs); hi += sum(x > 0 for x in rs); cell.append(sum(rs) / 3)
    Rm = sum(cell) / 4
    v = 'better' if Rm <= -1 and low >= 8 else 'worse' if Rm >= 1 and hi >= 8 else 'no clear difference'
    er = sum((sum(T[a + (ds, pl, s)]['e'] for s in SEEDS) / sum(T[b + (ds, pl, s)]['e'] for s in SEEDS)) for ds, pl in CELLS) / 4
    return f'{a[0]} {a[1]} vs {b[0]} {b[1]}: R {Rm:+.2f} lower {low}/12 {v} energy {er:.2f} cells ' + ', '.join(f'{c:+.2f}' for c in cell)
def gap(x, y):
    cell = []; neg = pos = 0
    for ds, pl in CELLS:
        g = [r_((x, L), (y, L), ds, pl, s) - r_((x, '96'), (y, '96'), ds, pl, s) for s in SEEDS]
        neg += sum(z < 0 for z in g); pos += sum(z > 0 for z in g); cell.append(sum(g) / 3)
    G = sum(cell) / 4
    v = f'toward {x}' if G <= -1 and neg >= 8 else f'toward {y}' if G >= 1 and pos >= 8 else 'no change'
    return f'gap {x} vs {y}: G {G:+.2f} negative {neg}/12 {v} cells ' + ', '.join(f'{c:+.2f}' for c in cell)
for a, b in [(('POP', L), ('LIF', L)), (('POP', L), ('SF', L)), (('LIF', L), ('SF', L)), (('LIF', L), ('LIF', '96')), (('POP', L), ('POP', '96')), (('SF', L), ('SF', '96')),
             (('POP', '96'), ('LIF', '96')), (('POP', '96'), ('SF', '96')), (('LIF', '96'), ('SF', '96'))]:
    print(cmpr(a, b))
for x, y in (('POP', 'LIF'), ('POP', 'SF'), ('LIF', 'SF')): print(gap(x, y))
for m in ('LIF', 'POP', 'SF'):
    for LL in ('96', L):
        ms = [sum(T[(m, LL, ds, pl, s)]['mse'] for s in SEEDS) / 3 for ds, pl in CELLS]
        print(f'{m:4s} {LL:>4s}', ' '.join(f'{z:.4f}' for z in ms), f'mean {sum(ms)/4:.4f}', 'energy %.2f' % (sum(sum(T[(m, LL, ds, pl, s)]['e'] for s in SEEDS) / 3 for ds, pl in CELLS) / 4),
              'val %.5f' % (sum(T[(m, LL, ds, pl, s)]['val'] for ds, pl in CELLS for s in SEEDS) / 12))
