"""Dry run of summarize_sfpt16.py (2026-11-23, exp/f-lif-pop-v3; PROJECT_LOG SF-de patch-16 pre-registration). No training.
D1 fake suite = the 12 rows of SF-de patch 8 (flif-v1-ett-sfde, the selected lr / layers) relabelled as patch 16 (tag _pt16_nalif, RESULT
   fields patch=16 emb=64 mlpr blocks=1 wd=0.06 added) with their energy rows relabelled: the summary must give SF-de patch 16 = SF-de
   patch 8 (R 0, 0/12, energy 1), and the main comparison = first chain vs SF-de patch 8 as in the 11-20 summary (R -1.02 %, 11/12, energy
   0.5412).
D2-D5 fail closed: a SKIP line, a missing run, a duplicate row, a missing energy row -> exit 2, no summary written.
usage (from cinc_mixer_v1/): python -B results/check-sfpt16-20261123/check_dry_sf16.py <scratch dir> > results/check-sfpt16-20261123/check_dry_sf16.txt
"""
import csv, json, os, shutil, subprocess, sys
C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SC = sys.argv[1]
LR, LY = {'ETTh2': '0.0005', 'ETTm2': '0.0001'}, {'ETTh2': '1', 'ETTm2': '2'}
MLP = {'ETTh2': '0.5', 'ETTm2': '1'}
src = os.path.join(C, 'results', 'flif-v1-ett-sfde')
fails = 0
def check(lab, cond, det=''):
    global fails; fails += not cond; print(f'{"PASS" if cond else "FAIL"} {lab}{(": " + det) if det else ""}')
lines, keep = [], set()
for l in open(os.path.join(src, 'raw.txt')):
    if not l.startswith('RESULT'):
        continue
    r = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
    ds = r['ds']
    if r.get('cond') != 'spk_ssa' or r.get('tag') != f'spk_ssa_lr{LR[ds]}_L{LY[ds]}_nalif' or r['pl'] not in ('96', '720') or r['seed'] not in ('42', '7', '2026'):
        continue
    new = l.rstrip('\n').replace(f'tag={r["tag"]}', f'tag=spk_ssa_lr{LR[ds]}_L{LY[ds]}_pt16_nalif')
    new = new.replace(f' layers={r["layers"]} ', f' layers={r["layers"]} patch=16 emb=64 mlpr={MLP[ds]} blocks=1 wd=0.06 ')
    lines.append(new); keep.add((r['tag'], ds, r['pl'], r['seed']))
check('D1 12 SF-de patch-8 rows found', len(lines) == 12, str(len(lines)))
erows = [e for e in csv.DictReader(open(os.path.join(src, 'energy_recount.csv'))) if (e['tag'], e['ds'], e['pl'], e['seed']) in keep]
for e in erows:
    e['tag'] = e['tag'].replace('_nalif', '_pt16_nalif')
check('D1 12 energy rows found', len(erows) == 12, str(len(erows)))
def make(name, raw, er):
    d = os.path.join(SC, name); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    open(os.path.join(d, 'raw.txt'), 'w').write(''.join(x + '\n' for x in raw))
    with open(os.path.join(d, 'energy_recount.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(er[0].keys())); w.writeheader(); [w.writerow(x) for x in er]
    return d
def run(d):
    p = subprocess.run([sys.executable, '-B', os.path.join(C, 'results', 'sfde-pt16-20261123', 'summarize_sfpt16.py')], cwd=C,
                       env=dict(os.environ, SF16_DIR=d), capture_output=True, text=True)
    return p.returncode, p.stdout
d = make('D1', lines, erows)
rc, out = run(d)
check('D1 summarizer exit 0', rc == 0, out.strip().splitlines()[-1] if rc else '')
if rc == 0:
    s = json.load(open(os.path.join(d, 'summary.json')))
    c = s['comparisons']
    sp = c['(report) SF-de patch 16 vs SF-de patch 8']
    check('D1 SF-de pt16 (fake) vs pt8: R 0, 0/12 lower, energy 1', abs(sp['R']) < 1e-12 and sp['lo'] == 0 and abs(sp['Es'] - 1) < 1e-12 and abs(sp['Es_syops_only'] - 1) < 1e-12, str(sp))
    m, ref = c['**main** first-chain final vs SF-de patch 16'], c['(report) first-chain final vs SF-de patch 8']
    check('D1 main = first chain vs SF-de pt8 (same numbers)', m == ref, f'{m["R"]:.4f} {m["lo"]} {m["Es"]:.4f}')
    check('D1 main = 11-20 values (R -1.02 %, 11/12, better, energy 0.5412, MAE R -0.59 %)',
          round(m['R'], 2) == -1.02 and m['lo'] == 11 and m['verdict'] == 'better' and round(m['Es'], 4) == 0.5412 and round(m['R_mae'], 2) == -0.59,
          f'{m["R"]:.4f} {m["lo"]} {m["verdict"]} {m["Es"]:.4f} {m["R_mae"]:.4f}')
    pa = s['pass']
    k8 = [k for k in pa if k.startswith('first-chain') and 'patch 8' in k][0]; k16 = [k for k in pa if k.startswith('first-chain') and 'patch 16' in k][0]
    check('D1 first chain passes with either SF-de (identical fake) as on 11-23 21:01', pa[k8]['pass'] and pa[k16] == pa[k8], f'{pa[k8]} / {pa[k16]}')
    ke = [k for k in pa if k.startswith('extension') and 'patch 8' in k][0]
    check('D1 extension fails (2) on energy as on 11-23 21:01', not pa[ke]['(2) Spikformer'] and pa[ke]['(1) SNN'] and pa[ke]['(3) DNN'], str(pa[ke]))
for lab, raw, er in (('D2 SKIP line', lines + ['SKIP(rc=1, no result) ds=ETTh2 pl=96 cond=spk_ssa seed=42 out=x'], erows),
                     ('D3 missing run', lines[1:], erows), ('D4 duplicate row', lines + [lines[0]], erows), ('D5 missing energy row', lines, erows[1:])):
    d = make(lab.split()[0], raw, er)
    rc, out = run(d)
    check(f'{lab}: exit 2, no summary', rc == 2 and not os.path.exists(os.path.join(d, 'summary.md')), (out.strip().splitlines() or [''])[-1])
print('ALL DRY CHECKS PASSED' if fails == 0 else f'{fails} FAILED'); sys.exit(1 if fails else 0)
