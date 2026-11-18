# recheck_adsf.py (2026-11-18, exp/f-lif-pop-v3; PROJECT_LOG AD SF-de pre-registration): independent recomputation of the SF-de lr selection
# and the AD comparisons. Reads raw.txt and the 11-03 selected_*.json only; does not import summarize_adsf / summarize_ad_tune. Fails closed on
# missing, duplicate or non-finite rows, SKIP lines and energy_check != ok.
# usage (from cinc_mixer_v1/): python -B recheck_adsf.py select|final [--sha <pre-reg commit>[,<later commit>...]] (rows of flif-v1-ad-sfde)
import json, math, os, subprocess, sys
ST = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ('select', 'final') else sys.exit('usage: recheck_adsf.py select|final [--sha c]')
SHA = sys.argv[sys.argv.index('--sha') + 1].split(',') if '--sha' in sys.argv else None
H = os.path.dirname(os.path.abspath(__file__))
if SHA and len(SHA) > 1:
    for c in SHA[1:]:
        d = subprocess.run(['git', 'diff', '--name-only', SHA[0], c, '--', '*.py', '*.sh'], capture_output=True, text=True, cwd=H)
        if d.returncode != 0 or d.stdout.strip():
            sys.exit(f'STOP: commit {c} changes code against {SHA[0]}: {d.stdout.strip() or d.stderr.strip()}')
        print(f'commit {c}: no .py / .sh change against {SHA[0]}')
NEW, OLD = (os.environ.get('ADSF_DRY') or H + '/results/flif-v1-ad-sfde') + '/', H + '/results/flif-v1-ad-spkmux-tune/'   # ADSF_DRY: dry runs
DSS, SEEDS, LRS = ('MSL', 'SMAP', 'PSM'), ('42', '7', '2026'), ('0.0001', '0.00025', '0.0005', '0.001', '0.002')


def rows(p):
    out = []
    for ln in open(p):
        if ln.startswith('SKIP'):
            sys.exit(f'STOP: SKIP line in {p}: {ln[:100]}')
        if ln.startswith('RESULT'):
            out.append(dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t))
    return out


def pick(rr, **kv):
    v = [r for r in rr if all(r.get(k) == x for k, x in kv.items())]
    if len(v) != 1:
        sys.exit(f'STOP: {len(v)} rows for {kv}')
    return v[0]


N = rows(NEW + 'raw.txt')
if SHA:
    bad = [r['sha'] for r in N if r['sha'] not in SHA]
    if bad:
        sys.exit(f'STOP: rows with sha outside {SHA}: {sorted(set(bad))}')
print(f'flif-v1-ad-sfde: {len(N)} rows, sha {sorted(set(r["sha"] for r in N))}')
if ST == 'select':
    out = {}
    for ds in DSS:
        m = {}
        for lr in LRS:
            v = [float(pick(N, tag=f'spk_ssa_lr{lr}_L2_ho_nalif', ds=ds, seed=s, holdout='1')['val_mse_best']) for s in SEEDS]
            if not all(math.isfinite(x) for x in v):
                sys.exit(f'STOP: non-finite validation MSE {ds} {lr}')
            m[lr] = sum(v) / 3
        sc = {lr: m[lr] / m['0.0005'] for lr in LRS}
        b = min(sc, key=lambda k: sc[k]); b = b if sc[b] < sc['0.0005'] - 1e-12 else '0.0005'
        out[ds] = b
        print(ds, 'selected', b, ' '.join(f'{k} {v:.5f}' for k, v in sc.items()))
    j = json.load(open(NEW + 'selected_lr.json'))['selected']
    print('selected_lr.json agrees:', j == out)
    if j != out:
        sys.exit('STOP: selection differs')
    sys.exit(0)
O = rows(OLD + 'raw.txt')
tl, ty, tr = (json.load(open(OLD + f'selected_{s}.json'))['selected'] for s in ('lr', 'layers', 'res'))
sel = json.load(open(NEW + 'selected_lr.json'))['selected']
g = lambda x: f'{float(x):g}'


def arm(name, ds):
    if name == 'SF-de':
        rr = [pick(N, tag=f'spk_ssa_lr{sel[ds]}_L2_nalif', ds=ds, seed=s, holdout='0') for s in SEEDS]
    elif name == 'SF-sj':
        rr = [pick(O, cond='spk_ssa', tag=f'spk_ssa_lr{g(tl["spk_ssa"][ds])}_L2', ds=ds, seed=s, holdout='0') for s in SEEDS]
    else:
        k = 'spkmx_lif_muxlin_or'
        res, lr, ly = (tr[k][ds], g(tl[k][ds]), ty[k][ds]) if name == 'mux_t' else ('or', '0.0005', 2)
        rr = [r for r in O if r['cond'] == f'spkmx_lif_muxlin_{res}' and r['ds'] == ds and r['holdout'] == '0' and float(r['lr']) == float(lr)
              and int(r['layers']) == int(ly)]
        rr = [pick(rr, seed=s) for s in SEEDS]
    for r in rr:
        if not math.isfinite(float(r['aucpr'])) or r.get('energy_check') != 'ok':
            sys.exit(f'STOP: {name} {ds} bad row')
    return [(float(r['aucpr']), float(r['total_fft_uJ_corr'])) for r in rr]


def cmp(name, a, b, bar):
    dm, pairs, ers = [], [], []
    for ds in DSS:
        A, B = arm(a, ds), arm(b, ds)
        r = [100 * (x[0] - y[0]) / y[0] for x, y in zip(A, B)]
        dm.append(sum(r) / 3); pairs += r
        ers.append((sum(x[1] for x in A) / 3) / (sum(y[1] for y in B) / 3))
    R = sum(dm) / 3; hi = sum(z > 0 for z in pairs); lo = sum(z < 0 for z in pairs); er = sum(ers) / 3
    v = 'better' if R >= 2 and hi >= 6 else 'worse' if R <= -2 and lo >= 6 else 'no clear difference'
    pb = f" paper bar {'met' if R > 0 and er < 1 else 'not met'}" if bar else ''
    print(f'{name}: R {R:+.2f} higher {hi}/9 {v} energy {er:.2f} per dataset ' + ', '.join(f'{d:+.2f}' for d in dm) + pb)


cmp('E1 muxlin(tuned) vs SF-de(tuned)', 'mux_t', 'SF-de', True)
cmp('(report) muxlin(default) vs SF-de(tuned)', 'mux_d', 'SF-de', True)
cmp('(report) SF-de(tuned) vs SF-sj(tuned)', 'SF-de', 'SF-sj', False)
cmp('(report) muxlin(tuned) vs SF-sj(tuned)', 'mux_t', 'SF-sj', False)
for n in ('SF-de', 'SF-sj', 'mux_t', 'mux_d'):
    print(f'{n:6s}', ' '.join(f'{ds} AUC-PR {100 * sum(x[0] for x in arm(n, ds)) / 3:.2f} E {sum(x[1] for x in arm(n, ds)) / 3:.2f}' for ds in DSS))
