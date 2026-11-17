"""MODI 2026-11-17: fail-closed checks of a finished input-length pool (docs/MODI_PROJECT_LOG.md), run before the verdicts are read.

For suite flif-v1-ett-len<L> (and, for L = 336, the input-96 control suite flif-v1-ett-len96ctl):
  keys      every (dataset, horizon, condition, seed) exactly once, nothing else
  values    mse, mae, val_loss_best, last_mse, total_fft_uJ, rho_hippo, rho_neo (not Spikformer-style), params, time_s finite
  sha       every RESULT line carries the pre-registration commit;  seq_len as expected
  params    base model == nomem == nshuf in every cell
  logargs   every run's logargs.txt against the input-96 run of the same model / cell / seed (flif-v1-gen-ett for Spikformer-style,
            flif-v1-ett-ms for the base model): the only differences allowed are seq_len and the control flag (paths, date, epoch
            count and options that did not exist when the reference run was made are ignored; the latter are printed)
usage (classification folder): python check_len_results.py --len 336 --sha c9bef7da0
exit code 1 if anything fails.
"""
import argparse, collections, glob, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__)); FC = os.path.join(HERE, '..', 'forecasting')
IGN = {'log_dir', 'tag', 'date', 'time', 'epoch', 'model_params'}
BASES = [f'nsmt4q_{n}_msn' for n in ('pop_lif', 'pop_flif')]


def logargs(d):
    a = {}
    for ln in open(os.path.join(d, 'logargs.txt')):
        m = re.match(r'^(\w+?)-{2,}(.*)$', ln.rstrip('\n'))
        if m:
            a[m.group(1)] = m.group(2).strip()
    return a


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--len', type=int, required=True); ap.add_argument('--sha', required=True)
    a = ap.parse_args()
    exp = {f'flif-v1-ett-len{a.len}': (str(a.len), ['spk_ssa'] + [b + v for b in BASES for v in ('', '_nomem', '_nshuf')])}
    if a.len == 336:
        exp['flif-v1-ett-len96ctl'] = ('96', [b + v for b in BASES for v in ('_nomem', '_nshuf')])
    bad = []; new = set(); nla = 0
    for su, (L, conds) in exp.items():
        rows = [dict(re.findall(r'(\w+)=(\S*)', ln)) for ln in open(os.path.join(FC, 'results', su, 'raw.txt')) if ln.startswith('RESULT')]
        skip = sum('SKIP' in ln for ln in open(os.path.join(FC, 'results', su, 'pool_runs.txt')))
        if skip:
            bad.append(f'{su}: {skip} SKIP lines in pool_runs.txt')
        keys = collections.Counter((r['ds'], r['pl'], r['cond'], r['seed']) for r in rows)
        want = {(ds, pl, c, s) for ds in ('ETTh2', 'ETTm2') for pl in ('96', '720') for c in conds for s in ('42', '7', '2026')}
        if set(keys) != want:
            bad.append(f'{su} keys: missing {sorted(want - set(keys))[:5]} extra {sorted(set(keys) - want)[:5]}')
        if any(v != 1 for v in keys.values()):
            bad.append(f'{su} duplicates {[k for k, v in keys.items() if v != 1]}')
        par = collections.defaultdict(set)
        for r in rows:
            nm = f'{su} {r["cond"]} {r["ds"]} {r["pl"]} {r["seed"]}'
            for k in ('mse', 'mae', 'val_loss_best', 'last_mse', 'total_fft_uJ', 'rho_neo', 'rho_hippo', 'params', 'time_s'):
                try:
                    x = float(r.get(k))
                except (TypeError, ValueError):
                    x = float('nan')
                if not math.isfinite(x) and not (k == 'rho_neo' and r['cond'] == 'spk_ssa'):
                    bad.append(f'{nm} {k}={r.get(k)}')
            if r.get('sha') != a.sha:
                bad.append(f'{nm} sha {r.get("sha")}')
            if r.get('seq_len') != L:
                bad.append(f'{nm} seq_len {r.get("seq_len")}')
            base = re.sub(r'_(nomem|nshuf)$', '', r['cond'])
            par[r['ds'], r['pl'], base].add(r['params'])
            if not os.path.exists(os.path.join(r['dir'], 'logargs.txt')):
                bad.append(f'{nm} logargs missing'); continue
            g = glob.glob(os.path.join(FC, 'log', 'flif-v1-gen-ett' if base == 'spk_ssa' else 'flif-v1-ett-ms', r['ds'], f'p{r["pl"]}', '*',
                                       f'*+{base}+seed{r["seed"]}+*', f'seed{r["seed"]}_*'))
            if len(g) != 1:
                bad.append(f'{nm} reference dir: {len(g)} found'); continue
            x, b = logargs(r['dir']), logargs(g[0]); nla += 1
            new.update((k, v) for k, v in x.items() if k not in b)
            diff = {k: (b[k], x.get(k)) for k in sorted(b) if x.get(k) != b[k] and k not in IGN and '/' not in str(x.get(k)) and '/' not in b[k]}
            e = {}
            if L != '96':
                e['seq_len'] = ('96', L)
            if r['cond'].endswith('_nomem'):
                e['neo_memoryless'] = ('False', 'True')
            if r['cond'].endswith('_nshuf'):
                e['neo_shuffle'] = ('False', 'True')
            if diff != e:
                bad.append(f'{nm} logargs differ {diff}, expected {e}')
            for k, v in (('seq_len', L), ('pred_len', r['pl']), ('seed', r['seed'])):
                if x.get(k) != v:
                    bad.append(f'{nm} logargs {k}={x.get(k)}')
            if base != 'spk_ssa' and x.get('neo_res') != 'ms':
                bad.append(f'{nm} logargs neo_res={x.get("neo_res")}')
        for k, v in sorted(par.items()):
            if len(v) != 1:
                bad.append(f'{su} params differ {k} {sorted(v)}')
            print('params', su, *k, sorted(v))
        print(f'{su}: {len(rows)} RESULT lines, {len(keys)} keys, SKIP {skip}')
    print(f'logargs compared: {nla}; options absent from the reference runs (defaults of later options): {sorted(new)}')
    if bad:
        print('FAILED CHECKS:\n  ' + '\n  '.join(bad)); sys.exit(1)
    print('all checks passed')


if __name__ == '__main__':
    main()
