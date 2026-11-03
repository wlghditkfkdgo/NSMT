"""H1 head results (2026-11-03, exp/f-lif-pop-v3; PROJECT_LOG H1 pre-registration; head_mem.py). Per task, against the tuned final
runs of that task's tuning (the control rows are selected with that task's own summarizer, so they are exactly the rows of its
tuning verdicts):
    G1 muxlin+H1 vs muxlin            does H1 help muxlin
    G2 Spikformer-style+H1 vs Spikformer-style
    G3 muxlin+H1 vs Spikformer-style+H1  (criterion (2))
    report only: G1 gain - G2 gain per seed (and cell / dataset)
Verdict rules = the task's rules: CinC (CinC score) / MIT-BIH (macro-F1): mean >= +0.5 %p and >= 2 of 3 seeds; ETT (test MSE,
ETTh2-96 + ETTm2-96): R <= -1 % and >= 4 of 6 pairs lower (summarize_ett_tune.compare); AD (AUC-PR): R >= +2 % and >= 6 of 9 pairs
higher (summarize_ad_tune.compare). Energy: CinC total_fft_int_uJ (else total_fft_uJ), MIT-BIH / ETT energy_recount.csv
total_fft_uJ_corr, AD total_fft_uJ_corr. Fail closed: a missing run, a non-finite metric or a missing energy -> incomplete.
usage: python summarize_hm.py --task cinc|ett|mitbih|ad   ->  results/flif-v1-headmem/summary_<task>.md
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_rhythm import parse                                        # noqa: E402

RES = os.path.join(HERE, 'results')
OUT = os.path.join(RES, 'flif-v1-headmem')
SEEDS = (42, 7, 2026)


def hm_raw(task):
    p = os.path.join(RES, f'flif-v1-headmem-{task}', 'raw.txt')
    rows = parse(p)
    if any('_hm' not in str(r.get('tag', '')) for r in rows):
        raise SystemExit(f'{p}: a row without the _hm tag')
    return rows


def hm_energy(task, keyf):
    p = os.path.join(RES, f'flif-v1-headmem-{task}', 'energy_recount.csv')
    if not os.path.exists(p):
        raise SystemExit(f'{p} missing (run the energy recount first)')
    return {keyf(e): float(e['total_fft_uJ_corr']) for e in csv.DictReader(open(p))}


def finite(rows, key):
    return rows is not None and all(math.isfinite(float(x[key])) and math.isfinite(float(x['E'])) for x in rows)


def cls_compare(A, B, key, crit):
    if not (finite(A, key) and finite(B, key)):
        return None
    d = [100 * (float(a[key]) - float(b[key])) for a, b in zip(A, B)]
    m = float(np.mean(d))
    v = 'better' if m >= 0.5 and sum(z > 0 for z in d) >= 2 else 'worse' if m <= -0.5 and sum(z < 0 for z in d) >= 2 else 'no clear difference'
    er = float(np.mean([float(a['E']) for a in A])) / float(np.mean([float(b['E']) for b in B]))
    c2 = ('met' if v == 'better' or (m >= -0.5 and er <= 0.5) else 'not met') if crit else ''
    return dict(m=m, d=d, verdict=v, energy=er, crit=c2)


def cls_table(L, groups, key, unit_rows):
    L += ['| question | mean %p | per seed (42, 7, 2026) | verdict | energy A/B | criterion (2) |', '|---|---:|---|---|---:|---|']
    res = {}
    for q, A, B, crit in groups:
        c = cls_compare(A, B, key, crit); res[q.split(' ')[0]] = c
        L.append(f'| {q} | incomplete | | incomplete | | |' if c is None else
                 f"| {q} | {c['m']:+.2f} | {', '.join(f'{z:+.2f}' for z in c['d'])} | {c['verdict']} | {c['energy']:.2f} | {c['crit']} |")
    if res.get('G1') and res.get('G2'):
        gd = [a - b for a, b in zip(res['G1']['d'], res['G2']['d'])]
        L += ['', f"Report only — G1 gain − G2 gain per seed: {', '.join(f'{z:+.2f}' for z in gd)} (mean {np.mean(gd):+.2f} %p)"]
    L += [''] + unit_rows
    return res


def task_cinc(L):
    import summarize_cinc_tune as sc
    D = sc.load()
    sel = {s: json.load(open(os.path.join(sc.TUNE, f'selected_{s}.json')))['selected'] for s in ('lr', 'layers', 'res')}
    def cfg(b):
        ck = sc.CONDKEY[b]
        return ('-', sel['lr'][ck], sc.L0) if b == sc.REF else (sel['res'][ck], sel['lr'][ck], sel['layers'][ck])
    H = {}
    for r in hm_raw('cinc'):
        base, res = sc.split(r['cond'])
        k = (base, res, float(r['lr']), int(r['layers']), int(r['seed']))
        if k in H:
            raise SystemExit(f'duplicate H1 run {k}')
        r['E'] = float(r['total_fft_int_uJ']) if 'total_fft_int_uJ' in r else float(r['total_fft_uJ'])
        H[k] = r
    c_m, c_r = sc.seeds(D, sc.MUX, cfg(sc.MUX)), sc.seeds(D, sc.REF, cfg(sc.REF))
    h_m, h_r = sc.seeds(H, sc.MUX, cfg(sc.MUX)), sc.seeds(H, sc.REF, cfg(sc.REF))
    L += [f'Settings (res, lr, mixer layers): muxlin {cfg(sc.MUX)}, Spikformer-style {cfg(sc.REF)}; CinC fold 0, test CinC score.', '']
    rows = ['| condition | CinC score % | macro-F1 % | F1 class 0 / 1 / 2 / 3 % | energy µJ | best epoch:val |', '|---|---:|---:|---|---:|---|']
    for name, rr in (('Spikformer-style', c_r), ('Spikformer-style + H1', h_r), ('muxlin', c_m), ('muxlin + H1', h_m)):
        if rr is None:
            rows.append(f'| {name} | — | | | | |'); continue
        rows.append(f"| {name} | {np.mean([100 * x['score'] for x in rr]):.2f} ± {np.std([100 * x['score'] for x in rr]):.2f} | "
                    f"{np.mean([100 * x['macro_f1'] for x in rr]):.2f} | " + ' / '.join(f"{np.mean([100 * x[f'f1_{i}'] for x in rr]):.2f}" for i in range(4))
                    + f" | {np.mean([x['E'] for x in rr]):.2f} | {', '.join(str(x.get('best_epoch_val', '')) for x in rr)} |")
    return cls_table(L, (('G1 muxlin+H1 vs muxlin', h_m, c_m, False), ('G2 Spikformer-style+H1 vs Spikformer-style', h_r, c_r, False),
                         ('G3 muxlin+H1 vs Spikformer-style+H1', h_m, h_r, True)), 'score', rows)


def task_mitbih(L):
    import summarize_mitbih_tune as sm
    D = sm.load()
    sel = {s: json.load(open(os.path.join(sm.TUNE, f'selected_{s}.json')))['selected'] for s in ('lr', 'layers', 'res')}
    en = hm_energy('mitbih', lambda e: (e['cond'], e.get('tag', ''), int(e['seed'])))
    H = {}
    for r in hm_raw('mitbih'):
        base, res = sm.split(r['cond'])
        k = (base, res, float(r['lr']), int(r['layers']), int(r['val_fold']), int(r['seed']))
        if k in H:
            raise SystemExit(f'duplicate H1 run {k}')
        r['E'] = en.get((r['cond'], str(r['tag']), int(r['seed'])), float('nan'))
        H[k] = r
    c_m, c_r = sm.seeds(D, sm.MUX, sm.tuned_cfg(sm.MUX, sel), -1), sm.seeds(D, sm.REF, sm.tuned_cfg(sm.REF, sel), -1)
    h_m, h_r = sm.seeds(H, sm.MUX, sm.tuned_cfg(sm.MUX, sel), -1), sm.seeds(H, sm.REF, sm.tuned_cfg(sm.REF, sel), -1)
    L += [f'Settings (res, lr, mixer layers): muxlin {sm.tuned_cfg(sm.MUX, sel)}, Spikformer-style {sm.tuned_cfg(sm.REF, sel)}; '
          'test macro-F1 of the last-epoch weights.', '']
    rows = ['| condition | macro-F1 % | F1 N / S / V / F / Q % | energy µJ |', '|---|---:|---|---:|']
    for name, rr in (('Spikformer-style', c_r), ('Spikformer-style + H1', h_r), ('muxlin', c_m), ('muxlin + H1', h_m)):
        if rr is None:
            rows.append(f'| {name} | — | | |'); continue
        rows.append(f"| {name} | {np.mean([100 * x['f1'] for x in rr]):.2f} ± {np.std([100 * x['f1'] for x in rr]):.2f} | "
                    + ' / '.join(f"{np.mean([100 * x[f'f1_{i}'] for x in rr]):.2f}" for i in range(5)) + f" | {np.mean([x['E'] for x in rr]):.2f} |")
    return cls_table(L, (('G1 muxlin+H1 vs muxlin', h_m, c_m, False), ('G2 Spikformer-style+H1 vs Spikformer-style', h_r, c_r, False),
                         ('G3 muxlin+H1 vs Spikformer-style+H1', h_m, h_r, True)), 'f1', rows)


def gain_diff(L, cells, g1, g2, key, sign=1):
    """report only: per-pair relative change of G1 minus that of G2."""
    out = []
    for c in cells:
        a1, b1, a2, b2 = g1[0](*c), g1[1](*c), g2[0](*c), g2[1](*c)
        if None in (a1, b1, a2, b2):
            return
        out += [100 * (float(x[key]) - float(y[key])) / float(y[key]) - 100 * (float(u[key]) - float(v[key])) / float(v[key])
                for x, y, u, v in zip(a1, b1, a2, b2)]
    L += ['', f"Report only — G1 relative change − G2 relative change per pair ({key}): {', '.join(f'{z:+.2f}' for z in out)} "
          f'(mean {np.mean(out):+.2f} %)']


def task_ett(L):
    import summarize_ett_tune as se
    D = se.load()
    sel = {s: json.load(open(os.path.join(se.TUNE, f'selected_{s}.json')))['selected'] for s in ('lr', 'layers', 'res')}
    ck, rk = se.CONDKEY[se.MUX], se.CONDKEY[se.REF]
    en = hm_energy('ett', lambda e: (e['cond'], e.get('tag', ''), e['ds'], int(e['pl']), int(e['seed'])))
    H = {}
    for r in hm_raw('ett'):
        base, res = se.split(r['cond'])
        k = (base, res, r['ds'], int(r['pl']), float(r['lr']), int(r['layers']), int(r['seed']))
        if k in H:
            raise SystemExit(f'duplicate H1 run {k}')
        r['val'] = float(str(r['best_epoch_val_epochs']).split(':')[1])
        r['E'] = en.get((r['cond'], str(r['tag']), r['ds'], int(r['pl']), int(r['seed'])), float('nan'))
        H[k] = r
    mres, mlr, mlay = ({ds: sel[s][ck][ds] for ds in se.DSS} for s in ('res', 'lr', 'layers'))
    rlr = {ds: sel['lr'][rk][ds] for ds in se.DSS}
    c_m, c_r = se.rows(D, se.MUX, mres, mlr, mlay), se.rows(D, se.REF, '-', rlr, se.L0)
    h_m, h_r = se.rows(H, se.MUX, mres, mlr, mlay), se.rows(H, se.REF, '-', rlr, se.L0)
    cells = (('ETTh2', 96), ('ETTm2', 96))
    L += [f'Settings: muxlin res {mres} lr {mlr} layers {mlay}; Spikformer-style lr {rlr}; cells {cells}; test MSE of the best-validation '
          'weights; r = 100 (MSE_A − MSE_B) / MSE_B.', '',
          '| question | R (MSE, %) | pairs lower | verdict | energy A/B | criterion (2) | R on MAE |', '|---|---:|---|---|---:|---|---:|']
    det = ['', '| comparison | per-cell mean r (seeds 42, 7, 2026) |', '|---|---|']
    res = {}
    for q, A, B, crit in (('G1 muxlin+H1 vs muxlin', h_m, c_m, False), ('G2 Spikformer-style+H1 vs Spikformer-style', h_r, c_r, False),
                          ('G3 muxlin+H1 vs Spikformer-style+H1', h_m, h_r, True)):
        c = se.compare(None, A, B, cells)
        if c is not None and not (all(math.isfinite(float(x['E'])) for cc in cells for x in A(*cc) + B(*cc))):
            c = None                                                     # fail closed: an energy is missing
        res[q.split(' ')[0]] = c
        if c is None:
            L.append(f'| {q} | | | incomplete | | | |'); continue
        L.append(f"| {q} | {c['R']:+.2f} | {c['lo']}/{c['n']} | {c['verdict']} | {c['energy']:.2f} | {c['crit'] if crit else ''} | {c['R_mae']:+.2f} |")
        det.append(f"| {q.split(' ')[0]} | {' ; '.join(c['cells'])} |")
    L += det
    gain_diff(L, cells, (h_m, c_m), (h_r, c_r), 'mse')
    L += ['', '| condition | ' + ' | '.join(f'{d} H{p} MSE / MAE / energy µJ' for d, p in cells) + ' |', '|---|---|---|']
    for name, f in (('Spikformer-style', c_r), ('Spikformer-style + H1', h_r), ('muxlin', c_m), ('muxlin + H1', h_m)):
        cs = []
        for d, p in cells:
            v = f(d, p)
            cs.append('—' if v is None else f"{np.mean([x['mse'] for x in v]):.4f} / {np.mean([x['mae'] for x in v]):.4f} / "
                      f"{np.mean([float(x['E']) for x in v]):.2f}")
        L.append(f'| {name} | ' + ' | '.join(cs) + ' |')
    return res


def task_ad(L):
    import summarize_ad_tune as sa
    D = sa.load()
    sel = {s: json.load(open(os.path.join(sa.TUNE, f'selected_{s}.json')))['selected'] for s in ('lr', 'layers', 'res')}
    ck = sa.CONDKEY[sa.MUX]
    H = {}
    for r in hm_raw('ad'):
        base, res = sa.split(r['cond'])
        k = (base, res, r['ds'], float(r['lr']), int(r['layers']), int(r['holdout']), int(r['seed']))
        if k in H:
            raise SystemExit(f'duplicate H1 run {k}')
        H[k] = r
    mcfg = lambda ds: (sel['res'][ck][ds], sel['lr'][ck][ds], sel['layers'][ck][ds])
    rcfg = lambda ds: ('-', sel['lr']['spk_ssa'][ds], sa.L0)
    c_m, c_r = (lambda ds: sa.seeds(D, sa.MUX, ds, mcfg(ds), 0)), (lambda ds: sa.seeds(D, sa.REF, ds, rcfg(ds), 0))
    h_m, h_r = (lambda ds: sa.seeds(H, sa.MUX, ds, mcfg(ds), 0)), (lambda ds: sa.seeds(H, sa.REF, ds, rcfg(ds), 0))
    L += ['Settings: ' + json.dumps({ds: dict(muxlin=mcfg(ds), spk_ssa=rcfg(ds)) for ds in sa.DSS}) + '; whole training file, test AUC-PR; '
          'r = 100 (AUC-PR_A − AUC-PR_B) / AUC-PR_B.', '',
          '| question | R (AUC-PR, %) | pairs higher | verdict | energy A/B | criterion (2) |', '|---|---:|---|---|---:|---|']
    det = ['', '| comparison | per-dataset mean r (seeds 42, 7, 2026) |', '|---|---|']
    res = {}
    for q, A, B, crit in (('G1 muxlin+H1 vs muxlin', h_m, c_m, False), ('G2 Spikformer-style+H1 vs Spikformer-style', h_r, c_r, False),
                          ('G3 muxlin+H1 vs Spikformer-style+H1', h_m, h_r, True)):
        c = sa.compare(A, B); res[q.split(' ')[0]] = c
        if c is None:
            L.append(f'| {q} | | | incomplete | | |'); continue
        L.append(f"| {q} | {c['R']:+.2f} | {c['hi']}/{c['n']} | {c['verdict']} | {c['energy']:.2f} | {c['crit'] if crit else ''} |")
        det.append(f"| {q.split(' ')[0]} | {' ; '.join(c['cells'])} |")
    L += det
    gain_diff(L, [(ds,) for ds in sa.DSS], (h_m, c_m), (h_r, c_r), 'aucpr')
    for ds in sa.DSS:
        L += ['', f'## {ds} (seed mean ± std, %; report)', '',
              '| condition | AUC-PR | AUC-ROC | raw F1 | point-adjusted F1 | energy µJ | best epoch |', '|---|---:|---:|---:|---:|---:|---:|']
        for name, f in (('Spikformer-style', c_r), ('Spikformer-style + H1', h_r), ('muxlin', c_m), ('muxlin + H1', h_m)):
            rr = f(ds)
            if rr is None:
                L.append(f'| {name} | — | | | | | |'); continue
            ms = lambda k: f"{np.mean([100 * x[k] for x in rr]):.2f} ± {np.std([100 * x[k] for x in rr]):.2f}"
            L.append(f"| {name} | {ms('aucpr')} | {ms('aucroc')} | {ms('raw_f1')} | {ms('pa_f1')} | "
                     f"{np.mean([x['total_fft_uJ_corr'] for x in rr]):.2f} | {np.mean([x['best_epoch'] for x in rr]):.1f} |")
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--task', choices=['cinc', 'ett', 'mitbih', 'ad'], required=True); a = ap.parse_args()
    L = [f'# H1 head (the head also reads the last neuron layer\'s charge) — {a.task}', '']
    res = {'cinc': task_cinc, 'ett': task_ett, 'mitbih': task_mitbih, 'ad': task_ad}[a.task](L)
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, f'summary_{a.task}.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    if any(v is None for v in res.values()):
        sys.exit(2)                                                      # fail closed: an incomplete comparison stops the driver


if __name__ == '__main__':
    main()
