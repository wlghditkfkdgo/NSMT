"""MODI 2026-11-23: hyperparameter range 2 (lr x batch size x warmup) -- the grid, the search queue and the validation-only selection
(docs/MODI_PROJECT_LOG.md 2026-11-23, pre-registered rules; user decision 11-23 "기준과 (나)를 둘 다 hyperparameter 조정에 넣고 결정하자").
The test set is never read here.

Models (every neuron SpikeDE, no relative position bias, no aux loss, range-1 training = the runner defaults):
  spk      Spikformer(DE) spkall_lif                       (run_ett_mixer.sh)   current-setting runs: flif-v1-ett-r1 / flif-v1-mitbih-r1
  base_<n> the query base nsmt4q_<n>_msn_sde               (run_ett_dual.sh)                          flif-v1-ett-r1 / flif-v1-mitbih-r1
  na_<n>   (나) nsmt4q_<n>_msn_sde_ncna                    (run_ett_dual.sh)                          flif-v1-ett-nc / flif-v1-mitbih-nc
  n = pop_lif, pop_flif. Every model gets the same 18 combinations.
Grid (current value marked *):
  ETT      lr 0.00025 / 0.0005* / 0.001   x  batch 32 / 64* / 128     x  warmup none* / 1 epoch (linear over the first epoch's steps)
  MIT-BIH  lr 0.005 / 0.01* / 0.02        x  batch 256 / 512* / 1024  x  cosine warmup 6 epochs* / none
Search (seed 42):
  ETT      suite flif-v1-ett-r2s: the 17 new combinations x 5 models x ETTh2 / ETTm2 x H96 / H720 (340 runs); the current combination is
           the existing seed-42 run. Score per model and dataset: s(c) = mean over H96 / H720 of val(c) / val(current) - 1, val = the best
           validation L1 (val_loss_best). The lowest s; it replaces the current setting only if s < 0.
  MIT-BIH  suite flif-v1-mitbih-r2s: all 18 combinations x 5 models with VAL=1 (80 / 20 split of the training file; 90 runs). Score = the
           last-epoch validation macro-F1 (val_f1). The highest; it replaces the current setting only if it is above the current one.
  Ties keep the earlier combination in the grid order (lr, then batch, then warmup; ascending / current warmup first).
Final (seeds 42 / 7 / 2026), only where the selected setting is not the current one:
  ETT      suite flif-v1-ett-r2f, both horizons of that dataset;   MIT-BIH  suite flif-v1-mitbih-r2f, VAL=0 (full training file).
  The seed-42 ETT run repeats the search run; summarize_r2.py requires the two to agree exactly.
Fail-closed: a missing / duplicated row, a SKIP line in a search file that is not answered by exactly one recovered row (below), a non-finite score, a RESULT lr / bs / wu field or a run folder
(lr, tag _b / _w, suffix _wu) that does not match the combination, a row that is not range-1 / bias-off / SpikeDE, MIT-BIH not VAL=1 or
not 30 epochs, more than one sha among the search rows, an existing selection file.
MODI 2026-11-24: 5 ETT search runs finished but got a SKIP line (the runners' folder lookup had no lr when LR was unset; fixed, check_r2.py
f). recover_r2_rows.py appended their RESULT lines with out=<the SKIP line's stdout> collected=1. A SKIP line is now accepted only when
exactly one collected=1 line names its stdout, and every collected=1 line must answer exactly one SKIP line; the selection file lists them.
usage (classification folder):
  python select_r2.py --queue_search FILE                 write the search queue (430 lines), nothing else
  python select_r2.py --select --queue_final FILE         read the search results, write results/flif-v1-r2-selected.json + the final queue
"""
import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.join(HERE, '..', 'forecasting', 'results')
CL = os.path.join(HERE, 'results')
NEURONS = ('pop_lif', 'pop_flif')
MODELS = ('spk',) + tuple(f'{k}_{n}' for n in NEURONS for k in ('base', 'na'))
DATASETS = ('ETTh2', 'ETTm2')
HORIZONS = (96, 720)
SEEDS = (42, 7, 2026)
GRID = {'ett': dict(lr=('0.00025', '0.0005', '0.001'), bs=('32', '64', '128'), wu=('none', '1')),
        'mb': dict(lr=('0.005', '0.01', '0.02'), bs=('256', '512', '1024'), wu=('6', '0'))}
CUR = {'ett': ('0.0005', '64', 'none'), 'mb': ('0.01', '512', '6')}
SEARCH = {'ett': 'flif-v1-ett-r2s', 'mb': 'flif-v1-mitbih-r2s'}
FINAL = {'ett': 'flif-v1-ett-r2f', 'mb': 'flif-v1-mitbih-r2f'}
MX = 'nsmt_flif_v1/forecasting/scripts/run_ett_mixer.sh'
ET = 'nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh'
M4 = 'nsmt_flif_v1/classification/scripts/run_mitbih4.sh'


def cond(m):
    if m == 'spk':
        return 'spkall_lif'
    kind, n = m.split('_', 1)
    return f'nsmt4q_{n}_msn_sde' + ('_ncna' if kind == 'na' else '')


def ref_suite(task, m):
    arm = 'nc' if m.startswith('na_') else 'r1'
    return f'flif-v1-ett-{arm}' if task == 'ett' else f'flif-v1-mitbih-{arm}'


def combos(task):
    g = GRID[task]
    return [(lr, bs, wu) for lr in g['lr'] for bs in g['bs'] for wu in g['wu']]


def name(c):
    return f'lr {c[0]}, batch {c[1]}, warmup {c[2]}'


def env(task, c, val=False):
    """the runner environment of one combination; a current value is left unset (ETT) so the command line is the existing one."""
    lr, bs, wu = c; e = []
    if task == 'ett':
        if lr != CUR['ett'][0]:
            e.append(f'LR={lr}')
        if bs != CUR['ett'][1]:
            e.append(f'BS={bs}')
        if wu != 'none':
            e.append(f'WU={wu}')
    else:
        if val:
            e.append('VAL=1')
        if val or lr != CUR['mb'][0]:
            e.append(f'LR={lr}')
        if bs != CUR['mb'][1]:
            e.append(f'BS={bs}')
        if wu != CUR['mb'][2]:
            e.append(f'WU={wu}')
    return e


def line(task, m, suite, c, ds=None, pl=None, seed=42, val=False):
    pre = ' '.join([f'COND={cond(m)}', f'SUITE={suite}'] + env(task, c, val))
    if task == 'ett':
        return f"{pre} bash {MX if m == 'spk' else ET} {ds} {pl} {seed}"
    return f'{pre} bash {M4} {seed}'


def search_queue():
    """ETTm2 (longest) first, then MIT-BIH, then ETTh2; within a block the small batches (longest) first."""
    L = []
    order = sorted((c for c in combos('ett') if c != CUR['ett']), key=lambda c: int(c[1]))
    for pl in (720, 96):
        for c in order:
            for m in MODELS:
                L.append(line('ett', m, SEARCH['ett'], c, 'ETTm2', pl))
    for c in sorted(combos('mb'), key=lambda c: int(c[1])):
        for m in MODELS:
            L.append(line('mb', m, SEARCH['mb'], c, val=True))
    for pl in (720, 96):
        for c in order:
            for m in MODELS:
                L.append(line('ett', m, SEARCH['ett'], c, 'ETTh2', pl))
    return L


def unanswered(res, skip):
    """SKIP lines (by stdout) not answered by exactly one collected=1 RESULT line, and collected=1 lines without exactly one SKIP line."""
    outs = [ln.split(' out=', 1)[1].split()[0] if ' out=' in ln else f'SKIP without out=: {ln.strip()}' for ln in skip]
    col = [d.get('out') for d in res if d.get('collected') == '1']
    return [o for o in outs if col.count(o) != 1] + [f'collected without SKIP: {o}' for o in col if outs.count(o) != 1]


def read(path):
    res, skip = [], []
    for ln in open(path):
        if ln.startswith('RESULT'):
            res.append(dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t))
        elif ln.startswith('SKIP'):
            skip.append(ln)
    return res, skip


def want_fields(task, c, val=False):
    """RESULT fields a row of combination c must carry (None = the field must be absent)."""
    lr, bs, wu = c
    if task == 'ett':
        return dict(lr=None if lr == CUR['ett'][0] else lr, bs=None if bs == CUR['ett'][1] else bs, wu=None if wu == 'none' else wu)
    return dict(lr=lr, bs=None if bs == CUR['mb'][1] else bs, wu=None if wu == CUR['mb'][2] else wu, val='1' if val else '0')


def check_row(task, d, m, c, what, val=False):
    """the row is combination c of model m, trained under the fixed conditions (range 1, bias off, every neuron SpikeDE)."""
    lr, bs, wu = c
    for k, w in want_fields(task, c, val).items():
        if d.get(k) != w:
            raise RuntimeError(f'{what}: RESULT {k}={d.get(k)} but the combination needs {w}')
    path = d['dir'].rstrip('/'); run = os.path.basename(path); top = os.path.basename(os.path.dirname(path))
    tb = '' if (bs == CUR[task][1]) else f'_b{bs}'
    tw = '' if (wu == ('none' if task == 'ett' else CUR['mb'][2])) else f'_w{wu}'
    tag = cond(m) + (tb + tw if task == 'ett' else f"_lr{lr}_v{'1' if val else '0'}_wdx{tb}{tw}")
    if f'+{tag}+seed{d["seed"]}+' not in top or f'+lr+{lr}+' not in top:
        raise RuntimeError(f'{what}: folder tag / lr ({top})')
    if (f'_wu{wu}' in run) != (tw != '') or ('_wu' in run) != (tw != ''):
        raise RuntimeError(f'{what}: warmup suffix ({run})')
    if '_nalif' not in run or '_nope' not in run or '_wdx' not in run or d.get('pe') != '0' or d.get('wdx') != '1':
        raise RuntimeError(f'{what}: not SpikeDE / bias-off / range-1 ({run})')
    if (('_nc' in run) or ('_noattn' in run)) != m.startswith('na_') or ('_noattn' in run) != m.startswith('na_'):
        raise RuntimeError(f'{what}: structure suffix ({run})')
    if task == 'ett':
        if '_vpe4' not in run or d.get('vpe') != '4' or int(d.get('seq_len', 96)) != 96 or len(d['best_epoch_val_epochs'].split(':')) != 4:
            raise RuntimeError(f'{what}: not the range-1 validation / input 96 ({run})')
    else:
        if ('_mbval' in run) != val or int(d['epochs_run']) != 30 or int(d['epochs']) != 30:
            raise RuntimeError(f'{what}: validation split / epochs ({run}, {d.get("epochs_run")})')
    return d


def one(rows, what, **sel):
    r = [d for d in rows if all(d.get(k) == (None if v is None else str(v)) for k, v in sel.items())]
    if len(r) != 1:
        raise RuntimeError(f'{what}: {len(r)} rows')
    return r[0]


def score(d, key, what):
    v = float(d[key])
    if not math.isfinite(v) or v <= 0:
        raise RuntimeError(f'{what}: {key} {d[key]}')
    return v


def select(fc=FC, cl=CL, suites=None):
    S = dict(SEARCH, **(suites or {}))
    rows_e, skip_e = read(os.path.join(fc, S['ett'], 'raw.txt'))
    rows_m, skip_m = read(os.path.join(cl, S['mb'], 'raw.txt'))
    bad = unanswered(rows_e, skip_e) + unanswered(rows_m, skip_m)
    if bad:
        raise RuntimeError(f'SKIP lines not answered by a recovered row (or the reverse): {bad}')
    ne, nm = (len(combos('ett')) - 1) * len(MODELS) * len(DATASETS) * len(HORIZONS), len(combos('mb')) * len(MODELS)
    if len(rows_e) != ne or len(rows_m) != nm:
        raise RuntimeError(f'search rows: ETT {len(rows_e)} (expected {ne}), MIT-BIH {len(rows_m)} (expected {nm})')
    shas = {d['sha'] for d in rows_e} | {d['sha'] for d in rows_m}
    if len(shas) != 1:
        raise RuntimeError(f'search shas {shas}')
    out = dict(sha=next(iter(shas)), ett={}, mb={}, collected=sorted(d['dir'] for d in rows_e + rows_m if d.get('collected') == '1'))
    for m in MODELS:
        ref, _ = read(os.path.join(fc, ref_suite('ett', m), 'raw.txt'))
        out['ett'][m] = {}
        for ds in DATASETS:
            val = {}
            for c in combos('ett'):
                cur = c == CUR['ett']
                for pl in HORIZONS:
                    what = f'ETT {m} {ds} H{pl} seed 42 [{name(c)}]'
                    d = one(ref if cur else rows_e, what, cond=cond(m), ds=ds, pl=pl, seed=42, **want_fields('ett', c))
                    val[c, pl] = score(check_row('ett', d, m, c, what), 'val_loss_best', what)
            s = {c: 100 * sum(val[c, pl] / val[CUR['ett'], pl] - 1 for pl in HORIZONS) / len(HORIZONS) for c in combos('ett')}
            best = min((c for c in combos('ett') if c != CUR['ett']), key=lambda c: s[c])      # min keeps the first of equal scores
            sel = best if s[best] < 0 else CUR['ett']
            out['ett'][m][ds] = dict(selected=list(sel), changed=sel != CUR['ett'], best_new=list(best),
                                     table=[dict(lr=c[0], bs=c[1], wu=c[2], score_pct=s[c], **{f'val_H{pl}': val[c, pl] for pl in HORIZONS}) for c in combos('ett')],
                                     edge=dict(lr=sel[0] != GRID['ett']['lr'][1], bs=sel[1] != GRID['ett']['bs'][1]))
        v = {}
        for c in combos('mb'):
            what = f'MIT-BIH {m} seed 42 VAL=1 [{name(c)}]'
            d = one(rows_m, what, cond=cond(m), seed=42, **want_fields('mb', c, True))
            v[c] = (score(check_row('mb', d, m, c, what, True), 'val_f1', what), score(d, 'f1', what))
        best = max((c for c in combos('mb') if c != CUR['mb']), key=lambda c: v[c][0])          # max keeps the first of equal scores
        sel = best if v[best][0] > v[CUR['mb']][0] else CUR['mb']
        out['mb'][m] = dict(selected=list(sel), changed=sel != CUR['mb'], best_new=list(best),
                            table=[dict(lr=c[0], bs=c[1], wu=c[2], val_f1=v[c][0], test_f1_val_run_report_only=v[c][1]) for c in combos('mb')],
                            edge=dict(lr=sel[0] != GRID['mb']['lr'][1], bs=sel[1] != GRID['mb']['bs'][1]))
    return out


def final_queue(state):
    L = []
    for pl in (720, 96):
        for m in MODELS:
            r = state['ett'][m]['ETTm2']
            if r['changed']:
                L += [line('ett', m, FINAL['ett'], tuple(r['selected']), 'ETTm2', pl, s) for s in SEEDS]
    for m in MODELS:
        r = state['mb'][m]
        if r['changed']:
            L += [line('mb', m, FINAL['mb'], tuple(r['selected']), seed=s) for s in SEEDS]
    for pl in (720, 96):
        for m in MODELS:
            r = state['ett'][m]['ETTh2']
            if r['changed']:
                L += [line('ett', m, FINAL['ett'], tuple(r['selected']), 'ETTh2', pl, s) for s in SEEDS]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue_search')
    ap.add_argument('--select', action='store_true')
    ap.add_argument('--queue_final')
    ap.add_argument('--state', default=os.path.join(CL, 'flif-v1-r2-selected.json'))
    a = ap.parse_args()
    if bool(a.queue_search) == a.select or (a.select and not a.queue_final):
        ap.error('either --queue_search FILE, or --select --queue_final FILE')
    if a.queue_search:
        L = search_queue()
        if len(L) != 430 or len(set(L)) != 430:
            raise RuntimeError(f'search queue: {len(L)} lines, {len(set(L))} distinct')
        open(a.queue_search, 'w').write(''.join(x + '\n' for x in L))
        print(f'{a.queue_search}: {len(L)} jobs (search: ETT 340, MIT-BIH 90)')
        return
    if os.path.exists(a.state):
        raise RuntimeError(f'{a.state} exists: the selection was already made')
    state = select()
    L = final_queue(state)
    json.dump(state, open(a.state, 'w'), indent=1)
    open(a.queue_final, 'w').write(''.join(x + '\n' for x in L))
    for m in MODELS:
        for ds in DATASETS:
            r = state['ett'][m][ds]; sc = {(t['lr'], t['bs'], t['wu']): t['score_pct'] for t in r['table']}
            print(f"ETT {m} {ds}: {'-> ' + name(r['selected']) if r['changed'] else 'keeps the current setting'}  "
                  f"(best new {name(r['best_new'])} {sc[tuple(r['best_new'])]:+.3f} %)")
        r = state['mb'][m]; vf = {(t['lr'], t['bs'], t['wu']): t['val_f1'] for t in r['table']}
        print(f"MIT-BIH {m}: {'-> ' + name(r['selected']) if r['changed'] else 'keeps the current setting'}  "
              f"(best new {name(r['best_new'])} val F1 {100 * vf[tuple(r['best_new'])]:.2f}, current {100 * vf[CUR['mb']]:.2f})")
    print(f'{a.queue_final}: {len(L)} jobs (final)')


if __name__ == '__main__':
    main()
