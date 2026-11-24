"""MODI 2026-11-23: summary of hyperparameter range 2 (lr x batch size x warmup; docs/MODI_PROJECT_LOG.md 2026-11-23 22:37 KST,
pre-registered rules; user decision 11-23 "기준과 (나)를 둘 다 hyperparameter 조정에 넣고 결정하자"). Grid, models, suites and the
validation-only selection: select_r2.py (results/flif-v1-r2-selected.json). This file reads the test results.

Settings per model: "current" = the setting before range 2 (existing runs flif-v1-ett-r1 / -nc, flif-v1-mitbih-r1 / -nc);
"final" = the selected setting (flif-v1-ett-r2f / flif-v1-mitbih-r2f where it changed, else the current runs). Seeds 42 / 7 / 2026.
Comparisons (rules of summarize_sde.py: ETT 12 pairs of test MSE, R <= -1 % with 8/12 lower = better, R >= +1 % with 8/12 higher = worse;
MIT-BIH mean macro-F1 difference >= +0.5 %p with 2/3 seeds positive = better, the mirror = worse):
  T   per model: final - current (the effect of the tuning); "unchanged" where the setting did not change
  S1  per neuron: (나) final - base final (structure)
  S2  per neuron: base final - Spikformer(DE) final, (나) final - Spikformer(DE) final; overall = summarize_sde.overall; energy ratios
Recommendation (the structure decision is the user's; the rule of 4-2):
  1. S1 has no "worse" (2 neurons x 2 tasks)  -> recommend (나)
  2. else                                     -> keep the base; a neuron whose S1 has no "worse" is listed as a neuron-specific candidate
Report only: the validation tables of the 18 combinations, the selected settings and range-edge flags, the MIT-BIH test macro-F1 of the
VAL=1 search runs, best epoch / epochs / run time / firing rates, the same comparisons at the current settings (the 4-2 numbers).
Fail-closed: select_r2.check_row on every row used (combination fields, folder tag, range-1 / bias-off / SpikeDE), one row per seed, a SKIP
line or a wrong row count in a final file, more than one sha among the final rows, a missing energy recount, and the seed-42 ETT final
run must repeat the search run exactly (test MSE / MAE / best validation).
usage (classification folder): python summarize_r2.py [--out FILE]      (after energy_recount_ett.py --suite flif-v1-ett-r2f)
"""
import argparse
import csv
import json
import os

import numpy as np

import select_r2 as S
from compare_query_pop_lif_paper import CELLS, COLS, SEEDS, cmp_row, compare, num
from summarize_ms import conclusion, mb_verdict
from summarize_sde import overall
from summarize_xnor import robust

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(S.CL, 'flif-v1-r2-selected.json')
LABEL = {'spk': 'Spikformer(DE)', 'base_pop_lif': 'base pop LIF', 'base_pop_flif': 'base pop f-LIF', 'na_pop_lif': '(나) pop LIF', 'na_pop_flif': '(나) pop f-LIF'}
NNAME = {'pop_lif': 'pop LIF', 'pop_flif': 'pop f-LIF'}


def rows_of(task, suite):
    res, skip = S.read(os.path.join(S.FC if task == 'ett' else S.CL, suite, 'raw.txt'))
    return res, skip


def recount(suite):
    out = {}
    for r in csv.DictReader(open(os.path.join(S.FC, suite, 'energy_recount.csv'))):
        k = (r['ds'], int(r['pl']), r['cond'], int(r['seed']))
        if k in out:
            raise RuntimeError(f'{suite}: duplicate recount {k}')
        out[k] = float(r['total_fft_uJ_corr'])
    return out


def setting(state, task, m, which, ds=None):
    r = state['ett'][m][ds] if task == 'ett' else state['mb'][m]
    return tuple(r['selected']) if which == 'fin' and r['changed'] else S.CUR[task]


def ett(state, m, which):
    out, shas = {}, set()
    for ds in S.DATASETS:
        c = setting(state, 'ett', m, which, ds)
        suite = S.ref_suite('ett', m) if c == S.CUR['ett'] else S.FINAL['ett']
        rows, _ = rows_of('ett', suite); rc = recount(suite)
        for pl in S.HORIZONS:
            for sd in SEEDS:
                what = f'{suite} {m} {ds} H{pl} seed {sd} [{S.name(c)}]'
                d = S.check_row('ett', S.one(rows, what, cond=S.cond(m), ds=ds, pl=pl, seed=sd, **S.want_fields('ett', c)), m, c, what)
                k = (ds, pl, S.cond(m), sd)
                if k not in rc:
                    raise RuntimeError(f'{what}: no energy recount')
                be = d['best_epoch_val_epochs'].split(':')
                out[ds, pl, sd] = dict(mse=num(d, 'mse'), mae=num(d, 'mae'), val=num(d, 'val_loss_best'), tot=rc[k], be=int(be[0]), ep=int(be[2]),
                                       time=num(d, 'time_s'), rh=num(d, 'rho_hippo'), rn=num(d, 'rho_neo') if m != 'spk' else float('nan'), raw=d)
                if suite == S.FINAL['ett']:
                    shas.add(d['sha'])
    return out, shas


def mb(state, m, which):
    c = setting(state, 'mb', m, which)
    suite = S.ref_suite('mb', m) if c == S.CUR['mb'] else S.FINAL['mb']
    rows, _ = rows_of('mb', suite); out, shas = {}, set()
    for sd in SEEDS:
        what = f'{suite} {m} MIT-BIH seed {sd} [{S.name(c)}]'
        d = S.check_row('mb', S.one(rows, what, cond=S.cond(m), seed=sd, **S.want_fields('mb', c, False)), m, c, what, False)
        out[sd] = dict(f1=100 * num(d, 'f1'), acc=100 * num(d, 'overall_acc'), tot=num(d, 'total_fft_uJ'), rh=num(d, 'rho_hippo'),
                       rn=num(d, 'rho_neo') if m != 'spk' else float('nan'), time=num(d, 'time_s'))
        if suite == S.FINAL['mb']:
            shas.add(d['sha'])
    return out, shas


def final_files(state):
    """the final files hold exactly the rows of the changed settings, no SKIP; the seed-42 ETT rows repeat the search rows."""
    ne = sum(state['ett'][m][ds]['changed'] for m in S.MODELS for ds in S.DATASETS) * len(S.HORIZONS) * len(SEEDS)
    nm = sum(state['mb'][m]['changed'] for m in S.MODELS) * len(SEEDS)
    for task, n in (('ett', ne), ('mb', nm)):
        path = os.path.join(S.FC if task == 'ett' else S.CL, S.FINAL[task], 'raw.txt')
        if not n:
            if os.path.exists(path):
                raise RuntimeError(f'{path}: exists although no setting changed')
            continue
        res, skip = S.read(path)
        if skip or len(res) != n:
            raise RuntimeError(f'{path}: {len(res)} RESULT (expected {n}), {len(skip)} SKIP')
    rep = 0
    if ne:
        fin, _ = rows_of('ett', S.FINAL['ett']); sea, _ = rows_of('ett', S.SEARCH['ett'])
        for m in S.MODELS:
            for ds in S.DATASETS:
                if not state['ett'][m][ds]['changed']:
                    continue
                c = tuple(state['ett'][m][ds]['selected'])
                for pl in S.HORIZONS:
                    what = f'{m} {ds} H{pl} seed 42 [{S.name(c)}]'
                    sel = dict(cond=S.cond(m), ds=ds, pl=pl, seed=42, **S.want_fields('ett', c))
                    a, b = S.one(fin, 'final ' + what, **sel), S.one(sea, 'search ' + what, **sel)
                    if any(a[k] != b[k] for k in ('mse', 'mae', 'val_loss_best')):
                        raise RuntimeError(f'{what}: the final run does not repeat the search run '
                                           f"(mse {a['mse']} / {b['mse']}, mae {a['mae']} / {b['mae']}, val {a['val_loss_best']} / {b['val_loss_best']})")
                    rep += 1
    return ne, nm, rep


def ett_changed(state, m):
    return any(state['ett'][m][ds]['changed'] for ds in S.DATASETS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(S.CL, 'flif-v1-r2-summary.md'))
    ap.add_argument('--state', default=STATE)
    a = ap.parse_args()
    state = json.load(open(a.state))
    ne, nm, rep = final_files(state)
    E, M, shas = {}, {}, set()
    for m in S.MODELS:
        for which in ('cur', 'fin'):
            E[m, which], s1 = ett(state, m, which)
            M[m, which], s2 = mb(state, m, which)
            shas |= s1 | s2
    if (ne or nm) and len(shas) != 1:
        raise RuntimeError(f'final shas {shas}')
    L = ['# hyperparameter range 2 (lr x batch size x warmup) — summary', '',
         f"Pre-registered rules: docs/MODI_PROJECT_LOG.md 2026-11-23 22:37 KST. Search runs at sha {state['sha']}; final runs: ETT {ne}, MIT-BIH {nm}"
         + (f" at sha {next(iter(shas))}" if shas else '') + f'. Seed-42 ETT final runs that repeat their search run exactly: {rep}/{ne // 3}.'
         + f" Search rows recovered from their run folders after a SKIP line (recover_r2_rows.py, collected=1): {len(state.get('collected', []))}.", '',
         '## selected settings (validation only, seed 42; * = range edge)', '',
         '| model | ETTh2 | score % (best new) | ETTm2 | score % (best new) | MIT-BIH | validation macro-F1 % (best new / current) |', '|---|---|---|---|---|---|---|']

    def sel_cell(r, cur):
        c = tuple(r['selected'])
        if not r['changed']:
            return 'current kept'
        return f"lr {c[0]}{'*' if r['edge']['lr'] else ''}, batch {c[1]}{'*' if r['edge']['bs'] else ''}, warmup {c[2]}"
    for m in S.MODELS:
        cells = []
        for ds in S.DATASETS:
            r = state['ett'][m][ds]; sc = {(t['lr'], t['bs'], t['wu']): t['score_pct'] for t in r['table']}
            cells += [sel_cell(r, S.CUR['ett']), f"{sc[tuple(r['best_new'])]:+.3f}"]
        r = state['mb'][m]; vf = {(t['lr'], t['bs'], t['wu']): 100 * t['val_f1'] for t in r['table']}
        cells += [sel_cell(r, S.CUR['mb']), f"{vf[tuple(r['best_new'])]:.2f} / {vf[S.CUR['mb']]:.2f}"]
        L.append(f'| {LABEL[m]} | ' + ' | '.join(cells) + ' |')

    L += ['', '## results at the final settings (mean ± sd of seeds; ETT test MSE, MIT-BIH test macro-F1 %)', '', '| model | setting | ' + COLS + ' | MIT-BIH macro-F1 % |',
          '|---|---|---|---|---|---|---|']
    for m in S.MODELS:
        for which in ('cur', 'fin'):
            if which == 'fin' and not ett_changed(state, m) and not state['mb'][m]['changed']:
                continue
            e, x = E[m, which], M[m, which]
            cells = ' | '.join(f"{np.mean([e[ds, pl, s]['mse'] for s in SEEDS]):.4f} ± {np.std([e[ds, pl, s]['mse'] for s in SEEDS], ddof=1):.4f}" for ds, pl in CELLS)
            L.append(f"| {LABEL[m]} | {'current' if which == 'cur' else 'final'} | {cells} | {np.mean([x[s]['f1'] for s in SEEDS]):.2f} ± {np.std([x[s]['f1'] for s in SEEDS], ddof=1):.2f} |")

    C = {}
    comps = []                                                         # (key, label, A, B)
    for m in S.MODELS:
        comps.append((('T', m), f'T {LABEL[m]}: final − current', (m, 'fin'), (m, 'cur')))
    for n in S.NEURONS:
        comps.append((('S1', n), f'S1 {NNAME[n]}: (나) − base, final', (f'na_{n}', 'fin'), (f'base_{n}', 'fin')))
    for n in S.NEURONS:
        for k in ('base', 'na'):
            comps.append((('S2', k, n), f"S2 {NNAME[n]}: {'base' if k == 'base' else '(나)'} − Spikformer(DE), final", (f'{k}_{n}', 'fin'), ('spk', 'fin')))
    for n in S.NEURONS:
        comps.append((('S1c', n), f'(reference) {NNAME[n]}: (나) − base, current', (f'na_{n}', 'cur'), (f'base_{n}', 'cur')))
        for k in ('base', 'na'):
            comps.append((('S2c', k, n), f"(reference) {NNAME[n]}: {'base' if k == 'base' else '(나)'} − Spikformer(DE), current", (f'{k}_{n}', 'cur'), ('spk', 'cur')))
    L += ['', '## ETT (r = A / B − 1 of test MSE, %)', '', '| comparison | ' + COLS + ' | R % | pairs A lower | verdict | energy A/B | leave-one-cell-out |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for key, label, A, B in comps:
        if key[0] == 'T' and not ett_changed(state, key[1]):
            C[key, 'ett'] = None
            L.append(f'| {label} | — | — | — | — | — | unchanged (the current setting was kept) | — | — |')
            continue
        C[key, 'ett'] = x = compare(E[A], E[B])
        L.append(cmp_row(label, x) + f" {robust(x, x['v'], x['R'])} |")
    L += ['', '## MIT-BIH (d = A − B of test macro-F1, %p per seed 42 / 7 / 2026)', '', '| comparison | d per seed | mean | verdict | energy A/B |', '|---|---|---|---|---|']
    for key, label, A, B in comps:
        if key[0] == 'T' and not state['mb'][key[1]]['changed']:
            C[key, 'mb'] = None
            L.append(f'| {label} | — | — | unchanged (the current setting was kept) | — |')
            continue
        C[key, 'mb'] = x = mb_verdict(M[A], M[B])
        er = np.mean([M[A][s]['tot'] for s in SEEDS]) / np.mean([M[B][s]['tot'] for s in SEEDS])
        L.append(f"| {label} | {' / '.join(f'{v:+.2f}' for v in x[0])} | {x[1]:+.2f} | {x[2]} | {er:.2f} |")

    L += ['', '## conclusions (rules of the pre-registration)', '', '| comparison | task | pop LIF | pop f-LIF | conclusion over the two neurons |', '|---|---|---|---|---|']
    ve = [C[('S1', n), 'ett']['v'] for n in S.NEURONS]; vm = [C[('S1', n), 'mb'][2] for n in S.NEURONS]
    L.append(f'| S1 (나) − base | ETT | {ve[0]} | {ve[1]} | {conclusion(ve)} |')
    L.append(f'| S1 (나) − base | MIT-BIH | {vm[0]} | {vm[1]} | {conclusion(vm)} |')
    for k, what in (('base', 'base'), ('na', '(나)')):
        xe = [C[('S2', k, n), 'ett']['v'] for n in S.NEURONS]; xm = [C[('S2', k, n), 'mb'][2] for n in S.NEURONS]
        L.append(f'| S2 {what} − Spikformer(DE) | ETT | {xe[0]} | {xe[1]} | {conclusion(xe)} |')
        L.append(f'| S2 {what} − Spikformer(DE) | MIT-BIH | {xm[0]} | {xm[1]} | {conclusion(xm)} |')
        L.append(f"| overall: {what} better than Spikformer(DE) | both | {overall(C[('S2', k, S.NEURONS[0]), 'ett'], C[('S2', k, S.NEURONS[0]), 'mb'])} | "
                 f"{overall(C[('S2', k, S.NEURONS[1]), 'ett'], C[('S2', k, S.NEURONS[1]), 'mb'])} | — |")
    worse = {n: 'worse' in (C[('S1', n), 'ett']['v'], C[('S1', n), 'mb'][2]) for n in S.NEURONS}
    if not any(worse.values()):
        rec = '(나): S1 has no "worse"'
    else:
        ok = [NNAME[n] for n in S.NEURONS if not worse[n]]
        rec = 'keep the base: S1 has a "worse"' + (f'; neuron-specific candidate (a choice made after seeing the results): (나) with {", ".join(ok)}' if ok else '')
    L += ['', f'**Recommendation (pre-registered rule; the structure decision is the user\'s):** {rec}. S1 verdicts (ETT pop LIF, ETT pop f-LIF, MIT-BIH pop LIF, '
          f'MIT-BIH pop f-LIF): {", ".join(ve + vm)}.']

    L += ['', '## report only', '', '| model | setting | ETT energy µJ / Spikformer(DE) final | MIT-BIH energy µJ / Spikformer(DE) final | ETT rho_hippo | ETT rho_neo | MIT-BIH rho_hippo | '
          'MIT-BIH rho_neo | MIT-BIH accuracy % | ETT best at epoch 0 (of 12) | ETT epochs (min–max) | ETT run time s (min–max) | MIT-BIH run time s (min–max) |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    f = lambda v: '—' if np.isnan(v) else f'{v:.4f}'
    for m in S.MODELS:
        for which in ('cur', 'fin'):
            e, x = E[m, which], M[m, which]; e0, x0 = E['spk', 'fin'], M['spk', 'fin']
            er = np.mean([np.mean([e[ds, pl, s]['tot'] for s in SEEDS]) / np.mean([e0[ds, pl, s]['tot'] for s in SEEDS]) for ds, pl in CELLS])
            mr = np.mean([x[s]['tot'] for s in SEEDS]) / np.mean([x0[s]['tot'] for s in SEEDS])
            L.append(f"| {LABEL[m]} | {'current' if which == 'cur' else 'final'} | {np.mean([e[q]['tot'] for q in e]):.2f} / {er:.2f} | {np.mean([x[s]['tot'] for s in SEEDS]):.2f} / {mr:.2f} | "
                     f"{f(np.mean([e[q]['rh'] for q in e]))} | {f(np.mean([e[q]['rn'] for q in e]))} | {f(np.mean([x[s]['rh'] for s in SEEDS]))} | {f(np.mean([x[s]['rn'] for s in SEEDS]))} | "
                     f"{np.mean([x[s]['acc'] for s in SEEDS]):.2f} | {sum(e[q]['be'] == 0 for q in e)} | {min(e[q]['ep'] for q in e)}–{max(e[q]['ep'] for q in e)} | "
                     f"{min(e[q]['time'] for q in e):.0f}–{max(e[q]['time'] for q in e):.0f} | {min(x[s]['time'] for s in SEEDS):.0f}–{max(x[s]['time'] for s in SEEDS):.0f} |")
    for ds in S.DATASETS:
        L += ['', f'### {ds}: validation score of every combination, seed 42 (s = mean over H96 / H720 of val / val(current) − 1, %; lower is better; **selected**)', '',
              '| lr | batch | warmup | ' + ' | '.join(LABEL[m] for m in S.MODELS) + ' |', '|---|---|---|' + '---|' * len(S.MODELS)]
        for c in S.combos('ett'):
            cells = []
            for m in S.MODELS:
                r = state['ett'][m][ds]; sc = {(t['lr'], t['bs'], t['wu']): t['score_pct'] for t in r['table']}
                v = f'{sc[c]:+.3f}'
                cells.append(f'**{v}**' if tuple(r['selected']) == c else v)
            L.append(f"| {c[0]} | {c[1]} | {c[2]} | " + ' | '.join(cells) + ' |' + (' (current)' if c == S.CUR['ett'] else ''))
    L += ['', '### MIT-BIH: last-epoch validation macro-F1 % of every combination, seed 42, VAL=1 (higher is better; **selected**); in brackets the test macro-F1 % of the same run (not used)', '',
          '| lr | batch | warmup epochs | ' + ' | '.join(LABEL[m] for m in S.MODELS) + ' |', '|---|---|---|' + '---|' * len(S.MODELS)]
    for c in S.combos('mb'):
        cells = []
        for m in S.MODELS:
            r = state['mb'][m]; t = next(t for t in r['table'] if (t['lr'], t['bs'], t['wu']) == c)
            v = f"{100 * t['val_f1']:.2f} [{100 * t['test_f1_val_run_report_only']:.2f}]"
            cells.append(f'**{v}**' if tuple(r['selected']) == c else v)
        L.append(f"| {c[0]} | {c[1]} | {c[2]} | " + ' | '.join(cells) + ' |' + (' (current)' if c == S.CUR['mb'] else ''))
    open(a.out, 'w').write('\n'.join(L) + '\n')
    print(a.out)


if __name__ == '__main__':
    main()
