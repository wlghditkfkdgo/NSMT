"""MODI 2026-11-18: summary of the long-input learning-rate test (docs/MODI_PROJECT_LOG.md, pre-registered rules; step 2 of user
decision 11-18 "1. 진행  2. 1번에서 개선이 없으면 진행").

Question: at inputs 336 and 720 most runs had their best validation at the first epoch (43 / 84 and 54 / 84) with the lr chosen at
input 96 (0.0005). With the lr re-chosen per model and input length on validation, (S1) does each model improve, and (S2 / S3) does
the base model (Neocortex-MS query population, pop LIF / pop f-LIF) now differ from Spikformer-style, or does the gap change?
Arms (ETTh2 / ETTm2 x H96 / H720, seeds 42 / 7 / 2026, best-validation weights, test MSE):
  default: lr 0.0005, suites flif-v1-ett-len336 / -len720 (reused; the input-length test)
  selected: the lr of select_lrlong.py (seed-42 validation over lr in {5e-05, 0.0001, 0.00025, 0.0005}), suite flif-v1-ett-lr<L>-<lr>
            (seed 42 from phase A, seeds 7 / 2026 from phase B); when the default is selected the arm is the default arm.
Comparisons per input length (rules of compare(): r = A / B - 1 of test MSE over the 12 cell x seed pairs; better = mean <= -1 % and
>= 8/12 lower, worse = mirror; per cell +-1 % and 2/3; two neurons: both -> that verdict, one -> neuron-dependent (weak)):
  S1 selected - default, per model                               (does re-choosing the lr help that model)
  S2 base(selected) - Spikformer-style(selected), per neuron     (the main comparison, each model at its own lr)
  S3 gap change g = r_S2 - r_ref per pair (%p), r_ref = base(default) / Spikformer-style(default) - 1 (= L1 of the input-length test);
     grew = mean <= -1 and >= 8/12 negative, shrank = mirror
  leave-one-cell-out as in summarize_xnor.py (robust / one-cell (weak)).
Report only: the seed-42 selection table (validation and, not used for anything, test at every lr), first-epoch stops, epochs run,
run time, MAE, energy (head recount), the retrain check (input 720, ETTh2 H96 seed 42, suite flif-v1-ett-lr-repro).
The selection JSON must equal a fresh selection; a missing, duplicated or non-finite value, a wrong seq_len / lr / run folder or new rows
at several shas raise (fail-closed).
usage (classification folder): python summarize_lrlong.py [--selected F] [--out FILE]
"""
import argparse
import json
import os

import numpy as np

from compare_query_pop_lif_paper import CELLS, COLS, FC, HERE, SEEDS, by_seed, cmp_row, compare, ett_recounted, num, results
from select_lrlong import DEFAULT, GRID, LENGTHS, MODELS, select, suite
from summarize_len import gap_change, gap_conclusion
from summarize_ms import NAMES, NEURONS, conclusion
from summarize_xnor import robust

MNAME = {'spk_ssa': 'Spikformer-style', 'nsmt4q_pop_lif_msn': 'query pop LIF base', 'nsmt4q_pop_flif_msn': 'query pop f-LIF base'}


def load(c, L, lr):
    s = suite(L, lr)
    out = ett_recounted(c, (s,))
    shas = set()
    for ds, pl in CELLS:
        for sd, d in by_seed(results(os.path.join(FC, s, 'raw.txt'), ds=ds, pl=pl, cond=c), f'{s} {c} {ds} {pl}').items():
            if int(d.get('seq_len', -1)) != L:
                raise RuntimeError(f'{s} {c} {ds} {pl} seed {sd}: seq_len {d.get("seq_len")} != {L}')
            if f'+lr+{lr}+' not in d['dir'] or (lr != DEFAULT and d.get('lr') != lr):
                raise RuntimeError(f'{s} {c} {ds} {pl} seed {sd}: not lr {lr} ({d.get("lr")}, {d["dir"]})')
            be = d['best_epoch_val_epochs'].split(':')
            out[ds, pl, sd].update(rh=num(d, 'rho_hippo'), time=num(d, 'time_s'), be=int(be[0]), ep=int(be[2]),
                                   rn=num(d, 'rho_neo') if c != 'spk_ssa' else float('nan'))
            if lr != DEFAULT:
                shas.add(d['sha'])
    return out, shas


def seed42(c, L, lr):
    out = {}
    for ds, pl in CELLS:
        r = results(os.path.join(FC, suite(L, lr), 'raw.txt'), ds=ds, pl=pl, cond=c, seed=42)
        if len(r) != 1:
            raise RuntimeError(f'{suite(L, lr)} {c} {ds} {pl} seed 42: {len(r)} rows')
        be = r[0]['best_epoch_val_epochs'].split(':')
        out[ds, pl] = dict(val=num(r[0], 'val_loss_best'), mse=num(r[0], 'mse'), be=int(be[0]), ep=int(be[2]), sha=r[0]['sha'])
    return out


def repro():
    lines, shas = [], set()
    for c in MODELS:
        new = results(os.path.join(FC, 'flif-v1-ett-lr-repro', 'raw.txt'), ds='ETTh2', pl=96, cond=c)
        old = results(os.path.join(FC, 'flif-v1-ett-len720', 'raw.txt'), ds='ETTh2', pl=96, cond=c, seed=42)
        if [int(r['seed']) for r in new] != [42] or len(old) != 1 or int(new[0]['seq_len']) != 720 or 'lr' in new[0]:
            raise RuntimeError(f'repro {c}: rows {len(new)} / {len(old)}')
        a, b = new[0], old[0]
        same = all(a[k] == b[k] for k in ('mse', 'mae', 'val_loss_best', 'total_fft_uJ', 'last_mse'))
        lines.append(f"| {c} | {b['mse']} / {b['mae']} / {b['val_loss_best']} ({b['sha']}) | {a['mse']} / {a['mae']} / {a['val_loss_best']} ({a['sha']}) | "
                     f"{100 * (num(a, 'mse') / num(b, 'mse') - 1):+.2f} % | {'yes' if same else 'no'} |")
        shas.add(a['sha'])
    return lines, shas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selected', default=os.path.join(HERE, 'results', 'flif-v1-lrlong-selected.json'))
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'flif-v1-lrlong-summary.md'))
    a = ap.parse_args()
    SEL = select(FC)
    saved = json.load(open(a.selected))
    if {k: v['selected'] for k, v in saved.items()} != {k: v['selected'] for k, v in SEL.items()}:
        raise RuntimeError('the saved selection differs from a fresh selection')
    rep, shas = repro()
    L_ = ['# Long-input learning-rate test (inputs 336 / 720; Spikformer-style and the msn base pop LIF / pop f-LIF) — summary', '',
          'See the module docstring and docs/MODI_PROJECT_LOG.md for the pre-registered rules. Default lr 0.0005 = the input-length test runs.', '',
          '## Retrain check (report only): input 720, ETTh2 H96 seed 42, default lr, retrained at this sha (test MSE / MAE / validation L1)', '',
          '| condition | logged (sha) | retrained (sha) | test MSE change | MSE, MAE, validation, energy, last MSE identical |', '|---|---|---|---|---|'] + rep
    L_ += ['', '## Selection (seed 42; validation decides, test is shown only for reference)', '',
           's = mean over the four cells of validation L1 / validation L1 at 0.0005 − 1 (%). Test r = the same ratio of test MSE (seed 42, report only). '
           'first-epoch = cells (of 4) whose best validation was the first epoch.', '',
           '| input | model | ' + ' | '.join(f'lr {x}: s / test r / first-epoch' for x in (DEFAULT,) + GRID) + ' | selected |',
           '|---|---|' + '---|' * (len(GRID) + 2)]
    s42 = {}
    for L in LENGTHS:
        for c in MODELS:
            s42[c, L] = {lr: seed42(c, L, lr) for lr in (DEFAULT,) + GRID}
            d0 = s42[c, L][DEFAULT]
            cells = []
            for lr in (DEFAULT,) + GRID:
                d = s42[c, L][lr]
                if lr != DEFAULT:
                    shas |= {d[k]['sha'] for k in d}
                tr = 100 * float(np.mean([d[k]['mse'] / d0[k]['mse'] - 1 for k in d]))
                cells.append(f"{SEL[f'{c} L{L}']['rel_val_pct'][lr]:+.2f} / {tr:+.2f} / {sum(d[k]['be'] == 0 for k in d)}")
            L_.append(f"| {L} | {MNAME[c]} | " + ' | '.join(cells) + f" | **{SEL[f'{c} L{L}']['selected']}** |")
    C, A = {}, {}
    for L in LENGTHS:
        for c in MODELS:
            A[c, L, 'def'], _ = load(c, L, DEFAULT)
            sel = SEL[f'{c} L{L}']['selected']
            if sel == DEFAULT:
                A[c, L, 'sel'] = A[c, L, 'def']
            else:
                A[c, L, 'sel'], sh = load(c, L, sel)
                shas |= sh
    if len(shas) != 1:
        raise RuntimeError(f'new rows at several shas {shas}')
    L_ += ['', f'New runs at sha {shas.pop()}.']
    head = ['| comparison | ' + COLS + ' | R % | pairs A lower | verdict | energy A/B | leave-one-cell-out |', '|---|---|---|---|---|---|---|---|---|---|']
    for L in LENGTHS:
        L_ += ['', f'## Input {L}: S1 / S2 (r = A / B − 1 of test MSE, %; energy = head recount)', ''] + head
        for c in MODELS:
            sel = SEL[f'{c} L{L}']['selected']
            if sel == DEFAULT:
                C['S1', c, L] = None
                L_.append(f'| S1 {MNAME[c]}: selected − default | — | — | — | — | — | — | default selected (no new runs) | — | — |')
            else:
                C['S1', c, L] = x = compare(A[c, L, 'sel'], A[c, L, 'def'])
                L_.append(cmp_row(f'S1 {MNAME[c]}: lr {sel} − lr {DEFAULT}', x) + f" {robust(x, x['v'], x['R'])} |")
        for n in NEURONS:
            b = f'nsmt4q_{n}_msn'
            C['S2', n, L] = x = compare(A[b, L, 'sel'], A['spk_ssa', L, 'sel'])
            L_.append(cmp_row(f"S2 {NAMES[n]}: base (lr {SEL[f'{b} L{L}']['selected']}) − Spikformer-style (lr {SEL[f'spk_ssa L{L}']['selected']})", x)
                      + f" {robust(x, x['v'], x['R'])} |")
        for n in NEURONS:
            C['ref', n, L] = x = compare(A[f'nsmt4q_{n}_msn', L, 'def'], A['spk_ssa', L, 'def'])
            L_.append(cmp_row(f'(reference) {NAMES[n]}: base − Spikformer-style, both lr {DEFAULT}', x) + f" {robust(x, x['v'], x['R'])} |")
        L_ += ['', f'### Input {L}: S3 — change of the gap (g = r_S2 − r_ref per cell and seed, %p; negative = the base gains more with the re-chosen lr)', '',
               '| comparison | ' + COLS + ' | mean g | pairs negative | verdict | leave-one-cell-out |', '|---|---|---|---|---|---|---|---|---|']
        for n in NEURONS:
            g = gap_change(C['S2', n, L], C['ref', n, L]); C['S3', n, L] = g
            L_.append(f"| S3 {NAMES[n]} | " + ' | '.join(f"{np.mean(g['per'][c]):+.2f}" for c in CELLS) + f" | {g['G']:+.2f} | {g['neg']}/12 | {g['v']} | "
                      f"{robust(g, g['v'], g['G'])} |")
    L_ += ['', '## conclusions (rule of the pre-registration)', '', '| input | comparison | Spikformer-style | pop LIF | pop f-LIF | conclusion over the two neurons |',
           '|---|---|---|---|---|---|']
    for L in LENGTHS:
        s1 = {c: ('default selected' if C['S1', c, L] is None else C['S1', c, L]['v']) for c in MODELS}
        L_.append(f"| {L} | S1 selected − default | {s1['spk_ssa']} | {s1['nsmt4q_pop_lif_msn']} | {s1['nsmt4q_pop_flif_msn']} | (per model) |")
        L_.append(f"| {L} | S2 base − Spikformer-style (own lr) | — | {C['S2', 'pop_lif', L]['v']} | {C['S2', 'pop_flif', L]['v']} | "
                  f"{conclusion([C['S2', n, L]['v'] for n in NEURONS])} |")
        L_.append(f"| {L} | S3 gap change | — | {C['S3', 'pop_lif', L]['v']} | {C['S3', 'pop_flif', L]['v']} | {gap_conclusion([C['S3', n, L]['v'] for n in NEURONS])} |")
    L_ += ['', '## report only', '', '### test MSE (mean ± sd of seeds), first-epoch stops, epochs, run time, energy', '',
           '| input | model | arm (lr) | ' + COLS + ' | best validation at epoch 0 (of 12) | epochs run (min–max) | run time s (min–max) | energy µJ (mean) | rho_neo | rho_hippo |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for L in LENGTHS:
        for c in MODELS:
            for arm in ('def', 'sel'):
                lr = DEFAULT if arm == 'def' else SEL[f'{c} L{L}']['selected']
                if arm == 'sel' and lr == DEFAULT:
                    continue
                t = A[c, L, arm]
                cells = ' | '.join(f"{np.mean([t[ds, pl, s]['mse'] for s in SEEDS]):.4f} ± {np.std([t[ds, pl, s]['mse'] for s in SEEDS], ddof=1):.4f}" for ds, pl in CELLS)
                ep = [t[k]['ep'] for k in t]; tm = [t[k]['time'] for k in t]; rn = [t[k]['rn'] for k in t]
                L_.append(f"| {L} | {MNAME[c]} | {'default' if arm == 'def' else 'selected'} ({lr}) | {cells} | {sum(t[k]['be'] == 0 for k in t)} | {min(ep)}–{max(ep)} | "
                          f"{min(tm):.0f}–{max(tm):.0f} | {np.mean([t[k]['tot'] for k in t]):.2f} | " + ('—' if np.isnan(rn[0]) else f'{np.mean(rn):.4f}')
                          + f" | {np.mean([t[k]['rh'] for k in t]):.4f} |")
    L_ += ['', '### test MAE (mean of seeds)', '', '| input | model | arm (lr) | ' + COLS + ' |', '|---|---|---|---|---|---|---|']
    for L in LENGTHS:
        for c in MODELS:
            for arm in ('def', 'sel'):
                lr = DEFAULT if arm == 'def' else SEL[f'{c} L{L}']['selected']
                if arm == 'sel' and lr == DEFAULT:
                    continue
                t = A[c, L, arm]
                L_.append(f"| {L} | {MNAME[c]} | {lr} | " + ' | '.join(f"{np.mean([t[ds, pl, s]['mae'] for s in SEEDS]):.4f}" for ds, pl in CELLS) + ' |')
    open(a.out, 'w').write('\n'.join(L_) + '\n')
    print(a.out)


if __name__ == '__main__':
    main()
