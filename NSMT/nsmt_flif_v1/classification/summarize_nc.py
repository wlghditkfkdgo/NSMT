"""MODI 2026-11-21: summary of 4-2, the coupling placement test (docs/MODI_PROJECT_LOG.md, pre-registered rules; user decision 11-20
"4-1 검증(작음) -> 범위 1(이후 모든 비교의 학습 조건을 정함) -> 4-2 (가) + (나)").

Models (every neuron SpikeDE, no relative position bias, no aux loss, range-1 training = the runner defaults since 11-21):
  spk  = Spikformer(DE) spkall_lif                                  existing runs flif-v1-ett-r1 / flif-v1-mitbih-r1
  base = the query base nsmt4q_pop_<n>_msn_sde                      existing runs flif-v1-ett-r1 / flif-v1-mitbih-r1
  ga   = (가) nsmt4q_pop_<n>_msn_sde_nc   (--neo_current 1: Neocortex output x learned per-channel alpha into the hippocampus
         embedding neuron's input current; attention kept)        new runs flif-v1-ett-nc / flif-v1-mitbih-nc
  na   = (나) nsmt4q_pop_<n>_msn_sde_ncna (+ --hippo_attn 0: no hippocampus attention; the Neocortex is the only token mixer)
  n = pop LIF, pop f-LIF; ETT ETTh2 / ETTm2 x H96 / H720 x seeds 42 / 7 / 2026 (best-validation weights, test MSE); MIT-BIH seeds
  42 / 7 / 2026 (30 epochs, last-epoch weights, macro-F1 %).
Comparisons (rules of summarize_sde.py: ETT 12 pairs, R <= -1 % with 8/12 lower = better, R >= +1 % with 8/12 higher = worse;
MIT-BIH mean d >= +0.5 %p with 2/3 seeds positive = better, the mirror = worse; two neurons: summarize_ms.conclusion):
  G1 (가) - base      G2 (나) - base      per neuron
  G3 (가) - Spikformer(DE)   G4 (나) - Spikformer(DE)   per neuron, + overall (summarize_sde.overall) and energy ratios
Recommendation (pre-registered; the structure decision is the user's):
  1. (나) has no "worse" in G2 (2 neurons x 2 tasks)                    -> recommend (나)
  2. else (가) has a "better" and no "worse" in G1                      -> recommend (가)
  3. else                                                              -> keep the base
Report only: alpha after training (mean, mean |alpha|, share < 0) from the stored weights; energy (ETT recounted head; MIT-BIH as
logged) and an upper bound of the uncounted alpha addition (C x T_h x N x D additions x 0.9 pJ, every Neocortex output spike at every
token); firing rates; epochs; best checkpoint at an inner validation; MIT-BIH accuracy.
Fail-closed: the checks of summarize_aux.ett / mbt; every row bias-off (pe=0, _nope), range-1 (wdx=1, _wdx; ETT vpe=4, _vpe4, 4-field
best epoch); new rows: folder _nc, _noattn exactly for (나), one sha; reference rows: no _nc; alpha present exactly in the new models.
usage (classification folder): python summarize_nc.py [--out FILE]
"""
import argparse
import os

import numpy as np
import torch

import summarize_aux as SA
from compare_query_pop_lif_paper import CELLS, CL, COLS, FC, HERE, SEEDS, cmp_row, compare, results
from summarize_ms import NAMES, NEURONS, conclusion, mb_verdict
from summarize_sde import overall
from summarize_xnor import robust

SUITES = {'ref': ('flif-v1-ett-r1', 'flif-v1-mitbih-r1'), 'new': ('flif-v1-ett-nc', 'flif-v1-mitbih-nc')}
LABEL = {'spk': 'Spikformer(DE)', 'base': 'base', 'ga': '(가) Neocortex current + attention', 'na': '(나) Neocortex current, no attention'}
E_AC = 0.9e-6                                                       # µJ per addition


def cond(kind, n=None):
    return {'spk': 'spkall_lif', 'base': f'nsmt4q_{n}_msn_sde', 'ga': f'nsmt4q_{n}_msn_sde_nc', 'na': f'nsmt4q_{n}_msn_sde_ncna'}[kind]


def arm_check(path, kind, ett, **sel):
    rows = results(path, **sel)
    if not rows:
        raise RuntimeError(f'{path} {sel}: no rows')
    for d in rows:
        base = os.path.basename(d['dir'].rstrip('/'))
        if '_nope' not in base or d.get('pe') != '0':
            raise RuntimeError(f'{path} {sel} seed {d.get("seed")}: not bias-off ({base}, pe {d.get("pe")})')
        if '_wdx' not in base or d.get('wdx') != '1':
            raise RuntimeError(f'{path} {sel} seed {d.get("seed")}: not range-1 weight decay ({base}, wdx {d.get("wdx")})')
        if ett:
            be = d['best_epoch_val_epochs'].split(':')
            if '_vpe4' not in base or d.get('vpe') != '4' or len(be) != 4:
                raise RuntimeError(f'{path} {sel} seed {d.get("seed")}: not range-1 validation ({base}, vpe {d.get("vpe")}, {be})')
        elif 'vpe' in d and d['vpe'] != '1' or '_vpe' in base:
            raise RuntimeError(f'{path} {sel}: vpe on MIT-BIH')
        if ('_nc' in base) != (kind in ('ga', 'na')) or ('_noattn' in base) != (kind == 'na'):
            raise RuntimeError(f'{path} {sel} seed {d.get("seed")}: structure suffix mismatch for {kind} ({base})')
    return rows


def best_inner(path, **sel):
    n = 0
    for d in results(path, **sel):
        step, tot = (int(v) for v in d['best_epoch_val_epochs'].split(':')[3].split('/'))
        n += step < tot
    return n


def alpha_stats(path, kind, ett, **sel):
    """(mean alpha, mean |alpha|, share < 0) over the stored weights of every row; NaN-free or stop."""
    out = []
    for d in results(path, **sel):
        f = os.path.join(d['dir'], 'model_state', 'checkpoint.pth' if ett else 'best+model.pt')
        w = torch.load(f, map_location='cpu')
        keys = [k for k in w if k.endswith('neo_alpha_c')]
        if kind in ('ga', 'na') and len(keys) != 1 or kind not in ('ga', 'na') and keys:
            raise RuntimeError(f'{f}: alpha keys {keys} for {kind}')
        if keys:
            a = w[keys[0]].float()
            if not torch.isfinite(a).all():
                raise RuntimeError(f'{f}: non-finite alpha')
            out.append((a.mean().item(), a.abs().mean().item(), (a < 0).float().mean().item(), a.numel()))
    return out


def load(kind, n):
    arm = 'ref' if kind in ('spk', 'base') else 'new'
    se, sm = SUITES[arm]
    c = cond(kind, n)
    e, s1 = SA.ett(c, se, True, False)
    sel = {} if kind == 'spk' else dict(lr='0.01', val='0')
    mb, s2 = SA.mbt(c, sm, True, False, **sel)
    arm_check(os.path.join(FC, se, 'raw.txt'), kind, True, cond=c)
    arm_check(os.path.join(CL, sm, 'raw.txt'), kind, False, cond=c, **sel)
    return e, mb, s1, s2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'flif-v1-nc-summary.md'))
    a = ap.parse_args()
    E, M, sha = {}, {}, {'ref': set(), 'new': set()}
    E['spk'], M['spk'], s1, s2 = load('spk', None); sha['ref'] |= s1 | s2
    for n in NEURONS:
        for kind in ('base', 'ga', 'na'):
            E[kind, n], M[kind, n], s1, s2 = load(kind, n)
            sha['ref' if kind == 'base' else 'new'] |= s1 | s2
    if len(sha['new']) != 1 or len(sha['ref']) != 1:
        raise RuntimeError(f'shas: new {sha["new"]}, reference {sha["ref"]}')
    L = ['# 4-2 coupling placement ((가) Neocortex current + attention, (나) Neocortex current without attention) — summary', '',
         f"See the module docstring and docs/MODI_PROJECT_LOG.md for the pre-registered rules. New runs at sha {next(iter(sha['new']))}; "
         f"reference runs (range 1) at sha {next(iter(sha['ref']))}.", '',
         '## Results (mean ± sd of seeds; ETT test MSE, MIT-BIH test macro-F1 %)', '', '| model | ' + COLS + ' | MIT-BIH macro-F1 % |',
         '|---|---|---|---|---|---|']

    def res_row(name, e, mb):
        cells = ' | '.join(f"{np.mean([e[ds, pl, s]['mse'] for s in SEEDS]):.4f} ± {np.std([e[ds, pl, s]['mse'] for s in SEEDS], ddof=1):.4f}" for ds, pl in CELLS)
        return f"| {name} | {cells} | {np.mean([mb[s]['f1'] for s in SEEDS]):.2f} ± {np.std([mb[s]['f1'] for s in SEEDS], ddof=1):.2f} |"
    L.append(res_row(LABEL['spk'], E['spk'], M['spk']))
    for n in NEURONS:
        for kind in ('base', 'ga', 'na'):
            L.append(res_row(f'{NAMES[n]} {LABEL[kind]}', E[kind, n], M[kind, n]))
    C = {}
    L += ['', '## ETT (r = A / B − 1 of test MSE, %)', '', '| comparison | ' + COLS + ' | R % | pairs A lower | verdict | energy A/B | leave-one-cell-out |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for g, kind, ref in (('G1', 'ga', 'base'), ('G2', 'na', 'base'), ('G3', 'ga', 'spk'), ('G4', 'na', 'spk')):
        for n in NEURONS:
            B = E['spk'] if ref == 'spk' else E[ref, n]
            C[g, 'ett', n] = x = compare(E[kind, n], B)
            L.append(cmp_row(f"{g} {NAMES[n]}: {LABEL[kind]} − {LABEL[ref] if ref == 'spk' else 'base'}", x) + f" {robust(x, x['v'], x['R'])} |")
    L += ['', '## MIT-BIH (d = A − B of test macro-F1, %p per seed 42 / 7 / 2026)', '', '| comparison | d per seed | mean | verdict | energy A/B |', '|---|---|---|---|---|']
    for g, kind, ref in (('G1', 'ga', 'base'), ('G2', 'na', 'base'), ('G3', 'ga', 'spk'), ('G4', 'na', 'spk')):
        for n in NEURONS:
            B = M['spk'] if ref == 'spk' else M[ref, n]
            C[g, 'mb', n] = x = mb_verdict(M[kind, n], B)
            er = np.mean([M[kind, n][s]['tot'] for s in SEEDS]) / np.mean([B[s]['tot'] for s in SEEDS])
            L.append(f"| {g} {NAMES[n]}: {LABEL[kind]} − {LABEL[ref] if ref == 'spk' else 'base'} | {' / '.join(f'{v:+.2f}' for v in x[0])} | "
                     f"{x[1]:+.2f} | {x[2]} | {er:.2f} |")
    L += ['', '## conclusions (rules of the pre-registration)', '', '| comparison | task | pop LIF | pop f-LIF | conclusion over the two neurons |', '|---|---|---|---|---|']
    for g, what in (('G1', '(가) − base'), ('G2', '(나) − base'), ('G3', '(가) − Spikformer(DE)'), ('G4', '(나) − Spikformer(DE)')):
        ve = [C[g, 'ett', n]['v'] for n in NEURONS]; vm = [C[g, 'mb', n][2] for n in NEURONS]
        L.append(f"| {g} {what} | ETT | {ve[0]} | {ve[1]} | {conclusion(ve)} |")
        L.append(f"| {g} {what} | MIT-BIH | {vm[0]} | {vm[1]} | {conclusion(vm)} |")
    for g, what in (('G3', '(가)'), ('G4', '(나)')):
        L.append(f"| overall: {what} better than Spikformer(DE) | both | {overall(C[g, 'ett', NEURONS[0]], C[g, 'mb', NEURONS[0]])} | "
                 f"{overall(C[g, 'ett', NEURONS[1]], C[g, 'mb', NEURONS[1]])} | — |")
    g2 = [C['G2', t, n]['v'] if t == 'ett' else C['G2', t, n][2] for t in ('ett', 'mb') for n in NEURONS]
    g1 = [C['G1', t, n]['v'] if t == 'ett' else C['G1', t, n][2] for t in ('ett', 'mb') for n in NEURONS]
    if 'worse' not in g2:
        rec = '(나): no "worse" in G2'
    elif 'better' in g1 and 'worse' not in g1:
        rec = '(가): G2 has a "worse"; G1 has a "better" and no "worse"'
    else:
        rec = 'keep the base: G2 has a "worse" and G1 has no "better" without a "worse"'
    L += ['', f'**Recommendation (pre-registered rule; the structure decision is the user\'s):** {rec}. G2 verdicts (ETT pop LIF, ETT pop f-LIF, '
          f'MIT-BIH pop LIF, MIT-BIH pop f-LIF): {", ".join(g2)}; G1: {", ".join(g1)}.']
    L += ['', '## report only', '', '| model | ETT energy µJ / Spikformer(DE) | MIT-BIH energy µJ / Spikformer(DE) | ETT rho_hippo | ETT rho_neo | MIT-BIH rho_hippo | '
          'MIT-BIH rho_neo | MIT-BIH accuracy % | ETT best at epoch 0 (of 12) | ETT epochs (min–max) | ETT best at an inner validation (of 12) |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    f = lambda v: '—' if np.isnan(v) else f'{v:.4f}'
    rows = [('spk', None)] + [(k, n) for n in NEURONS for k in ('base', 'ga', 'na')]
    for kind, n in rows:
        e, mb = (E['spk'], M['spk']) if kind == 'spk' else (E[kind, n], M[kind, n])
        er = np.mean([np.mean([e[ds, pl, s]['tot'] for s in SEEDS]) / np.mean([E['spk'][ds, pl, s]['tot'] for s in SEEDS]) for ds, pl in CELLS])
        mr = np.mean([mb[s]['tot'] for s in SEEDS]) / np.mean([M['spk'][s]['tot'] for s in SEEDS])
        arm = 'ref' if kind in ('spk', 'base') else 'new'
        inner = best_inner(os.path.join(FC, SUITES[arm][0], 'raw.txt'), cond=cond(kind, n))
        name = LABEL['spk'] if kind == 'spk' else f'{NAMES[n]} {LABEL[kind]}'
        L.append(f"| {name} | {np.mean([e[q]['tot'] for q in e]):.2f} / {er:.2f} | {np.mean([mb[s]['tot'] for s in SEEDS]):.2f} / {mr:.2f} | "
                 f"{f(np.mean([e[q]['rh'] for q in e]))} | {f(np.mean([e[q]['rn'] for q in e]))} | {f(np.mean([mb[s]['rh'] for s in SEEDS]))} | "
                 f"{f(np.mean([mb[s]['rn'] for s in SEEDS]))} | {np.mean([mb[s]['acc'] for s in SEEDS]):.2f} | {sum(e[q]['be'] == 0 for q in e)} | "
                 f"{min(e[q]['ep'] for q in e)}–{max(e[q]['ep'] for q in e)} | {inner} |")
    L += ['', '| model | ETT alpha mean | ETT mean abs alpha | ETT share alpha < 0 | MIT-BIH alpha mean | MIT-BIH mean abs alpha | MIT-BIH share alpha < 0 |',
          '|---|---|---|---|---|---|---|']
    for kind, n in rows:
        arm = 'ref' if kind in ('spk', 'base') else 'new'
        ae = alpha_stats(os.path.join(FC, SUITES[arm][0], 'raw.txt'), kind, True, cond=cond(kind, n))
        am = alpha_stats(os.path.join(CL, SUITES[arm][1], 'raw.txt'), kind, False, cond=cond(kind, n), **({} if kind == 'spk' else dict(lr='0.01', val='0')))
        if kind not in ('ga', 'na'):
            continue
        if len(ae) != 12 or len(am) != 3:
            raise RuntimeError(f'{kind} {n}: alpha rows {len(ae)} / {len(am)}')
        cell = lambda v, i: f"{np.mean([x[i] for x in v]):+.3f}" if i < 2 else f"{np.mean([x[i] for x in v]):.2f}"
        L.append(f"| {NAMES[n]} {LABEL[kind]} | {cell(ae, 0)} | {cell(ae, 1)} | {cell(ae, 2)} | {cell(am, 0)} | {cell(am, 1)} | {cell(am, 2)} |")
    # upper bound of the uncounted alpha addition: C rows x T_h steps x N tokens x D channels, one addition each
    L += ['', 'Uncounted alpha addition (syops counts modules; alpha * spikes is not a module): upper bound with every Neocortex output unit '
          'firing at every token, C x T_h x N x D additions x 0.9 pJ.', '', '| task | C x T_h x N x D | bound µJ | share of the smallest (나) energy |', '|---|---|---|---|']
    for task, (C_, T_, N_, D_) in (('ETT (7 channels, 24 tokens, D 64)', (7, 4, 24, 64)), ('MIT-BIH (11 tokens, D 384)', (1, 4, 11, 384))):
        ops = C_ * T_ * N_ * D_
        if task.startswith('ETT'):
            low = min(np.mean([E['na', n][q]['tot'] for q in E['na', n]]) for n in NEURONS)
        else:
            low = min(np.mean([M['na', n][s]['tot'] for s in SEEDS]) for n in NEURONS)
        L.append(f"| {task} | {ops} | {ops * E_AC:.4f} | {100 * ops * E_AC / low:.2f} % |")
    open(a.out, 'w').write('\n'.join(L) + '\n')
    print(a.out)


if __name__ == '__main__':
    main()
