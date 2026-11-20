"""MODI 2026-11-20: summary of hyperparameter range 1 (docs/MODI_PROJECT_LOG.md, pre-registered rules; user decision 11-20
"4-1 검증(작음) -> 범위 1(이후 모든 비교의 학습 조건을 정함) -> 4-2 (가) + (나)").

Range 1 = two training changes, together: (a) no weight decay on the BatchNorm weights / biases (--wd_exclude_1d 1, runner env WDX=1;
ETT and MIT-BIH) and (b) four validations per epoch for the best checkpoint and early stopping (--val_per_epoch 4, env VPE=4; ETT only,
MIT-BIH tests its last epoch). Everything else = the bias-off reference (default PE=0), every neuron SpikeDE, no aux loss:
  spk = Spikformer(DE) spkall_lif;  pop_lif / pop_flif = the msn base nsmt4q_<n>_msn_sde
  old (existing runs): flif-v1-ett-nope, flif-v1-mitbih-nope        new: flif-v1-ett-r1, flif-v1-mitbih-r1
  ETT: ETTh2 / ETTm2 x H96 / H720, seeds 42 / 7 / 2026, best-validation weights, test MSE; MIT-BIH: 30 epochs, last-epoch weights, macro-F1 %.
Comparisons (rules of summarize_sde.py, as in summarize_nope.py):
  R1 new - old, per model (the effect of range 1)
  R2 base(new) - Spikformer(DE)(new), per neuron (main: the comparison under the new training condition)
  R3 gap change = R2 - the bias-off comparison (N2 of summarize_nope.py)
  overall per neuron (summarize_sde.overall) from R2.
Report only: energy, firing rates, epochs, where the best checkpoint lies (inner validation or epoch end, epoch), MIT-BIH accuracy.
Fail-closed: the checks of summarize_aux.ett / mbt, both arms bias-off (pe=0, _nope); new rows: RESULT wdx=1 and folder _wdx (ETT also
vpe=4, _vpe4 and a best_epoch.txt with the step), old rows: none of these; new rows at one sha.
usage (classification folder): python summarize_r1.py [--out FILE]
"""
import argparse
import os

import numpy as np

import summarize_aux as SA
from compare_query_pop_lif_paper import CELLS, CL, COLS, FC, HERE, SEEDS, cmp_row, compare, results
from summarize_len import gap_change, gap_conclusion
from summarize_ms import NAMES, NEURONS, conclusion, mb_verdict
from summarize_sde import mb_gap, overall
from summarize_xnor import robust

SUITES = {'old': ('flif-v1-ett-nope', 'flif-v1-mitbih-nope'), 'new': ('flif-v1-ett-r1', 'flif-v1-mitbih-r1')}
MODELS = ('spk',) + tuple(NEURONS)
LABEL = {'spk': 'Spikformer(DE)', **{n: f'{NAMES[n]} base (SpikeDE)' for n in NEURONS}}


def cond(m):
    return 'spkall_lif' if m == 'spk' else f'nsmt4q_{m}_msn_sde'


def arm_check(path, arm, ett, **sel):
    rows = results(path, **sel)
    if not rows:
        raise RuntimeError(f'{path} {sel}: no rows')
    for d in rows:
        base = os.path.basename(d['dir'].rstrip('/'))
        if '_nope' not in base or d.get('pe') != '0':
            raise RuntimeError(f'{path} {sel} seed {d.get("seed")}: not bias-off ({base}, pe {d.get("pe")})')
        new = arm == 'new'
        if (('_wdx' in base) != new) or ((d.get('wdx') == '1') != new) or ('wdx' in d and d['wdx'] != '1'):
            raise RuntimeError(f'{path} {sel} seed {d.get("seed")}: weight-decay arm mismatch ({base}, wdx {d.get("wdx")})')
        if ett:
            be = d['best_epoch_val_epochs'].split(':')
            if (('_vpe4' in base) != new) or ((d.get('vpe') == '4') != new) or ((len(be) == 4) != new):
                raise RuntimeError(f'{path} {sel} seed {d.get("seed")}: validation arm mismatch ({base}, vpe {d.get("vpe")}, {be})')
        elif 'vpe' in d or '_vpe' in base:
            raise RuntimeError(f'{path} {sel}: vpe on MIT-BIH')
    return rows


def best_pos(path, **sel):
    """(epoch of the best checkpoint, True if it is an inner validation) per (ds, pl, seed) of the new ETT rows."""
    out = {}
    for d in results(path, **sel):
        be = d['best_epoch_val_epochs'].split(':')
        step, n = (int(v) for v in be[3].split('/'))
        out[d['ds'], int(d['pl']), int(d['seed'])] = (int(be[0]), step < n, step / n)
    return out


def load(m, arm):
    se, sm = SUITES[arm]
    c = cond(m)
    e, s1 = SA.ett(c, se, True, False)
    sel = {} if m == 'spk' else dict(lr='0.01', val='0')
    mb, s2 = SA.mbt(c, sm, True, False, **sel)
    arm_check(os.path.join(FC, se, 'raw.txt'), arm, True, cond=c)
    arm_check(os.path.join(CL, sm, 'raw.txt'), arm, False, cond=c, **sel)
    return e, mb, s1 | s2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'flif-v1-r1-summary.md'))
    a = ap.parse_args()
    E, M, shas = {}, {}, set()
    for m in MODELS:
        for arm in SUITES:
            E[m, arm], M[m, arm], s = load(m, arm)
            if arm == 'new':
                shas |= s
    if len(shas) != 1:
        raise RuntimeError(f'new rows at several shas {shas}')
    L = ['# Hyperparameter range 1 (BN / bias without weight decay; ETT four validations per epoch) — summary', '',
         f'See the module docstring and docs/MODI_PROJECT_LOG.md for the pre-registered rules. New runs at sha {shas.pop()}.', '',
         '## Results (mean ± sd of seeds; ETT test MSE, MIT-BIH test macro-F1 %)', '', '| model | training | ' + COLS + ' | MIT-BIH macro-F1 % |',
         '|---|---|---|---|---|---|---|']
    for m in MODELS:
        for arm in SUITES:
            e, mb = E[m, arm], M[m, arm]
            cells = ' | '.join(f"{np.mean([e[ds, pl, s]['mse'] for s in SEEDS]):.4f} ± {np.std([e[ds, pl, s]['mse'] for s in SEEDS], ddof=1):.4f}" for ds, pl in CELLS)
            L.append(f"| {LABEL[m]} | {arm} | {cells} | {np.mean([mb[s]['f1'] for s in SEEDS]):.2f} ± {np.std([mb[s]['f1'] for s in SEEDS], ddof=1):.2f} |")
    C = {}
    L += ['', '## ETT (r = A / B − 1 of test MSE, %)', '', '| comparison | ' + COLS + ' | R % | pairs A lower | verdict | energy A/B | leave-one-cell-out |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for m in MODELS:
        C['R1', m] = x = compare(E[m, 'new'], E[m, 'old'])
        L.append(cmp_row(f'R1 {LABEL[m]}: new − old', x) + f" {robust(x, x['v'], x['R'])} |")
    for n in NEURONS:
        C['R2', n] = x = compare(E[n, 'new'], E['spk', 'new'])
        L.append(cmp_row(f'R2 {NAMES[n]}: base − Spikformer(DE), new training', x) + f" {robust(x, x['v'], x['R'])} |")
        C['ref', n] = x = compare(E[n, 'old'], E['spk', 'old'])
        L.append(cmp_row(f'(reference) {NAMES[n]}: base − Spikformer(DE), old training (bias-off N2)', x) + f" {robust(x, x['v'], x['R'])} |")
    L += ['', '| comparison | ' + COLS + ' | mean g | pairs negative | verdict | leave-one-cell-out |', '|---|---|---|---|---|---|---|---|---|']
    for n in NEURONS:
        g = gap_change(C['R2', n], C['ref', n]); C['R3', n] = g
        L.append(f"| R3 gap change {NAMES[n]} (negative = the base gains with the new training) | " + ' | '.join(f"{np.mean(g['per'][c]):+.2f}" for c in CELLS)
                 + f" | {g['G']:+.2f} | {g['neg']}/12 | {g['v']} | {robust(g, g['v'], g['G'])} |")
    L += ['', '## MIT-BIH (d = A − B of test macro-F1, %p per seed 42 / 7 / 2026)', '', '| comparison | d per seed | mean | verdict | energy A/B |', '|---|---|---|---|---|']

    def mrow(name, x, ea, eb):
        e = np.mean([ea[s]['tot'] for s in SEEDS]) / np.mean([eb[s]['tot'] for s in SEEDS])
        return f"| {name} | {' / '.join(f'{v:+.2f}' for v in x[0])} | {x[1]:+.2f} | {x[2]} | {e:.2f} |"
    for m in MODELS:
        C['M1', m] = mb_verdict(M[m, 'new'], M[m, 'old'])
        L.append(mrow(f'R1 {LABEL[m]}: new − old', C['M1', m], M[m, 'new'], M[m, 'old']))
    for n in NEURONS:
        C['M2', n] = mb_verdict(M[n, 'new'], M['spk', 'new'])
        L.append(mrow(f'R2 {NAMES[n]}: base − Spikformer(DE), new training', C['M2', n], M[n, 'new'], M['spk', 'new']))
        C['Mref', n] = mb_verdict(M[n, 'old'], M['spk', 'old'])
        L.append(mrow(f'(reference) {NAMES[n]}: base − Spikformer(DE), old training', C['Mref', n], M[n, 'old'], M['spk', 'old']))
        g, mm, vv = mb_gap(M[n, 'new'], M['spk', 'new'], M[n, 'old'], M['spk', 'old']); C['M3', n] = vv
        L.append(f"| R3 gap change {NAMES[n]} (positive = the base gains with the new training) | {' / '.join(f'{v:+.2f}' for v in g)} | {mm:+.2f} | {vv} | |")
    L += ['', '## conclusions (rule of the pre-registration)', '', '| comparison | task | Spikformer(DE) | pop LIF | pop f-LIF | conclusion over the two neurons |', '|---|---|---|---|---|---|']
    L.append(f"| R1 new − old | ETT | {C['R1', 'spk']['v']} | {C['R1', 'pop_lif']['v']} | {C['R1', 'pop_flif']['v']} | (per model) |")
    L.append(f"| R1 new − old | MIT-BIH | {C['M1', 'spk'][2]} | {C['M1', 'pop_lif'][2]} | {C['M1', 'pop_flif'][2]} | (per model) |")
    L.append(f"| R2 base − Spikformer(DE), new training | ETT | — | {C['R2', 'pop_lif']['v']} | {C['R2', 'pop_flif']['v']} | {conclusion([C['R2', n]['v'] for n in NEURONS])} |")
    L.append(f"| R2 base − Spikformer(DE), new training | MIT-BIH | — | {C['M2', 'pop_lif'][2]} | {C['M2', 'pop_flif'][2]} | {conclusion([C['M2', n][2] for n in NEURONS])} |")
    L.append(f"| R3 gap change | ETT | — | {C['R3', 'pop_lif']['v']} | {C['R3', 'pop_flif']['v']} | {gap_conclusion([C['R3', n]['v'] for n in NEURONS])} |")
    L.append(f"| R3 gap change | MIT-BIH | — | {C['M3', 'pop_lif']} | {C['M3', 'pop_flif']} | {gap_conclusion([C['M3', n] for n in NEURONS])} |")
    L.append(f"| overall: base better than Spikformer(DE), new training | both | — | {overall(C['R2', 'pop_lif'], C['M2', 'pop_lif'])} | {overall(C['R2', 'pop_flif'], C['M2', 'pop_flif'])} | — |")
    L += ['', '## report only', '', '| model | training | ETT energy µJ / Spikformer(DE) same arm | MIT-BIH energy µJ / Spikformer(DE) same arm | ETT rho_hippo | ETT rho_neo | '
          'MIT-BIH rho_hippo | MIT-BIH rho_neo | MIT-BIH accuracy % | ETT best at epoch 0 (of 12) | ETT epochs (min–max) | ETT best at an inner validation (of 12) |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    f = lambda v: '—' if np.isnan(v) else f'{v:.4f}'
    for m in MODELS:
        for arm in SUITES:
            e, mb, se, sm = E[m, arm], M[m, arm], E['spk', arm], M['spk', arm]
            er = np.mean([np.mean([e[ds, pl, s]['tot'] for s in SEEDS]) / np.mean([se[ds, pl, s]['tot'] for s in SEEDS]) for ds, pl in CELLS])
            mr = np.mean([mb[s]['tot'] for s in SEEDS]) / np.mean([sm[s]['tot'] for s in SEEDS])
            inner = '—'
            if arm == 'new':
                bp = best_pos(os.path.join(FC, SUITES['new'][0], 'raw.txt'), cond=cond(m))
                inner = str(sum(v[1] for v in bp.values()))
            L.append(f"| {LABEL[m]} | {arm} | {np.mean([e[q]['tot'] for q in e]):.2f} / {er:.2f} | {np.mean([mb[s]['tot'] for s in SEEDS]):.2f} / {mr:.2f} | "
                     f"{f(np.mean([e[q]['rh'] for q in e]))} | {f(np.mean([e[q]['rn'] for q in e]))} | {f(np.mean([mb[s]['rh'] for s in SEEDS]))} | "
                     f"{f(np.mean([mb[s]['rn'] for s in SEEDS]))} | {np.mean([mb[s]['acc'] for s in SEEDS]):.2f} | {sum(e[q]['be'] == 0 for q in e)} | "
                     f"{min(e[q]['ep'] for q in e)}–{max(e[q]['ep'] for q in e)} | {inner} |")
    open(a.out, 'w').write('\n'.join(L) + '\n')
    print(a.out)


if __name__ == '__main__':
    main()
