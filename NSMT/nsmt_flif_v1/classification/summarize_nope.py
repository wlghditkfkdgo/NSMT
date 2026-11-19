"""MODI 2026-11-19: summary of the positional-encoding-off test (docs/MODI_PROJECT_LOG.md, pre-registered rules; user request 11-19
"relative positional encoding을 끄고 실험을 해줘. (Spikformer 도 마찬가지)").

The learned relative position bias of the attention is removed (--ssa_pe 0, runner env PE=0) and the models are retrained; the absolute
position of the embedding was already off. Every neuron SpikeDE (user decision 11-18 "(가)"), current training setting, no aux loss:
  spk = Spikformer(DE) spkall_lif;  pop_lif / pop_flif = the msn base nsmt4q_<n>_msn_sde
  on  (existing runs): flif-v1-ett-sde, flif-v1-mitbih-sde        off (new runs): flif-v1-ett-nope, flif-v1-mitbih-nope
  ETT: ETTh2 / ETTm2 x H96 / H720, seeds 42 / 7 / 2026, best-validation weights, test MSE; MIT-BIH: 30 epochs, last-epoch weights, macro-F1 %.
Comparisons (rules of summarize_sde.py):
  N1 off - on, per model (ETT: r = A / B - 1 over 12 pairs, better = mean <= -1 % and >= 8/12 lower, worse = mirror;
     MIT-BIH: d per seed, better = mean >= +0.5 %p and >= 2/3 positive, worse = mirror)
  N2 base(off) - Spikformer(DE)(off), per neuron (main); reference = the same with the bias on
  N3 gap change = N2 - reference (ETT: grew = mean <= -1 %p and >= 8/12 negative; MIT-BIH: grew = mean >= +0.5 %p and >= 2/3 positive)
  overall per neuron (summarize_sde.overall) from N2.
Report only: energy, firing rates, epochs, first-epoch stops, MIT-BIH accuracy.
Fail-closed: the checks of summarize_aux.ett / mbt (rows, seeds, SpikeDE suffix, no lr field, 30 epochs, aux 0), and: off rows need the
folder suffix _nope and RESULT pe=0, on rows must have neither; new rows at one sha.
usage (classification folder): python summarize_nope.py [--out FILE]
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

SUITES = {'on': ('flif-v1-ett-sde', 'flif-v1-mitbih-sde'), 'off': ('flif-v1-ett-nope', 'flif-v1-mitbih-nope')}
MODELS = ('spk',) + tuple(NEURONS)
LABEL = {'spk': 'Spikformer(DE)', **{n: f'{NAMES[n]} base (SpikeDE)' for n in NEURONS}}


def cond(m):
    return 'spkall_lif' if m == 'spk' else f'nsmt4q_{m}_msn_sde'


def pe_check(path, arm, **sel):
    rows = results(path, **sel)
    if not rows:
        raise RuntimeError(f'{path} {sel}: no rows')
    for d in rows:
        nope, pe = '_nope' in os.path.basename(d['dir'].rstrip('/')), d.get('pe')
        if (arm == 'off') != nope or pe != ('0' if arm == 'off' else None):
            raise RuntimeError(f'{path} {sel} seed {d.get("seed")}: arm {arm} but folder suffix _nope {nope}, RESULT pe {pe} ({d["dir"]})')


def load(m, arm):
    se, sm = SUITES[arm]
    c = cond(m)
    e, s1 = SA.ett(c, se, True, False)
    sel = {} if m == 'spk' else dict(lr='0.01', val='0')
    mb, s2 = SA.mbt(c, sm, True, False, **sel)
    pe_check(os.path.join(FC, se, 'raw.txt'), arm, cond=c)
    pe_check(os.path.join(CL, sm, 'raw.txt'), arm, cond=c, **sel)
    return e, mb, s1 | s2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'flif-v1-nope-summary.md'))
    a = ap.parse_args()
    E, M, shas = {}, {}, set()
    for m in MODELS:
        for arm in SUITES:
            E[m, arm], M[m, arm], s = load(m, arm)
            if arm == 'off':
                shas |= s
    if len(shas) != 1:
        raise RuntimeError(f'new rows at several shas {shas}')
    L = ['# Positional-encoding-off test (relative position bias of the attention removed, retrained; every neuron SpikeDE) — summary', '',
         f'See the module docstring and docs/MODI_PROJECT_LOG.md for the pre-registered rules. New runs at sha {shas.pop()}.', '',
         '## Results (mean ± sd of seeds; ETT test MSE, MIT-BIH test macro-F1 %)', '', '| model | bias | ' + COLS + ' | MIT-BIH macro-F1 % |', '|---|---|---|---|---|---|---|']
    for m in MODELS:
        for arm in SUITES:
            e, mb = E[m, arm], M[m, arm]
            cells = ' | '.join(f"{np.mean([e[ds, pl, s]['mse'] for s in SEEDS]):.4f} ± {np.std([e[ds, pl, s]['mse'] for s in SEEDS], ddof=1):.4f}" for ds, pl in CELLS)
            L.append(f"| {LABEL[m]} | {arm} | {cells} | {np.mean([mb[s]['f1'] for s in SEEDS]):.2f} ± {np.std([mb[s]['f1'] for s in SEEDS], ddof=1):.2f} |")
    C = {}
    L += ['', '## ETT (r = A / B − 1 of test MSE, %)', '', '| comparison | ' + COLS + ' | R % | pairs A lower | verdict | energy A/B | leave-one-cell-out |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for m in MODELS:
        C['N1', m] = x = compare(E[m, 'off'], E[m, 'on'])
        L.append(cmp_row(f'N1 {LABEL[m]}: off − on', x) + f" {robust(x, x['v'], x['R'])} |")
    for n in NEURONS:
        C['N2', n] = x = compare(E[n, 'off'], E['spk', 'off'])
        L.append(cmp_row(f'N2 {NAMES[n]}: base (off) − Spikformer(DE) (off)', x) + f" {robust(x, x['v'], x['R'])} |")
        C['ref', n] = x = compare(E[n, 'on'], E['spk', 'on'])
        L.append(cmp_row(f'(reference) {NAMES[n]}: base − Spikformer(DE), bias on', x) + f" {robust(x, x['v'], x['R'])} |")
    L += ['', '| comparison | ' + COLS + ' | mean g | pairs negative | verdict | leave-one-cell-out |', '|---|---|---|---|---|---|---|---|---|']
    for n in NEURONS:
        g = gap_change(C['N2', n], C['ref', n]); C['N3', n] = g
        L.append(f"| N3 gap change {NAMES[n]} (negative = the base gains when the bias is removed) | " + ' | '.join(f"{np.mean(g['per'][c]):+.2f}" for c in CELLS)
                 + f" | {g['G']:+.2f} | {g['neg']}/12 | {g['v']} | {robust(g, g['v'], g['G'])} |")
    L += ['', '## MIT-BIH (d = A − B of test macro-F1, %p per seed 42 / 7 / 2026)', '', '| comparison | d per seed | mean | verdict | energy A/B |', '|---|---|---|---|---|']

    def mrow(name, x, ea, eb):
        e = np.mean([ea[s]['tot'] for s in SEEDS]) / np.mean([eb[s]['tot'] for s in SEEDS])
        return f"| {name} | {' / '.join(f'{v:+.2f}' for v in x[0])} | {x[1]:+.2f} | {x[2]} | {e:.2f} |"
    for m in MODELS:
        C['M1', m] = mb_verdict(M[m, 'off'], M[m, 'on'])
        L.append(mrow(f'N1 {LABEL[m]}: off − on', C['M1', m], M[m, 'off'], M[m, 'on']))
    for n in NEURONS:
        C['M2', n] = mb_verdict(M[n, 'off'], M['spk', 'off'])
        L.append(mrow(f'N2 {NAMES[n]}: base (off) − Spikformer(DE) (off)', C['M2', n], M[n, 'off'], M['spk', 'off']))
        C['Mref', n] = mb_verdict(M[n, 'on'], M['spk', 'on'])
        L.append(mrow(f'(reference) {NAMES[n]}: base − Spikformer(DE), bias on', C['Mref', n], M[n, 'on'], M['spk', 'on']))
        g, mm, vv = mb_gap(M[n, 'off'], M['spk', 'off'], M[n, 'on'], M['spk', 'on']); C['M3', n] = vv
        L.append(f"| N3 gap change {NAMES[n]} (positive = the base gains when the bias is removed) | {' / '.join(f'{v:+.2f}' for v in g)} | {mm:+.2f} | {vv} | |")
    L += ['', '## conclusions (rule of the pre-registration)', '', '| comparison | task | Spikformer(DE) | pop LIF | pop f-LIF | conclusion over the two neurons |', '|---|---|---|---|---|---|']
    L.append(f"| N1 off − on | ETT | {C['N1', 'spk']['v']} | {C['N1', 'pop_lif']['v']} | {C['N1', 'pop_flif']['v']} | (per model) |")
    L.append(f"| N1 off − on | MIT-BIH | {C['M1', 'spk'][2]} | {C['M1', 'pop_lif'][2]} | {C['M1', 'pop_flif'][2]} | (per model) |")
    L.append(f"| N2 base − Spikformer(DE), bias off | ETT | — | {C['N2', 'pop_lif']['v']} | {C['N2', 'pop_flif']['v']} | {conclusion([C['N2', n]['v'] for n in NEURONS])} |")
    L.append(f"| N2 base − Spikformer(DE), bias off | MIT-BIH | — | {C['M2', 'pop_lif'][2]} | {C['M2', 'pop_flif'][2]} | {conclusion([C['M2', n][2] for n in NEURONS])} |")
    L.append(f"| N3 gap change | ETT | — | {C['N3', 'pop_lif']['v']} | {C['N3', 'pop_flif']['v']} | {gap_conclusion([C['N3', n]['v'] for n in NEURONS])} |")
    L.append(f"| N3 gap change | MIT-BIH | — | {C['M3', 'pop_lif']} | {C['M3', 'pop_flif']} | {gap_conclusion([C['M3', n] for n in NEURONS])} |")
    L.append(f"| overall: base better than Spikformer(DE), bias off | both | — | {overall(C['N2', 'pop_lif'], C['M2', 'pop_lif'])} | {overall(C['N2', 'pop_flif'], C['M2', 'pop_flif'])} | — |")
    L += ['', '## report only', '', '| model | bias | ETT energy µJ / Spikformer(DE) same arm | MIT-BIH energy µJ / Spikformer(DE) same arm | ETT rho_hippo | ETT rho_neo | '
          'MIT-BIH rho_hippo | MIT-BIH rho_neo | MIT-BIH accuracy % | ETT best validation at epoch 0 (of 12) | ETT epochs (min–max) |', '|---|---|---|---|---|---|---|---|---|---|---|']
    f = lambda v: '—' if np.isnan(v) else f'{v:.4f}'
    for m in MODELS:
        for arm in SUITES:
            e, mb, se, sm = E[m, arm], M[m, arm], E['spk', arm], M['spk', arm]
            er = np.mean([np.mean([e[ds, pl, s]['tot'] for s in SEEDS]) / np.mean([se[ds, pl, s]['tot'] for s in SEEDS]) for ds, pl in CELLS])
            mr = np.mean([mb[s]['tot'] for s in SEEDS]) / np.mean([sm[s]['tot'] for s in SEEDS])
            L.append(f"| {LABEL[m]} | {arm} | {np.mean([e[q]['tot'] for q in e]):.2f} / {er:.2f} | {np.mean([mb[s]['tot'] for s in SEEDS]):.2f} / {mr:.2f} | "
                     f"{f(np.mean([e[q]['rh'] for q in e]))} | {f(np.mean([e[q]['rn'] for q in e]))} | {f(np.mean([mb[s]['rh'] for s in SEEDS]))} | "
                     f"{f(np.mean([mb[s]['rn'] for s in SEEDS]))} | {np.mean([mb[s]['acc'] for s in SEEDS]):.2f} | {sum(e[q]['be'] == 0 for q in e)} | "
                     f"{min(e[q]['ep'] for q in e)}–{max(e[q]['ep'] for q in e)} |")
    open(a.out, 'w').write('\n'.join(L) + '\n')
    print(a.out)


if __name__ == '__main__':
    main()
