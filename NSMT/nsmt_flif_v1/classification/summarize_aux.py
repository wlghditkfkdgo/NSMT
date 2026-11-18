"""MODI 2026-11-19: summary of the aux-loss test (docs/MODI_PROJECT_LOG.md, pre-registered rules; user request 11-19 "1. auxiliary
loss를 ON해서 실험하여 비교해줘").

Question: does NSMT's reconstruction loss (weak_decoder on the Neocortex output reconstructs the low-pass of each patch) improve the
base model (Neocortex-MS query population, pop LIF / pop f-LIF), and does it make the base better than Spikformer-style? Run on both
bases: spikingjelly hippocampus (the msn base) and every neuron SpikeDE (msn_sde, the SpikeDE-basis comparison of 11-18).
Aux setting: ETT = the NSMT forecasting recipe (loss (1 - r) L1 + r MSE, r = alpha * H / 720; ETTh2 alpha 0.8 keep 0.3, ETTm2 alpha
0.5 keep 0.25; selection still on the validation L1); MIT-BIH = the ECG framework default (0.9 CE + 0.1 MSE, keep 0.15).
Arms (seeds 42 / 7 / 2026; ETT best-validation weights, test MSE; MIT-BIH last epoch, test macro-F1 %):
  basis sj:  base_off nsmt4q_<n>_msn (flif-v1-ett-ms / flif-v1-mitbih-ms), base_aux nsmt4q_<n>_msn_aux (flif-v1-ett-aux / -mitbih-aux),
             Spikformer-style spk_ssa (flif-v1-gen-ett / flif-v1-gen-mitbih)
  basis sde: base_off nsmt4q_<n>_msn_sde (flif-v1-ett-sde / -mitbih-sde), base_aux nsmt4q_<n>_msn_sde_aux, Spikformer-style spkall_lif
Comparisons per basis and neuron (rules of summarize_sde.py; two neurons: both -> that verdict, one -> neuron-dependent (weak)):
  A1 base_aux - base_off            ETT 12 pairs (better = R <= -1 % and >= 8/12 lower), MIT-BIH (better = mean >= +0.5 %p, >= 2/3)
  A2 base_aux - Spikformer-style    same rules; overall per neuron: strong / weak (direction only) / not supported (summarize_sde.overall)
  A3 gap change A2 vs (base_off - Spikformer-style)   ETT g per pair (grew = mean <= -1 %p, >= 8/12 negative), MIT-BIH g per seed
Report only: training reconstruction MSE (last epoch), Neocortex / hippocampus firing, energy (aux only adds a training loss; the
decoder is not run at test time), first-epoch stops, run time; with --diag the coupling diagnostics of the aux arms (sj basis).
A missing, duplicated or non-finite value, a run folder with the wrong neuron_all suffix or keep_ratio, a MIT-BIH aux run without
aux=0.1, or new rows at several shas raise (fail-closed).
usage (classification folder): python summarize_aux.py [--diag F] [--out FILE]
"""
import argparse
import csv
import json
import os

import numpy as np

from compare_query_pop_lif_paper import CELLS, CL, COLS, FC, HERE, SEEDS, by_seed, cmp_row, compare, ett_recounted, num, results
from summarize_len import gap_change, gap_conclusion
from summarize_ms import NAMES, NEURONS, conclusion, mb_verdict
from summarize_sde import mb_gap, overall
from summarize_xnor import robust

KEEP = {'ETTh2': '0.3', 'ETTm2': '0.25'}


def last_rec(d, col):
    rows = list(csv.DictReader(open(os.path.join(d, 'log', 'best_log_0.csv'))))
    return float(rows[-1][col]) if col in rows[0] else float('nan')


def ett(cond, suite, sde, aux):
    out = ett_recounted(cond, (suite,))
    shas = set()
    for ds, pl in CELLS:
        for sd, d in by_seed(results(os.path.join(FC, suite, 'raw.txt'), ds=ds, pl=pl, cond=cond), f'{suite} {cond} {ds} {pl}').items():
            if int(d.get('seq_len', 96)) != 96 or 'lr' in d:
                raise RuntimeError(f'{suite} {cond} {ds} {pl} seed {sd}: seq_len / lr')
            if ('_nalif' in d['dir']) != sde:
                raise RuntimeError(f'{suite} {cond} {ds} {pl} seed {sd}: neuron_all suffix mismatch')
            if aux and f'+keep_ratio+{KEEP[ds]}+' not in d['dir']:
                raise RuntimeError(f'{suite} {cond} {ds} {pl} seed {sd}: keep_ratio is not {KEEP[ds]}')
            be = d['best_epoch_val_epochs'].split(':')
            out[ds, pl, sd].update(be=int(be[0]), ep=int(be[2]), time=num(d, 'time_s'), rh=num(d, 'rho_hippo'),
                                   rn=num(d, 'rho_neo') if cond.startswith('nsmt4q_') else float('nan'),
                                   rec=last_rec(d['dir'], 'train_rec') if aux else float('nan'))
            if aux and not np.isfinite(out[ds, pl, sd]['rec']):
                raise RuntimeError(f'{suite} {cond} {ds} {pl} seed {sd}: no train_rec column')
            shas.add(d['sha'])
    return out, shas


def mbt(cond, suite, sde, aux, **sel):
    out, shas = {}, set()
    for s, d in by_seed(results(os.path.join(CL, suite, 'raw.txt'), cond=cond, **sel), f'{suite} {cond} MIT-BIH').items():
        if ('_nalif' in d['dir']) != sde or int(d['epochs_run']) != 30:
            raise RuntimeError(f'{suite} {cond} seed {s}: neuron_all suffix / epochs_run')
        if cond.startswith('nsmt4q_') and float(d.get('aux', 'nan')) != (0.1 if aux else 0.0):
            raise RuntimeError(f'{suite} {cond} seed {s}: aux {d.get("aux")}')
        out[s] = dict(f1=100 * num(d, 'f1'), acc=100 * num(d, 'overall_acc'), tot=num(d, 'total_fft_uJ'), rh=num(d, 'rho_hippo'),
                      rn=num(d, 'rho_neo') if cond.startswith('nsmt4q_') else float('nan'), time=num(d, 'time_s'),
                      rec=last_rec(d['dir'], 'train_mse'))
        shas.add(d['sha'])
    return out, shas


BASES = {'sj': dict(name='spikingjelly hippocampus', off_e='flif-v1-ett-ms', off_m='flif-v1-mitbih-ms', suf='', spk=('spk_ssa', 'flif-v1-gen-ett', 'flif-v1-gen-mitbih')),
         'sde': dict(name='every neuron SpikeDE', off_e='flif-v1-ett-sde', off_m='flif-v1-mitbih-sde', suf='_sde', spk=('spkall_lif', 'flif-v1-ett-sde', 'flif-v1-mitbih-sde'))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag', default='')
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'flif-v1-aux-summary.md'))
    a = ap.parse_args()
    shas = set(); A = {}
    for b, cfg in BASES.items():
        sde = b == 'sde'
        c, se, sm = cfg['spk']
        A[b, 'spk', 'e'], _ = ett(c, se, sde, False)
        A[b, 'spk', 'm'], _ = mbt(c, sm, sde, False)
        for n in NEURONS:
            off, on = f'nsmt4q_{n}_msn{cfg["suf"]}', f'nsmt4q_{n}_msn{cfg["suf"]}_aux'
            A[b, n, 'off', 'e'], _ = ett(off, cfg['off_e'], sde, False)
            A[b, n, 'off', 'm'], _ = mbt(off, cfg['off_m'], sde, False, lr='0.01', val='0')
            A[b, n, 'on', 'e'], s1 = ett(on, 'flif-v1-ett-aux', sde, True); shas |= s1
            A[b, n, 'on', 'm'], s1 = mbt(on, 'flif-v1-mitbih-aux', sde, True, lr='0.01', val='0'); shas |= s1
    if len(shas) != 1:
        raise RuntimeError(f'new rows at several shas {shas}')
    L = ['# Aux-loss test (NSMT reconstruction loss on the msn base; spikingjelly and SpikeDE bases) — summary', '',
         f'See the module docstring and docs/MODI_PROJECT_LOG.md for the pre-registered rules. New runs at sha {shas.pop()}.', '',
         '## Results (mean ± sd of seeds)', '', '| basis | model | ' + COLS + ' | MIT-BIH macro-F1 % |', '|---|---|---|---|---|---|---|']
    def row(b, label, e, m):
        cells = ' | '.join(f"{np.mean([e[ds, pl, s]['mse'] for s in SEEDS]):.4f} ± {np.std([e[ds, pl, s]['mse'] for s in SEEDS], ddof=1):.4f}" for ds, pl in CELLS)
        return f"| {b} | {label} | {cells} | {np.mean([m[s]['f1'] for s in SEEDS]):.2f} ± {np.std([m[s]['f1'] for s in SEEDS], ddof=1):.2f} |"
    for b in BASES:
        L.append(row(b, 'Spikformer-style', A[b, 'spk', 'e'], A[b, 'spk', 'm']))
        for n in NEURONS:
            for k in ('off', 'on'):
                L.append(row(b, f"{NAMES[n]} base, aux {k}", A[b, n, k, 'e'], A[b, n, k, 'm']))
    C = {}
    head = ['| comparison | ' + COLS + ' | R % | pairs A lower | verdict | energy A/B | leave-one-cell-out |', '|---|---|---|---|---|---|---|---|---|---|']
    def mrow(name, x, ea, eb):
        e = np.mean([ea[s]['tot'] for s in SEEDS]) / np.mean([eb[s]['tot'] for s in SEEDS])
        return f"| {name} | {' / '.join(f'{v:+.2f}' for v in x[0])} | {x[1]:+.2f} | {x[2]} | {e:.2f} |"
    for b, cfg in BASES.items():
        L += ['', f"## Basis {b} ({cfg['name']}): ETT (r = A / B − 1 of test MSE, %)", ''] + head
        for n in NEURONS:
            C['A1', b, n] = x = compare(A[b, n, 'on', 'e'], A[b, n, 'off', 'e'])
            L.append(cmp_row(f'A1 {NAMES[n]}: aux on − off', x) + f" {robust(x, x['v'], x['R'])} |")
        for n in NEURONS:
            C['A2', b, n] = x = compare(A[b, n, 'on', 'e'], A[b, 'spk', 'e'])
            L.append(cmp_row(f'A2 {NAMES[n]}: base aux − Spikformer-style', x) + f" {robust(x, x['v'], x['R'])} |")
        for n in NEURONS:
            C['ref', b, n] = x = compare(A[b, n, 'off', 'e'], A[b, 'spk', 'e'])
            L.append(cmp_row(f'(reference) {NAMES[n]}: base aux off − Spikformer-style', x) + f" {robust(x, x['v'], x['R'])} |")
        L += ['', '| comparison | ' + COLS + ' | mean g | pairs negative | verdict | leave-one-cell-out |', '|---|---|---|---|---|---|---|---|---|']
        for n in NEURONS:
            g = gap_change(C['A2', b, n], C['ref', b, n]); C['A3', b, n] = g
            L.append(f"| A3 {NAMES[n]} gap change (negative = aux moves the base ahead) | " + ' | '.join(f"{np.mean(g['per'][c]):+.2f}" for c in CELLS)
                     + f" | {g['G']:+.2f} | {g['neg']}/12 | {g['v']} | {robust(g, g['v'], g['G'])} |")
        L += ['', f"### Basis {b}: MIT-BIH (d = A − B of test macro-F1, %p per seed 42 / 7 / 2026)", '', '| comparison | d per seed | mean | verdict | energy A/B |', '|---|---|---|---|---|']
        for n in NEURONS:
            C['M1', b, n] = mb_verdict(A[b, n, 'on', 'm'], A[b, n, 'off', 'm'])
            L.append(mrow(f'A1 {NAMES[n]}: aux on − off', C['M1', b, n], A[b, n, 'on', 'm'], A[b, n, 'off', 'm']))
        for n in NEURONS:
            C['M2', b, n] = mb_verdict(A[b, n, 'on', 'm'], A[b, 'spk', 'm'])
            L.append(mrow(f'A2 {NAMES[n]}: base aux − Spikformer-style', C['M2', b, n], A[b, n, 'on', 'm'], A[b, 'spk', 'm']))
        for n in NEURONS:
            L.append(mrow(f'(reference) {NAMES[n]}: base aux off − Spikformer-style', mb_verdict(A[b, n, 'off', 'm'], A[b, 'spk', 'm']),
                          A[b, n, 'off', 'm'], A[b, 'spk', 'm']))
        for n in NEURONS:
            g, mm, vv = mb_gap(A[b, n, 'on', 'm'], A[b, 'spk', 'm'], A[b, n, 'off', 'm'], A[b, 'spk', 'm']); C['M3', b, n] = vv
            L.append(f"| A3 {NAMES[n]} gap change (positive = aux moves the base ahead) | {' / '.join(f'{v:+.2f}' for v in g)} | {mm:+.2f} | {vv} | |")
    L += ['', '## conclusions (rule of the pre-registration)', '', '| basis | comparison | task | pop LIF | pop f-LIF | conclusion over the two neurons |', '|---|---|---|---|---|---|']
    for b in BASES:
        L.append(f"| {b} | A1 aux on − off | ETT | {C['A1', b, 'pop_lif']['v']} | {C['A1', b, 'pop_flif']['v']} | {conclusion([C['A1', b, n]['v'] for n in NEURONS])} |")
        L.append(f"| {b} | A1 aux on − off | MIT-BIH | {C['M1', b, 'pop_lif'][2]} | {C['M1', b, 'pop_flif'][2]} | {conclusion([C['M1', b, n][2] for n in NEURONS])} |")
        L.append(f"| {b} | A2 base aux − Spikformer-style | ETT | {C['A2', b, 'pop_lif']['v']} | {C['A2', b, 'pop_flif']['v']} | {conclusion([C['A2', b, n]['v'] for n in NEURONS])} |")
        L.append(f"| {b} | A2 base aux − Spikformer-style | MIT-BIH | {C['M2', b, 'pop_lif'][2]} | {C['M2', b, 'pop_flif'][2]} | {conclusion([C['M2', b, n][2] for n in NEURONS])} |")
        L.append(f"| {b} | A3 gap change | ETT | {C['A3', b, 'pop_lif']['v']} | {C['A3', b, 'pop_flif']['v']} | {gap_conclusion([C['A3', b, n]['v'] for n in NEURONS])} |")
        L.append(f"| {b} | A3 gap change | MIT-BIH | {C['M3', b, 'pop_lif']} | {C['M3', b, 'pop_flif']} | {gap_conclusion([C['M3', b, n] for n in NEURONS])} |")
        L.append(f"| {b} | overall: base aux better than Spikformer-style | both | {overall(C['A2', b, 'pop_lif'], C['M2', b, 'pop_lif'])} | "
                 f"{overall(C['A2', b, 'pop_flif'], C['M2', b, 'pop_flif'])} | — |")
    L += ['', '## report only', '', '| basis | model | ETT energy µJ / Spikformer-style | MIT-BIH energy µJ / Spikformer-style | ETT rho_neo | MIT-BIH rho_neo | '
          'train reconstruction MSE, last epoch (ETT / MIT-BIH) | ETT best validation at epoch 0 (of 12) | ETT epochs (min–max) | ETT run time s (min–max) |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for b in BASES:
        se, sm = A[b, 'spk', 'e'], A[b, 'spk', 'm']
        for n in NEURONS:
            for k in ('off', 'on'):
                e, m = A[b, n, k, 'e'], A[b, n, k, 'm']
                er = np.mean([np.mean([e[ds, pl, s]['tot'] for s in SEEDS]) / np.mean([se[ds, pl, s]['tot'] for s in SEEDS]) for ds, pl in CELLS])
                mr = np.mean([m[s]['tot'] for s in SEEDS]) / np.mean([sm[s]['tot'] for s in SEEDS])
                rec_e = np.mean([e[q]['rec'] for q in e]) if k == 'on' else float('nan')
                L.append(f"| {b} | {NAMES[n]} aux {k} | {np.mean([e[q]['tot'] for q in e]):.2f} / {er:.2f} | {np.mean([m[s]['tot'] for s in SEEDS]):.2f} / {mr:.2f} | "
                         f"{np.mean([e[q]['rn'] for q in e]):.4f} | {np.mean([m[s]['rn'] for s in SEEDS]):.4f} | "
                         + ('—' if np.isnan(rec_e) else f'{rec_e:.4f}') + f" / {np.mean([m[s]['rec'] for s in SEEDS]):.4f} | {sum(e[q]['be'] == 0 for q in e)} | "
                         f"{min(e[q]['ep'] for q in e)}–{max(e[q]['ep'] for q in e)} | {min(e[q]['time'] for q in e):.0f}–{max(e[q]['time'] for q in e):.0f} |")
    if a.diag:
        D = json.load(open(a.diag))
        L += ['', '### coupling diagnostics of the aux arms (spikingjelly basis; means of seeds): ΔMSE % when the Neocortex / position / content is removed at test time', '',
              '| model | cell | varying share | ΔMSE Neocortex 0 | ΔMSE position 0 | ΔMSE content 0 | y rate |', '|---|---|---|---|---|---|---|']
        for cond in ('nsmt4q_pop_lif_msn_aux', 'nsmt4q_pop_flif_msn_aux'):
            for ds, pl in CELLS:
                g = [r for r in D if r['cond'] == cond and r['ds'] == ds and r['pl'] == pl]
                if len(g) != 3 or not all(r['batch']['ok'] for r in g):
                    raise RuntimeError(f'diag {cond} {ds} {pl}: {len(g)} rows or batch check failed')
                f = lambda fn: float(np.mean([fn(r) for r in g]))
                L.append(f"| {cond} | {ds} H{pl} | {f(lambda r: r['terms']['var_share']):.3f} | {f(lambda r: r['neo0']['d_mse_pct']):+.2f} | "
                         f"{f(lambda r: r['nopos']['d_mse_pct']):+.2f} | {f(lambda r: r['nocontent']['d_mse_pct']):+.2f} | {f(lambda r: r['rate']['y']):.4f} |")
    open(a.out, 'w').write('\n'.join(L) + '\n')
    print(a.out)


if __name__ == '__main__':
    main()
