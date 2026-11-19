"""MODI 2026-11-19: summary of the full-signal aux test, phase F of of_driver.sh (docs/MODI_PROJECT_LOG.md, pre-registered rules; user
request 11-19 "저주파 auxiliary loss가 끝나면 이어서 원 시계열을 복원하는 auxiliary loss로 교체 후 실험해줘").

The low-pass aux test (summarize_aux.py) with the raw patches as the reconstruction target (--keep_ratio 0), same alpha / AUX, both bases
(sj = spikingjelly hippocampus, sde = every neuron SpikeDE), pop LIF / pop f-LIF; ETT 4 cells x 3 seeds, MIT-BIH 3 seeds.
Comparisons per basis and neuron (rules of summarize_aux.py / summarize_sde.py):
  F1 full - off        F2 full - low-pass        F3 full - Spikformer-style of the same basis, and the gap change versus (off - Spikformer-style)
  overall per neuron (strong / weak / not supported) from F3.
Report only: reconstruction MSE (last epoch), firing rates, energy, first-epoch stops; with --diag the coupling diagnostics (sj basis).
Fail-closed as summarize_aux.py; the full-signal runs must have keep_ratio 0 in the run folder (ETT and MIT-BIH) and aux=0.1 (MIT-BIH).
usage (classification folder): python summarize_auxf.py [--diag F] [--out FILE]
"""
import argparse
import json
import os

import numpy as np

import summarize_aux as SA
from compare_query_pop_lif_paper import CELLS, CL, COLS, FC, HERE, SEEDS, by_seed, cmp_row, compare, ett_recounted, num, results
from summarize_len import gap_change, gap_conclusion
from summarize_ms import NAMES, NEURONS, conclusion, mb_verdict
from summarize_sde import mb_gap, overall
from summarize_xnor import robust


def ett_full(cond, suite, sde):
    out = ett_recounted(cond, (suite,))
    shas = set()
    for ds, pl in CELLS:
        for sd, d in by_seed(results(os.path.join(FC, suite, 'raw.txt'), ds=ds, pl=pl, cond=cond), f'{suite} {cond} {ds} {pl}').items():
            if int(d.get('seq_len', 96)) != 96 or 'lr' in d or ('_nalif' in d['dir']) != sde or '+keep_ratio+0.0+' not in d['dir']:
                raise RuntimeError(f'{suite} {cond} {ds} {pl} seed {sd}: seq_len / lr / neuron_all suffix / keep_ratio 0 ({d["dir"]})')
            be = d['best_epoch_val_epochs'].split(':')
            rec = SA.last_rec(d['dir'], 'train_rec')
            if not np.isfinite(rec):
                raise RuntimeError(f'{suite} {cond} {ds} {pl} seed {sd}: no train_rec')
            out[ds, pl, sd].update(be=int(be[0]), ep=int(be[2]), time=num(d, 'time_s'), rh=num(d, 'rho_hippo'), rn=num(d, 'rho_neo'), rec=rec)
            shas.add(d['sha'])
    return out, shas


def mb_full(cond, suite, sde):
    out, shas = SA.mbt(cond, suite, sde, True, lr='0.01', val='0')
    for d in results(os.path.join(CL, suite, 'raw.txt'), cond=cond, lr='0.01', val='0'):
        if '+keep_ratio+0.0+' not in d['dir']:
            raise RuntimeError(f'{suite} {cond} seed {d["seed"]}: keep_ratio is not 0 ({d["dir"]})')
    return out, shas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag', default='')
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'flif-v1-auxf-summary.md'))
    a = ap.parse_args()
    shas, A = set(), {}
    for b, cfg in SA.BASES.items():
        sde = b == 'sde'
        c, se, sm = cfg['spk']
        A[b, 'spk', 'e'], _ = SA.ett(c, se, sde, False)
        A[b, 'spk', 'm'], _ = SA.mbt(c, sm, sde, False)
        for n in NEURONS:
            base = f'nsmt4q_{n}_msn{cfg["suf"]}'
            A[b, n, 'off', 'e'], _ = SA.ett(base, cfg['off_e'], sde, False)
            A[b, n, 'off', 'm'], _ = SA.mbt(base, cfg['off_m'], sde, False, lr='0.01', val='0')
            A[b, n, 'low', 'e'], _ = SA.ett(base + '_aux', 'flif-v1-ett-aux', sde, True)
            A[b, n, 'low', 'm'], _ = SA.mbt(base + '_aux', 'flif-v1-mitbih-aux', sde, True, lr='0.01', val='0')
            A[b, n, 'full', 'e'], s1 = ett_full(base + '_auxf', 'flif-v1-ett-auxf', sde); shas |= s1
            A[b, n, 'full', 'm'], s1 = mb_full(base + '_auxf', 'flif-v1-mitbih-auxf', sde); shas |= s1
    if len(shas) != 1:
        raise RuntimeError(f'new rows at several shas {shas}')
    L = ['# Full-signal aux test (raw patches as the reconstruction target; spikingjelly and SpikeDE bases) — summary', '',
         f'See the module docstring and docs/MODI_PROJECT_LOG.md for the pre-registered rules. New runs at sha {shas.pop()}.', '',
         '## Results (mean ± sd of seeds)', '', '| basis | model | ' + COLS + ' | MIT-BIH macro-F1 % |', '|---|---|---|---|---|---|---|']
    def row(b, label, e, m):
        cells = ' | '.join(f"{np.mean([e[ds, pl, s]['mse'] for s in SEEDS]):.4f} ± {np.std([e[ds, pl, s]['mse'] for s in SEEDS], ddof=1):.4f}" for ds, pl in CELLS)
        return f"| {b} | {label} | {cells} | {np.mean([m[s]['f1'] for s in SEEDS]):.2f} ± {np.std([m[s]['f1'] for s in SEEDS], ddof=1):.2f} |"
    for b in SA.BASES:
        L.append(row(b, 'Spikformer-style', A[b, 'spk', 'e'], A[b, 'spk', 'm']))
        for n in NEURONS:
            for k, lab in (('off', 'aux off'), ('low', 'low-pass aux'), ('full', 'full-signal aux')):
                L.append(row(b, f'{NAMES[n]} base, {lab}', A[b, n, k, 'e'], A[b, n, k, 'm']))
    C = {}
    head = ['| comparison | ' + COLS + ' | R % | pairs A lower | verdict | energy A/B | leave-one-cell-out |', '|---|---|---|---|---|---|---|---|---|---|']
    def mrow(name, x, ea, eb):
        e = np.mean([ea[s]['tot'] for s in SEEDS]) / np.mean([eb[s]['tot'] for s in SEEDS])
        return f"| {name} | {' / '.join(f'{v:+.2f}' for v in x[0])} | {x[1]:+.2f} | {x[2]} | {e:.2f} |"
    for b, cfg in SA.BASES.items():
        L += ['', f"## Basis {b} ({cfg['name']}): ETT (r = A / B − 1 of test MSE, %)", ''] + head
        for key, a_, b_, name in (('F1', 'full', 'off', 'full − off'), ('F2', 'full', 'low', 'full − low-pass')):
            for n in NEURONS:
                C[key, b, n] = x = compare(A[b, n, a_, 'e'], A[b, n, b_, 'e'])
                L.append(cmp_row(f'{key} {NAMES[n]}: {name}', x) + f" {robust(x, x['v'], x['R'])} |")
        for n in NEURONS:
            C['F3', b, n] = x = compare(A[b, n, 'full', 'e'], A[b, 'spk', 'e'])
            L.append(cmp_row(f'F3 {NAMES[n]}: base full − Spikformer-style', x) + f" {robust(x, x['v'], x['R'])} |")
            C['ref', b, n] = compare(A[b, n, 'off', 'e'], A[b, 'spk', 'e'])
        L += ['', '| comparison | ' + COLS + ' | mean g | pairs negative | verdict | leave-one-cell-out |', '|---|---|---|---|---|---|---|---|---|']
        for n in NEURONS:
            g = gap_change(C['F3', b, n], C['ref', b, n]); C['G', b, n] = g
            L.append(f"| F3 gap change {NAMES[n]} (negative = the full aux moves the base ahead) | " + ' | '.join(f"{np.mean(g['per'][c]):+.2f}" for c in CELLS)
                     + f" | {g['G']:+.2f} | {g['neg']}/12 | {g['v']} | {robust(g, g['v'], g['G'])} |")
        L += ['', f'### Basis {b}: MIT-BIH (d = A − B of test macro-F1, %p per seed 42 / 7 / 2026)', '', '| comparison | d per seed | mean | verdict | energy A/B |', '|---|---|---|---|---|']
        for key, a_, b_, name in (('M1', 'full', 'off', 'full − off'), ('M2', 'full', 'low', 'full − low-pass')):
            for n in NEURONS:
                C[key, b, n] = mb_verdict(A[b, n, a_, 'm'], A[b, n, b_, 'm'])
                L.append(mrow(f'{key[0]}{key[1]} {NAMES[n]}: {name}', C[key, b, n], A[b, n, a_, 'm'], A[b, n, b_, 'm']))
        for n in NEURONS:
            C['M3', b, n] = mb_verdict(A[b, n, 'full', 'm'], A[b, 'spk', 'm'])
            L.append(mrow(f'M3 {NAMES[n]}: base full − Spikformer-style', C['M3', b, n], A[b, n, 'full', 'm'], A[b, 'spk', 'm']))
            g, mm, vv = mb_gap(A[b, n, 'full', 'm'], A[b, 'spk', 'm'], A[b, n, 'off', 'm'], A[b, 'spk', 'm']); C['MG', b, n] = vv
            L.append(f"| M3 gap change {NAMES[n]} (positive = the full aux moves the base ahead) | {' / '.join(f'{v:+.2f}' for v in g)} | {mm:+.2f} | {vv} | |")
    L += ['', '## conclusions (rule of the pre-registration)', '', '| basis | comparison | task | pop LIF | pop f-LIF | conclusion over the two neurons |', '|---|---|---|---|---|---|']
    for b in SA.BASES:
        for key, mk, name in (('F1', 'M1', 'full − off'), ('F2', 'M2', 'full − low-pass'), ('F3', 'M3', 'base full − Spikformer-style')):
            L.append(f"| {b} | {name} | ETT | {C[key, b, 'pop_lif']['v']} | {C[key, b, 'pop_flif']['v']} | {conclusion([C[key, b, n]['v'] for n in NEURONS])} |")
            L.append(f"| {b} | {name} | MIT-BIH | {C[mk, b, 'pop_lif'][2]} | {C[mk, b, 'pop_flif'][2]} | {conclusion([C[mk, b, n][2] for n in NEURONS])} |")
        L.append(f"| {b} | gap change | ETT | {C['G', b, 'pop_lif']['v']} | {C['G', b, 'pop_flif']['v']} | {gap_conclusion([C['G', b, n]['v'] for n in NEURONS])} |")
        L.append(f"| {b} | gap change | MIT-BIH | {C['MG', b, 'pop_lif']} | {C['MG', b, 'pop_flif']} | {gap_conclusion([C['MG', b, n] for n in NEURONS])} |")
        L.append(f"| {b} | overall: base full aux better than Spikformer-style | both | {overall(C['F3', b, 'pop_lif'], C['M3', b, 'pop_lif'])} | "
                 f"{overall(C['F3', b, 'pop_flif'], C['M3', b, 'pop_flif'])} | — |")
    L += ['', '## report only', '', '| basis | model | ETT energy µJ / Spikformer-style | MIT-BIH energy µJ / Spikformer-style | ETT rho_neo | MIT-BIH rho_neo | '
          'train reconstruction MSE, last epoch (ETT / MIT-BIH) | ETT best validation at epoch 0 (of 12) | ETT epochs (min–max) |', '|---|---|---|---|---|---|---|---|---|']
    for b in SA.BASES:
        se, sm = A[b, 'spk', 'e'], A[b, 'spk', 'm']
        for n in NEURONS:
            for k in ('off', 'low', 'full'):
                e, m = A[b, n, k, 'e'], A[b, n, k, 'm']
                er = np.mean([np.mean([e[ds, pl, s]['tot'] for s in SEEDS]) / np.mean([se[ds, pl, s]['tot'] for s in SEEDS]) for ds, pl in CELLS])
                mr = np.mean([m[s]['tot'] for s in SEEDS]) / np.mean([sm[s]['tot'] for s in SEEDS])
                rec_e = np.mean([e[q]['rec'] for q in e]) if k != 'off' else float('nan')
                L.append(f"| {b} | {NAMES[n]} {k} | {np.mean([e[q]['tot'] for q in e]):.2f} / {er:.2f} | {np.mean([m[s]['tot'] for s in SEEDS]):.2f} / {mr:.2f} | "
                         f"{np.mean([e[q]['rn'] for q in e]):.4f} | {np.mean([m[s]['rn'] for s in SEEDS]):.4f} | " + ('—' if np.isnan(rec_e) else f'{rec_e:.4f}')
                         + f" / {np.mean([m[s]['rec'] for s in SEEDS]):.4f} | {sum(e[q]['be'] == 0 for q in e)} | {min(e[q]['ep'] for q in e)}–{max(e[q]['ep'] for q in e)} |")
    if a.diag:
        D = json.load(open(a.diag))
        L += ['', '### coupling diagnostics of the full-signal aux arms (spikingjelly basis; means of seeds)', '',
              '| model | cell | varying share | ΔMSE Neocortex 0 | ΔMSE position 0 | ΔMSE content 0 | y rate |', '|---|---|---|---|---|---|---|']
        for cond in ('nsmt4q_pop_lif_msn_auxf', 'nsmt4q_pop_flif_msn_auxf'):
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
