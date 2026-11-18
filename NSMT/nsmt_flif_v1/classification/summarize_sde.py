"""MODI 2026-11-18: summary of the SpikeDE-basis comparison (docs/MODI_PROJECT_LOG.md, pre-registered rules; user decision 11-18
"(가)": Spikformer-style and the base model both built from SpikeDE neurons only).

Question: with no spikingjelly LIF left in either model (--neuron_all lif: embedding, attention and MLP neurons -> SpikeDE LIF; the
base keeps its SpikeDE population Neocortex), is the base model (Neocortex-MS query population, pop LIF / pop f-LIF) better than
Spikformer-style, and does the gap differ from the current comparison (spikingjelly LIF in the shared parts of both models)?
Arms (seeds 42 / 7 / 2026):
  ETT (ETTh2 / ETTm2 x H96 / H720, input 96, best-validation weights, test MSE; energy = head recount):
    spk_now  spk_ssa              flif-v1-gen-ett     spk_sde  spkall_lif                 flif-v1-ett-sde
    base_now nsmt4q_<n>_msn       flif-v1-ett-ms      base_sde nsmt4q_<n>_msn_sde         flif-v1-ett-sde
  MIT-BIH (30 epochs, last-epoch weights, test macro-F1 %; energy = total_fft_uJ):
    spk_now  spk_ssa              flif-v1-gen-mitbih  spk_sde  spkall_lif                 flif-v1-mitbih-sde
    base_now nsmt4q_<n>_msn       flif-v1-mitbih-ms   base_sde nsmt4q_<n>_msn_sde         flif-v1-mitbih-sde (lr 0.01, val 0)
Comparisons (per neuron; two neurons: both -> that verdict, one -> neuron-dependent (weak)):
  E1 ETT base_sde - spk_sde (main): r = A / B - 1 of test MSE over 12 pairs; better = mean <= -1 % and >= 8/12 lower, worse = mirror
  E2 ETT gap change g = r_E1 - r_now per pair, r_now = base_now / spk_now - 1; grew = mean <= -1 %p and >= 8/12 negative, shrank = mirror
  M1 MIT-BIH base_sde - spk_sde (main): d = macro-F1 difference per seed; better = mean >= +0.5 %p and >= 2/3 positive, worse = mirror
  M2 MIT-BIH gap change g = d_M1 - d_now per seed; grew = mean >= +0.5 %p and >= 2/3 positive, shrank = mirror
  overall per neuron: strong = E1 or M1 better and neither worse; weak (direction only) = no better, no worse, ETT R < 0 and MIT-BIH
  mean d > 0; otherwise not supported.
  leave-one-cell-out (ETT) as in summarize_xnor.py.
Report only: library effect (spk_sde - spk_now, base_sde - base_now), energy ratios (base_sde / spk_sde, and against spk_now: the
user prefers the base at or below Spikformer-style), MAE, accuracy, firing rates, first-epoch stops, the retrain checks (MIT-BIH
spk_ssa seed 42 in flif-v1-mitbih-sde-repro; ETTh2 H96 spkall_lif against its 10-29 runs in flif-v1-ett-nall).
A missing, duplicated or non-finite value, a run folder without (new) / with (reference) the neuron_all suffix _nalif, an ETT row with
seq_len other than 96 or an lr field, or new rows at several shas raise (fail-closed).
usage (classification folder): python summarize_sde.py [--out FILE]
"""
import argparse
import os

import numpy as np

from compare_query_pop_lif_paper import CELLS, CL, COLS, FC, HERE, SEEDS, by_seed, cmp_row, compare, ett_recounted, num, results
from summarize_len import gap_change, gap_conclusion
from summarize_ms import NAMES, NEURONS, conclusion, mb_verdict
from summarize_xnor import robust


def check_dir(d, new, what):
    if ('_nalif' in d['dir']) != new:
        raise RuntimeError(f'{what}: run folder {"lacks" if new else "has"} the neuron_all suffix: {d["dir"]}')


def ett(cond, suite, new):
    out = ett_recounted(cond, (suite,))
    shas = set()
    for ds, pl in CELLS:
        for sd, d in by_seed(results(os.path.join(FC, suite, 'raw.txt'), ds=ds, pl=pl, cond=cond), f'{suite} {cond} {ds} {pl}').items():
            if int(d.get('seq_len', 96)) != 96 or 'lr' in d:
                raise RuntimeError(f'{suite} {cond} {ds} {pl} seed {sd}: seq_len {d.get("seq_len")} lr {d.get("lr")}')
            check_dir(d, new, f'{suite} {cond} {ds} {pl} seed {sd}')
            be = d['best_epoch_val_epochs'].split(':')
            out[ds, pl, sd].update(be=int(be[0]), ep=int(be[2]), rh=num(d, 'rho_hippo'), time=num(d, 'time_s'),
                                   rn=num(d, 'rho_neo') if cond.startswith('nsmt4q_') else float('nan'))
            shas.add(d['sha'])
    return out, shas


def mbt(cond, suite, new, **sel):
    out, shas = {}, set()
    for s, d in by_seed(results(os.path.join(CL, suite, 'raw.txt'), cond=cond, **sel), f'{suite} {cond} MIT-BIH').items():
        check_dir(d, new, f'{suite} {cond} seed {s}')
        if int(d['epochs_run']) != 30:
            raise RuntimeError(f'{suite} {cond} seed {s}: epochs_run {d["epochs_run"]}')
        out[s] = dict(f1=100 * num(d, 'f1'), acc=100 * num(d, 'overall_acc'), tot=num(d, 'total_fft_uJ'), rh=num(d, 'rho_hippo'),
                      rn=num(d, 'rho_neo') if cond.startswith('nsmt4q_') else float('nan'), par=int(d['params']), time=num(d, 'time_s'))
        shas.add(d['sha'])
    return out, shas


def mb_gap(a, b, a0, b0):
    g = [(a[s]['f1'] - b[s]['f1']) - (a0[s]['f1'] - b0[s]['f1']) for s in SEEDS]; m = float(np.mean(g))
    v = 'grew' if m >= 0.5 and sum(x > 0 for x in g) >= 2 else 'shrank' if m <= -0.5 and sum(x < 0 for x in g) >= 2 else 'no clear change'
    return g, m, v


def overall(e, m):
    """per neuron: e = E1 compare dict, m = (d, mean, verdict) of M1."""
    vs = (e['v'], m[2])
    if 'worse' in vs:
        return 'not supported (a task is worse)'
    if 'better' in vs:
        return 'strong'
    if e['R'] < 0 and m[1] > 0:
        return 'weak (direction only)'
    return 'not supported'


def repro():
    lines, shas = [], set()
    new = results(os.path.join(CL, 'flif-v1-mitbih-sde-repro', 'raw.txt'), cond='spk_ssa')
    old = results(os.path.join(CL, 'flif-v1-gen-mitbih', 'raw.txt'), cond='spk_ssa', seed=42)
    if [r['seed'] for r in new] != ['42'] or len(old) != 1:
        raise RuntimeError(f'MIT-BIH repro rows {len(new)} / {len(old)}')
    a, b = new[0], old[0]
    keys = ('f1', 'overall_acc', 'total_fft_uJ')
    lines.append(f"| MIT-BIH spk_ssa seed 42 | {' / '.join(b[k] for k in keys)} ({b['sha']}) | {' / '.join(a[k] for k in keys)} ({a['sha']}) | "
                 f"{'yes' if all(a[k] == b[k] for k in keys) else 'no'} |")
    shas.add(a['sha'])
    keys = ('mse', 'mae', 'val_loss_best', 'total_fft_uJ', 'last_mse')
    for s in SEEDS:
        n = results(os.path.join(FC, 'flif-v1-ett-sde', 'raw.txt'), cond='spkall_lif', ds='ETTh2', pl=96, seed=s)
        o = results(os.path.join(FC, 'flif-v1-ett-nall', 'raw.txt'), cond='spkall_lif', ds='ETTh2', pl=96, seed=s)
        if len(n) != 1 or len(o) != 1:
            raise RuntimeError(f'ETT repro seed {s}: rows {len(n)} / {len(o)}')
        a, b = n[0], o[0]
        lines.append(f"| ETTh2 H96 spkall_lif seed {s} | {' / '.join(b[k] for k in keys[:3])} ({b['sha']}) | {' / '.join(a[k] for k in keys[:3])} ({a['sha']}) | "
                     f"{'yes' if all(a[k] == b[k] for k in keys) else 'no'} |")
    return lines, shas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'flif-v1-sde-summary.md'))
    a = ap.parse_args()
    shas = set()
    SPn, _ = ett('spk_ssa', 'flif-v1-gen-ett', False)
    SPs, s1 = ett('spkall_lif', 'flif-v1-ett-sde', True); shas |= s1
    Bn = {n: ett(f'nsmt4q_{n}_msn', 'flif-v1-ett-ms', False)[0] for n in NEURONS}
    Bs = {}
    for n in NEURONS:
        Bs[n], s1 = ett(f'nsmt4q_{n}_msn_sde', 'flif-v1-ett-sde', True); shas |= s1
    mSPn, _ = mbt('spk_ssa', 'flif-v1-gen-mitbih', False)
    mSPs, s1 = mbt('spkall_lif', 'flif-v1-mitbih-sde', True); shas |= s1
    mBn = {n: mbt(f'nsmt4q_{n}_msn', 'flif-v1-mitbih-ms', False, lr='0.01', val='0')[0] for n in NEURONS}
    mBs = {}
    for n in NEURONS:
        mBs[n], s1 = mbt(f'nsmt4q_{n}_msn_sde', 'flif-v1-mitbih-sde', True, lr='0.01', val='0'); shas |= s1
    rep, s1 = repro(); shas |= s1
    if len(shas) != 1:
        raise RuntimeError(f'new rows at several shas {shas}')
    L = ['# SpikeDE-basis comparison (Spikformer-style and the msn base, every neuron from SpikeDE) — summary', '',
         f'See the module docstring and docs/MODI_PROJECT_LOG.md for the pre-registered rules. New runs at sha {shas.pop()}.', '',
         '## Retrain checks (report only)', '', '| run | logged (sha) | retrained (sha) | identical (all logged metrics) |', '|---|---|---|---|'] + rep
    models = [('Spikformer-style, spikingjelly LIF (now)', SPn, mSPn), ('Spikformer-style, SpikeDE LIF', SPs, mSPs)]
    for n in NEURONS:
        models += [(f'{NAMES[n]} base, spikingjelly hippocampus (now)', Bn[n], mBn[n]), (f'{NAMES[n]} base, SpikeDE everywhere', Bs[n], mBs[n])]
    L += ['', '## Results (mean ± sd of seeds)', '', '| model | ' + COLS + ' | MIT-BIH macro-F1 % | MIT-BIH accuracy % |', '|---|---|---|---|---|---|---|']
    for name, t, m in models:
        cells = ' | '.join(f"{np.mean([t[ds, pl, s]['mse'] for s in SEEDS]):.4f} ± {np.std([t[ds, pl, s]['mse'] for s in SEEDS], ddof=1):.4f}" for ds, pl in CELLS)
        L.append(f"| {name} | {cells} | {np.mean([m[s]['f1'] for s in SEEDS]):.2f} ± {np.std([m[s]['f1'] for s in SEEDS], ddof=1):.2f} | "
                 f"{np.mean([m[s]['acc'] for s in SEEDS]):.2f} |")
    C = {}
    head = ['| comparison | ' + COLS + ' | R % | pairs A lower | verdict | energy A/B | leave-one-cell-out |', '|---|---|---|---|---|---|---|---|---|---|']
    L += ['', '## ETT (r = A / B − 1 of test MSE, %; energy = head recount)', ''] + head
    for n in NEURONS:
        C['E1', n] = x = compare(Bs[n], SPs)
        L.append(cmp_row(f'E1 {NAMES[n]}: base SpikeDE − Spikformer-style SpikeDE', x) + f" {robust(x, x['v'], x['R'])} |")
    for n in NEURONS:
        C['now', n] = x = compare(Bn[n], SPn)
        L.append(cmp_row(f'(reference) {NAMES[n]}: base − Spikformer-style, now', x) + f" {robust(x, x['v'], x['R'])} |")
    x = compare(SPs, SPn); L.append(cmp_row('(library) Spikformer-style SpikeDE − spikingjelly', x) + f" {robust(x, x['v'], x['R'])} |")
    for n in NEURONS:
        x = compare(Bs[n], Bn[n]); L.append(cmp_row(f'(library) {NAMES[n]} base SpikeDE − now', x) + f" {robust(x, x['v'], x['R'])} |")
    L += ['', '### E2 — change of the ETT gap (g = r_E1 − r_now per cell and seed, %p; negative = the base gains on the SpikeDE basis)', '',
          '| comparison | ' + COLS + ' | mean g | pairs negative | verdict | leave-one-cell-out |', '|---|---|---|---|---|---|---|---|---|']
    for n in NEURONS:
        g = gap_change(C['E1', n], C['now', n]); C['E2', n] = g
        L.append(f"| E2 {NAMES[n]} | " + ' | '.join(f"{np.mean(g['per'][c]):+.2f}" for c in CELLS) + f" | {g['G']:+.2f} | {g['neg']}/12 | {g['v']} | "
                 f"{robust(g, g['v'], g['G'])} |")
    L += ['', '## MIT-BIH (d = A − B of test macro-F1, %p per seed 42 / 7 / 2026; energy = total_fft_uJ)', '',
          '| comparison | d per seed | mean | verdict | energy A/B |', '|---|---|---|---|---|']
    def mrow(name, A, B, v=None):
        d, mm, vv = mb_verdict(A, B) if v is None else v
        e = np.mean([A[s]['tot'] for s in SEEDS]) / np.mean([B[s]['tot'] for s in SEEDS])
        return f"| {name} | {' / '.join(f'{x:+.2f}' for x in d)} | {mm:+.2f} | {vv} | {e:.2f} |"
    for n in NEURONS:
        C['M1', n] = mb_verdict(mBs[n], mSPs)
        L.append(mrow(f'M1 {NAMES[n]}: base SpikeDE − Spikformer-style SpikeDE', mBs[n], mSPs))
    for n in NEURONS:
        L.append(mrow(f'(reference) {NAMES[n]}: base − Spikformer-style, now', mBn[n], mSPn))
    L.append(mrow('(library) Spikformer-style SpikeDE − spikingjelly', mSPs, mSPn))
    for n in NEURONS:
        L.append(mrow(f'(library) {NAMES[n]} base SpikeDE − now', mBs[n], mBn[n]))
    L += ['', '| comparison | g per seed | mean g | verdict |', '|---|---|---|---|']
    for n in NEURONS:
        g, mm, vv = mb_gap(mBs[n], mSPs, mBn[n], mSPn); C['M2', n] = vv
        L.append(f"| M2 {NAMES[n]} (positive = the base gains on the SpikeDE basis) | {' / '.join(f'{x:+.2f}' for x in g)} | {mm:+.2f} | {vv} |")
    L += ['', '## conclusions (rule of the pre-registration)', '', '| comparison | pop LIF | pop f-LIF | conclusion over the two neurons |', '|---|---|---|---|']
    L.append(f"| E1 ETT base − Spikformer-style (SpikeDE) | {C['E1', 'pop_lif']['v']} | {C['E1', 'pop_flif']['v']} | {conclusion([C['E1', n]['v'] for n in NEURONS])} |")
    L.append(f"| E2 ETT gap change | {C['E2', 'pop_lif']['v']} | {C['E2', 'pop_flif']['v']} | {gap_conclusion([C['E2', n]['v'] for n in NEURONS])} |")
    L.append(f"| M1 MIT-BIH base − Spikformer-style (SpikeDE) | {C['M1', 'pop_lif'][2]} | {C['M1', 'pop_flif'][2]} | {conclusion([C['M1', n][2] for n in NEURONS])} |")
    L.append(f"| M2 MIT-BIH gap change | {C['M2', 'pop_lif']} | {C['M2', 'pop_flif']} | {gap_conclusion([C['M2', n] for n in NEURONS])} |")
    L.append(f"| overall (base better than Spikformer-style on the SpikeDE basis) | {overall(C['E1', 'pop_lif'], C['M1', 'pop_lif'])} | "
             f"{overall(C['E1', 'pop_flif'], C['M1', 'pop_flif'])} | — |")
    L += ['', '## report only', '', '### energy (µJ, mean of cells / seeds) and ratios to Spikformer-style', '',
          '| model | ETT energy | / Spikformer-style SpikeDE | / Spikformer-style now | MIT-BIH energy | / Spikformer-style SpikeDE | / Spikformer-style now |',
          '|---|---|---|---|---|---|---|']
    def er(t, ref):
        return float(np.mean([np.mean([t[ds, pl, s]['tot'] for s in SEEDS]) / np.mean([ref[ds, pl, s]['tot'] for s in SEEDS]) for ds, pl in CELLS]))
    for name, t, m in models:
        me = np.mean([m[s]['tot'] for s in SEEDS])
        L.append(f"| {name} | {np.mean([t[k]['tot'] for k in t]):.2f} | {er(t, SPs):.2f} | {er(t, SPn):.2f} | {me:.2f} | "
                 f"{me / np.mean([mSPs[s]['tot'] for s in SEEDS]):.2f} | {me / np.mean([mSPn[s]['tot'] for s in SEEDS]):.2f} |")
    L += ['', '### firing rates, first-epoch stops (ETT), parameters, run time', '',
          '| model | ETT rho_hippo | ETT rho_neo | MIT-BIH rho_hippo | MIT-BIH rho_neo | ETT best validation at epoch 0 (of 12) | ETT epochs (min–max) | params ETTh2 H96 / MIT-BIH | ETT run time s (min–max) |',
          '|---|---|---|---|---|---|---|---|---|']
    for name, t, m in models:
        rn = [t[k]['rn'] for k in t]; mrn = [m[s]['rn'] for s in SEEDS]
        L.append(f"| {name} | {np.mean([t[k]['rh'] for k in t]):.4f} | " + ('—' if np.isnan(rn[0]) else f'{np.mean(rn):.4f}')
                 + f" | {np.mean([m[s]['rh'] for s in SEEDS]):.4f} | " + ('—' if np.isnan(mrn[0]) else f'{np.mean(mrn):.4f}')
                 + f" | {sum(t[k]['be'] == 0 for k in t)} | {min(t[k]['ep'] for k in t)}–{max(t[k]['ep'] for k in t)} | {t['ETTh2', 96, 42]['par']} / {m[42]['par']} | "
                 f"{min(t[k]['time'] for k in t):.0f}–{max(t[k]['time'] for k in t):.0f} |")
    L += ['', '### ETT test MAE (mean of seeds)', '', '| model | ' + COLS + ' |', '|---|---|---|---|---|']
    for name, t, _ in models:
        L.append(f'| {name} | ' + ' | '.join(f"{np.mean([t[ds, pl, s]['mae'] for s in SEEDS]):.4f}" for ds, pl in CELLS) + ' |')
    open(a.out, 'w').write('\n'.join(L) + '\n')
    print(a.out)


if __name__ == '__main__':
    main()
