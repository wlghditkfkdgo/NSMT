"""Neuron implementation / placement test (2026-11-18, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; user request 11-18).
SF-de  = the Spikformer-style model with the SpkMixer's spikeDE LIF in every layer (--neuron_all lif, tag _nalif), tuned like the spikingjelly one
         (SF-sj) on 11-01: ETT lr {1e-4, 2.5e-4, 5e-4*, 1e-3, 2e-3} (layers ETTh2 1, ETTm2 2), MIT-BIH lr {0.0025, 0.005, 0.01*, 0.02} on fold 0.
SpkMixer arms (tuned LIF SpkMixer settings, not re-tuned; muxlin; ETT IAND, 1 mixer layer, lr 5e-4 ETTh2 / 1e-4 ETTm2; MIT-BIH ADD, 2 layers, lr 0.02):
    LIF, LIF+mean (11-18 mx3), pop LIF all (11-17 STFT test rep rows), and the new f-LIF, f-LIF+mean, pop LIF all+mean, pop LIF mixer(+mean),
    pop f-LIF all(+mean), pop f-LIF mixer(+mean); "mixer" = population only in the directional stacks + fusion (--spk_neuron_scope mixer).
Stages:
    select   SF-de lr from validation only (ETT: per dataset, score(v) = mean over H96 / H720 of [seed-mean best validation L1 at v] /
             [seed-mean at 5e-4], lowest wins, ties < 1e-12 keep 5e-4; MIT-BIH: highest seed-mean last-epoch validation macro-F1, ties < 1e-9
             keep 0.01) -> results/flif-v1-{ett,mitbih}-sfde/selected_lr.json (never overwritten with a different selection)
    sf_ett / sf_mit   part 1: A1 LIF SpkMixer vs SF-de (verdict); report SF-de vs SF-sj, LIF SpkMixer vs SF-sj
    ett / mitbih      parts 2 and 3: B1 pop LIF mixer vs LIF, B2 pop LIF all vs LIF, B3 pop LIF mixer vs pop LIF all, B4 pop f-LIF mixer vs f-LIF,
                      B5 pop f-LIF all vs f-LIF, B6 pop f-LIF mixer vs pop f-LIF all (no pooling); C1-C5 mean vs none for f-LIF, pop LIF all,
                      pop LIF mixer, pop f-LIF all, pop f-LIF mixer (verdicts); report: f-LIF vs LIF, LIF+mean vs LIF, B1-B6 with mean pooling,
                      every arm vs SF-de and SF-sj
    len336            part 4 (ETT input 336): D1 LIF vs SF-de, D2 pop LIF mixer vs LIF, D3 pop LIF mixer vs pop LIF all (verdicts); report:
                      pop LIF mixer vs SF-de, SF-de vs SF-sj
Rules (each task's existing one): ETT test MSE, r = 100 (A - B) / B per seed, R = mean over the 4 cells of the seed-mean r, better = R <= -1 % and
>= 8 of 12 pairs lower, worse = mirror; MIT-BIH test macro-F1 (last-epoch weights), per-seed difference in %p, better = mean >= +0.5 and >= 2 of 3
seeds higher, worse = mirror. Energy: total_fft_uJ_corr of energy_recount (per window / beat).
Fail closed (exit 2, nothing written): a missing run, a SKIP line of a used condition, a duplicate, a non-finite value, a recount that does not reproduce
the logged metric, a non-binary ETT head input, or (ETT) a new run whose best epoch is not the argmin of its validation loss.
usage: python -B summarize_pn.py --stage select|sf_ett|sf_mit|ett|mitbih|len336 [--out <file>]
"""
import argparse
import json
import os
import sys

from summarize_l2 import CELLS, SEEDS, RES, TUNE, compare, mean, num, read_result, spikformer, stop
from summarize_ms import LR_MX, LR_SF, LY_SF, MITB, best_ok, cmp_mit, load_ett, load_mit

SFE, SFM = os.path.join(RES, 'flif-v1-ett-sfde'), os.path.join(RES, 'flif-v1-mitbih-sfde')
PNE, PNM, P336 = os.path.join(RES, 'flif-v1-ett-spkmux-pn'), os.path.join(RES, 'flif-v1-mitbih-spkmux-pn'), os.path.join(RES, 'flif-v1-ett-spkmux-pn336')
MX3E, MX3M = os.path.join(RES, 'flif-v1-ett-spkmux-mx3'), os.path.join(RES, 'flif-v1-mitbih-spkmux-mx3')
STE, STM = os.path.join(RES, 'flif-v1-ett-spkmux-stft'), os.path.join(RES, 'flif-v1-mitbih-spkmux-stft')
L336 = os.path.join(RES, 'flif-v1-ett-spkmux-len336')
if os.environ.get('PN_DRY'):                                              # dry runs: the new suites from <PN_DRY>/<suite>
    _d = os.environ['PN_DRY']
    SFE, SFM, PNE, PNM, P336 = (os.path.join(_d, os.path.basename(x)) for x in (SFE, SFM, PNE, PNM, P336))
ETT_LR = ('0.0001', '0.00025', '0.0005', '0.001', '0.002')
MIT_LR = ('0.0025', '0.005', '0.01', '0.02')
# arm -> (suite folder, cond, tag suffix after _lr<LR>_L<layers>); ETT LR = LR_MX, layers 1; MIT-BIH lr 0.02, layers 2
ARMS = (('LIF', None, 'lif', ''), ('LIF+mean', None, 'lif', '_mpmean'), ('f-LIF', None, 'flif', ''), ('f-LIF+mean', None, 'flif', '_mpmean'),
        ('pop LIF all', None, 'pop_lif', ''), ('pop LIF all+mean', None, 'pop_lif', '_mpmean'), ('pop LIF mixer', None, 'pop_lif', '_nsmixer'),
        ('pop LIF mixer+mean', None, 'pop_lif', '_mpmean_nsmixer'), ('pop f-LIF all', None, 'pop_flif', ''), ('pop f-LIF all+mean', None, 'pop_flif', '_mpmean'),
        ('pop f-LIF mixer', None, 'pop_flif', '_nsmixer'), ('pop f-LIF mixer+mean', None, 'pop_flif', '_mpmean_nsmixer'))


def where(name, task):
    """suite folder of an arm: the earlier runs for LIF, LIF+mean, pop LIF all; the new suite otherwise."""
    if name == 'LIF':
        return TUNE if task == 'ett' else MITB
    if name == 'LIF+mean':
        return MX3E if task == 'ett' else MX3M
    if name == 'pop LIF all':
        return STE if task == 'ett' else STM
    return PNE if task == 'ett' else PNM


def selected(task):
    p = os.path.join(SFE if task == 'ett' else SFM, 'selected_lr.json')
    if not os.path.exists(p):
        stop(f'no selection {p} (run --stage select first)')
    return json.load(open(p))['selected']


def sf_tag_ett(ds, lr, L=''):
    return f'spk_ssa_lr{lr}_L{LY_SF[ds]}{L}_nalif'


def load_sfde_ett(L=''):
    sel = selected('ett')
    S = P336 if L else SFE
    T = load_ett(os.path.join(S, 'raw.txt'), os.path.join(S, 'energy_recount.csv'), lambda ds: sf_tag_ett(ds, sel[ds], L), 'spk_ssa')
    for k, r in T.items():
        best_ok(k, r)
    return T


def load_sfde_mit():
    lr = selected('mitbih')['MITBIH']
    return load_mit(os.path.join(SFM, 'raw.txt'), os.path.join(SFM, 'energy_recount.csv'), f'spk_ssa_lr{lr}_L2_nalif')


def arms_ett():
    T = {}
    for name, _, mixer, suf in ARMS:
        S = where(name, 'ett')
        cond = f'spkmx_{mixer}_muxlin_iand'
        T[name] = load_ett(os.path.join(S, 'raw.txt'), os.path.join(S, 'energy_recount.csv'), lambda ds, m=mixer, s=suf: f'spkmx_{m}_muxlin_iand_lr{LR_MX[ds]}_L1{s}', cond)
        if S == PNE:
            for k, r in T[name].items():
                best_ok(k, r)
    return T


def arms_mit():
    T = {}
    for name, _, mixer, suf in ARMS:
        S = where(name, 'mitbih')
        T[name] = load_mit(os.path.join(S, 'raw.txt'), os.path.join(S, 'energy_recount.csv'), f'spkmx_{mixer}_muxlin_add_lr0.02_L2{suf}')
    return T


def questions(T):
    B = [('B1 pop LIF mixer vs LIF', T['pop LIF mixer'], T['LIF']), ('B2 pop LIF all vs LIF', T['pop LIF all'], T['LIF']),
         ('B3 pop LIF mixer vs pop LIF all', T['pop LIF mixer'], T['pop LIF all']), ('B4 pop f-LIF mixer vs f-LIF', T['pop f-LIF mixer'], T['f-LIF']),
         ('B5 pop f-LIF all vs f-LIF', T['pop f-LIF all'], T['f-LIF']), ('B6 pop f-LIF mixer vs pop f-LIF all', T['pop f-LIF mixer'], T['pop f-LIF all'])]
    C = [(f'C{i} {n}+mean vs {n}', T[f'{n}+mean'], T[n]) for i, n in enumerate(('f-LIF', 'pop LIF all', 'pop LIF mixer', 'pop f-LIF all', 'pop f-LIF mixer'), 1)]
    return B + C


def reports(T, SFD, SFJ):
    R = [('(report) f-LIF vs LIF', T['f-LIF'], T['LIF']), ('(report) LIF+mean vs LIF (= 11-18 V2)', T['LIF+mean'], T['LIF'])]
    R += [('(report, mean) pop LIF mixer vs LIF', T['pop LIF mixer+mean'], T['LIF+mean']), ('(report, mean) pop LIF all vs LIF', T['pop LIF all+mean'], T['LIF+mean']),
          ('(report, mean) pop LIF mixer vs pop LIF all', T['pop LIF mixer+mean'], T['pop LIF all+mean']),
          ('(report, mean) pop f-LIF mixer vs f-LIF', T['pop f-LIF mixer+mean'], T['f-LIF+mean']), ('(report, mean) pop f-LIF all vs f-LIF', T['pop f-LIF all+mean'], T['f-LIF+mean']),
          ('(report, mean) pop f-LIF mixer vs pop f-LIF all', T['pop f-LIF mixer+mean'], T['pop f-LIF all+mean'])]
    R += [(f'(report) {n} vs SF-de', X, SFD) for n, X in T.items()]
    return R + [(f'(report) {n} vs SF-sj', X, SFJ) for n, X in T.items() if n in ('LIF', 'pop LIF mixer', 'pop f-LIF mixer')] + [('(report) SF-de vs SF-sj', SFD, SFJ)]


def ett_tables(T, title, Q, R, pairs=True):
    L = [f'# {title}', '', 'Rules and sources: module docstring of summarize_pn.py. Seeds 42 / 7 / 2026.', '', '## Test MSE (seed mean) and energy', '',
         '| arm | ' + ' | '.join(f'{ds} H{pl}' for ds, pl in CELLS) + ' | mean | mean best validation L1 | energy µJ (4-cell mean) | params (ETTh2 H96) |',
         '|---|--:|--:|--:|--:|--:|--:|--:|--:|']
    for n, X in T.items():
        v = [mean([num(X[(ds, pl, s)], 'mse') for s in SEEDS]) for ds, pl in CELLS]
        val = mean([num(X[(ds, pl, s)], 'val_loss_best') for ds, pl in CELLS for s in SEEDS])
        e = mean([mean([X[(ds, pl, s)]['E'] for s in SEEDS]) for ds, pl in CELLS])
        L.append(f'| {n} | ' + ' | '.join(f'{z:.4f}' for z in v) + f" | {mean(v):.4f} | {val:.5f} | {e:.2f} | {X[('ETTh2', 96, 42)].get('params', '-')} |")
    L += ['', '## Comparisons', '', '| comparison | R % | lower pairs | verdict | energy A/B | per cell r % | MAE R % | validation L1 R % |', '|---|--:|--:|---|--:|---|--:|--:|']
    for name, x, y in Q + R:
        c, m, v = compare(x, y, 'mse'), compare(x, y, 'mae'), compare(x, y, 'val_loss_best')
        L.append(f"| {name} | {c['R']:+.2f} | {c['lo']}/12 | {c['verdict']} | {c['energy']:.2f} | " + ', '.join(f'{z:+.2f}' for z in c['cells']) + f" | {m['R']:+.2f} | {v['R']:+.2f} |")
    if pairs:
        L += ['', 'Pairs r % (test MSE), seed 42, 7, 2026 per cell (verdict comparisons):', '']
        for name, x, y in Q:
            c = compare(x, y, 'mse')
            L.append(f'- {name}: ' + '; '.join(f"{ds} H{pl} " + ', '.join(f'{z:+.2f}' for z in c['pairs'][3 * i:3 * i + 3]) for i, (ds, pl) in enumerate(CELLS)))
    L += ['', '## Report only: stopping and firing rates (seed mean)', '', '| cell | arm | best epoch (seeds) | epochs run (seeds) | last-epoch MSE | rho_hippo | energy µJ | time s (seeds) |',
          '|---|---|---|---|--:|--:|--:|---|']
    for ds, pl in CELLS:
        for n, X in T.items():
            rr = [X[(ds, pl, s)] for s in SEEDS]
            be = [x['best_epoch_val_epochs'].split(':') for x in rr]
            rho = [float(x['rho_hippo']) for x in rr if x.get('rho_hippo') not in (None, 'None')]
            L.append(f"| {ds} H{pl} | {n} | {', '.join(b[0] for b in be)} | {', '.join(b[2] for b in be)} | {mean([num(x, 'last_mse') for x in rr]):.4f} | "
                     f"{(f'{mean(rho):.4f}' if len(rho) == 3 else '-')} | {mean([x['E'] for x in rr]):.2f} | {', '.join(x.get('time_s', '-') for x in rr)} |")
    return L


def mit_tables(T, title, Q, R):
    L = [f'# {title}', '', 'Rules and sources: module docstring of summarize_pn.py. Seeds 42 / 7 / 2026; last-epoch weights on the 30 % test file.', '',
         '## Test values (seed mean)', '', '| arm | macro-F1 % | per seed 42, 7, 2026 | accuracy % | F1 N / S / V / F / Q | rho_hippo | energy µJ | params | time s (seeds) |',
         '|---|--:|---|--:|---|--:|--:|--:|---|']
    for n, X in T.items():
        rr = [X[s] for s in SEEDS]
        cls = ' / '.join(f"{100 * mean([num(x, f'f1_{i}') for x in rr]):.2f}" for i in range(5))
        rho = [float(x['rho_hippo']) for x in rr if x.get('rho_hippo') not in (None, 'None')]
        L.append(f"| {n} | {100 * mean([num(x, 'f1') for x in rr]):.2f} | " + ', '.join(f"{100 * num(x, 'f1'):.2f}" for x in rr) + f" | {100 * mean([num(x, 'overall_acc') for x in rr]):.2f} | {cls} | "
                 f"{(f'{mean(rho):.4f}' if len(rho) == 3 else '-')} | {mean([x['E'] for x in rr]):.2f} | {rr[0].get('params', '-')} | {', '.join(x.get('time_s', '-') for x in rr)} |")
    L += ['', '## Comparisons', '', '| comparison | mean %p | seeds higher | per seed 42, 7, 2026 | verdict | energy A/B |', '|---|--:|--:|---|---|--:|']
    for name, x, y in Q + R:
        c = cmp_mit(x, y)
        L.append(f"| {name} | {c['mean']:+.2f} | {c['hi']}/3 | " + ', '.join(f'{z:+.2f}' for z in c['d']) + f" | {c['verdict']} | {c['energy']:.2f} |")
    return L


def stage_select():
    out = []
    # ETT: per dataset over its two horizons, validation only
    rows = [r for r in read_result(os.path.join(SFE, 'raw.txt'), lambda c: c == 'spk_ssa')]
    V = {}
    for r in rows:
        if not r['tag'].endswith('_nalif'):
            stop(f'unexpected tag {r["tag"]} in {SFE}')
        k = (r['ds'], int(r['pl']), r['lr'], int(r['seed']))
        if k in V:
            stop(f'duplicate {k}')
        if r['layers'] != LY_SF[r['ds']]:
            stop(f'{k}: layers {r["layers"]}, expected {LY_SF[r["ds"]]}')
        best_ok(k, r)
        V[k] = num(r, 'val_loss_best')
    sel_e, sc_e = {}, {}
    for ds in ('ETTh2', 'ETTm2'):
        sm = {}
        for lr in ETT_LR:
            for pl in (96, 720):
                for s in SEEDS:
                    if (ds, pl, lr, s) not in V:
                        stop(f'missing SF-de {ds} H{pl} lr {lr} seed {s}')
        for lr in ETT_LR:
            sm[lr] = mean([mean([V[(ds, pl, lr, s)] for s in SEEDS]) / mean([V[(ds, pl, '0.0005', s)] for s in SEEDS]) for pl in (96, 720)])
        best = min(ETT_LR, key=lambda v: (sm[v], v != '0.0005'))
        if sm[best] > sm['0.0005'] - 1e-12:
            best = '0.0005'
        sel_e[ds], sc_e[ds] = best, sm
    # MIT-BIH: fold-0 validation, last epoch
    F = {}
    for r in read_result(os.path.join(SFM, 'raw.txt'), lambda c: c == 'spk_ssa'):
        if int(r.get('val_fold', -1)) != 0:
            continue
        k = (r['lr'], int(r['seed']))
        if k in F:
            stop(f'duplicate MIT-BIH {k}')
        if r['layers'] != '2' or not r['tag'].endswith('_v0_nalif'):
            stop(f'MIT-BIH {k}: layers {r["layers"]} tag {r["tag"]}')
        F[k] = 100 * num(r, 'val_f1_last')
    for lr in MIT_LR:
        for s in SEEDS:
            if (lr, s) not in F:
                stop(f'missing SF-de MIT-BIH lr {lr} seed {s} on fold 0')
    sm_m = {lr: mean([F[(lr, s)] for s in SEEDS]) for lr in MIT_LR}
    best_m = max(MIT_LR, key=lambda v: (sm_m[v], v == '0.01'))
    if sm_m[best_m] < sm_m['0.01'] + 1e-9:
        best_m = '0.01'
    for p, sel, scores in ((os.path.join(SFE, 'selected_lr.json'), sel_e, sc_e), (os.path.join(SFM, 'selected_lr.json'), {'MITBIH': best_m}, {'MITBIH': sm_m})):
        if os.path.exists(p) and json.load(open(p))['selected'] != sel:
            stop(f'{p} holds a different selection {json.load(open(p))["selected"]} vs {sel}')
        json.dump({'stage': 'SF-de lr (validation only)', 'selected': sel, 'scores': scores}, open(p, 'w'), indent=1)
        out.append(f'{os.path.relpath(p, RES)}: {sel}')
    L = ['# SF-de learning-rate selection (validation only)', '']
    for ds in ('ETTh2', 'ETTm2'):
        L.append(f'- {ds}: selected {sel_e[ds]}; score (mean over H96 / H720 of the seed-mean best validation L1 relative to lr 5e-4): ' +
                 ', '.join(f'{lr} {sc_e[ds][lr]:.5f}' for lr in ETT_LR))
    L.append(f'- MIT-BIH: selected {best_m}; seed-mean last-epoch validation macro-F1 on fold 0: ' + ', '.join(f'{lr} {sm_m[lr]:.3f}' for lr in MIT_LR))
    return os.path.join(SFE, 'selection.md'), L + [''] + out


def stage_sf_ett():
    T = {'LIF SpkMixer': arms_ett_one('LIF'), 'SF-de': load_sfde_ett(), 'SF-sj': spikformer()}
    Q = [('A1 LIF SpkMixer vs SF-de', T['LIF SpkMixer'], T['SF-de'])]
    R = [('(report) SF-de vs SF-sj', T['SF-de'], T['SF-sj']), ('(report) LIF SpkMixer vs SF-sj', T['LIF SpkMixer'], T['SF-sj'])]
    return os.path.join(SFE, 'summary.md'), ett_tables(T, 'Part 1, ETT: LIF SpkMixer vs the Spikformer-style model with the same spikeDE LIF (SF-de)', Q, R)


def arms_ett_one(name):
    _, _, mixer, suf = next(a for a in ARMS if a[0] == name)
    S = where(name, 'ett')
    return load_ett(os.path.join(S, 'raw.txt'), os.path.join(S, 'energy_recount.csv'), lambda ds: f'spkmx_{mixer}_muxlin_iand_lr{LR_MX[ds]}_L1{suf}', f'spkmx_{mixer}_muxlin_iand')


def stage_sf_mit():
    T = {'LIF SpkMixer': load_mit(os.path.join(MITB, 'raw.txt'), os.path.join(MITB, 'energy_recount.csv'), 'spkmx_lif_muxlin_add_lr0.02_L2'),
         'SF-de': load_sfde_mit(), 'SF-sj': load_mit(os.path.join(MITB, 'raw.txt'), os.path.join(MITB, 'energy_recount.csv'), 'spk_ssa_lr0.005_L2')}
    Q = [('A1 LIF SpkMixer vs SF-de', T['LIF SpkMixer'], T['SF-de'])]
    R = [('(report) SF-de vs SF-sj', T['SF-de'], T['SF-sj']), ('(report) LIF SpkMixer vs SF-sj', T['LIF SpkMixer'], T['SF-sj'])]
    return os.path.join(SFM, 'summary.md'), mit_tables(T, 'Part 1, MIT-BIH: LIF SpkMixer vs the Spikformer-style model with the same spikeDE LIF (SF-de)', Q, R)


def stage_ett():
    T = arms_ett()
    SFD, SFJ = load_sfde_ett(), spikformer()
    full = dict(T, **{'SF-de': SFD, 'SF-sj': SFJ})
    return os.path.join(PNE, 'summary.md'), ett_tables(full, 'Parts 2 and 3, ETT: SpkMixer neuron x placement x mean pooling', questions(T), reports(T, SFD, SFJ))


def stage_mit():
    T = arms_mit()
    SFD = load_sfde_mit()
    SFJ = load_mit(os.path.join(MITB, 'raw.txt'), os.path.join(MITB, 'energy_recount.csv'), 'spk_ssa_lr0.005_L2')
    full = dict(T, **{'SF-de': SFD, 'SF-sj': SFJ})
    return os.path.join(PNM, 'summary.md'), mit_tables(full, 'Parts 2 and 3, MIT-BIH: SpkMixer neuron x placement x mean pooling', questions(T), reports(T, SFD, SFJ))


def stage_len336():
    raw, en = os.path.join(L336, 'raw.txt'), os.path.join(L336, 'energy_recount.csv')
    T = {'LIF 336': load_ett(raw, en, lambda ds: f'spkmx_lif_muxlin_iand_lr{LR_MX[ds]}_L1_s336', 'spkmx_lif_muxlin_iand'),
         'pop LIF all 336': load_ett(raw, en, lambda ds: f'spkmx_pop_lif_muxlin_iand_lr{LR_MX[ds]}_L1_s336', 'spkmx_pop_lif_muxlin_iand'),
         'SF-sj 336': load_ett(raw, en, lambda ds: f'spk_ssa_lr{LR_SF[ds]}_L{LY_SF[ds]}_s336', 'spk_ssa')}
    T['pop LIF mixer 336'] = load_ett(os.path.join(P336, 'raw.txt'), os.path.join(P336, 'energy_recount.csv'),
                                      lambda ds: f'spkmx_pop_lif_muxlin_iand_lr{LR_MX[ds]}_L1_nsmixer_s336', 'spkmx_pop_lif_muxlin_iand')
    for k, r in T['pop LIF mixer 336'].items():
        best_ok(k, r)
    T['SF-de 336'] = load_sfde_ett('_s336')
    Q = [('D1 LIF 336 vs SF-de 336', T['LIF 336'], T['SF-de 336']), ('D2 pop LIF mixer 336 vs LIF 336', T['pop LIF mixer 336'], T['LIF 336']),
         ('D3 pop LIF mixer 336 vs pop LIF all 336', T['pop LIF mixer 336'], T['pop LIF all 336'])]
    R = [('(report) pop LIF mixer 336 vs SF-de 336', T['pop LIF mixer 336'], T['SF-de 336']), ('(report) SF-de 336 vs SF-sj 336', T['SF-de 336'], T['SF-sj 336']),
         ('(report) LIF 336 vs SF-sj 336 (= 11-18 P2)', T['LIF 336'], T['SF-sj 336'])]
    return os.path.join(P336, 'summary.md'), ett_tables(T, 'Part 4, ETT input 336: SF-de and pop LIF mixer-only', Q, R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['select', 'sf_ett', 'sf_mit', 'ett', 'mitbih', 'len336']); ap.add_argument('--out')
    a = ap.parse_args()
    p, L = {'select': stage_select, 'sf_ett': stage_sf_ett, 'sf_mit': stage_sf_mit, 'ett': stage_ett, 'mitbih': stage_mit, 'len336': stage_len336}[a.stage]()
    p = a.out or p
    with open(p, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('wrote', p)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
