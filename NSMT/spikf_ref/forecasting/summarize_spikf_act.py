"""Summary of the SpikF-LIF experiment (2026-11-26, exp/f-lif-pop-v3; PROJECT_LOG SpikF-LIF pre-registration; user request 11-25:
"SpikF에 GeLU 대신 spiking neuron activation으로 교체한 실험 ... 성능과 에너지 소모량 비교 둘 다").
Input: results/<suite>/raw.txt (run_spikf.sh RESULT lines; 4 ETT datasets x H 96 / 192 / 336 / 720 x seeds 42 / 7 / 2026 x act gelu / lif),
the SpikF paper rows (cinc_mixer_v1/results/ett-full16-20261124/refs16.json) and our two LIF SpkMixer models (cinc_mixer_v1/results/
flif-v1-ett-spkmux-retune/judge.json, 11-25 13:30). Pre-registered quantities:
  A  reproduction: 16-cell means of SpikF-GELU (ours, 3 seeds) against the paper's SpikF rows (report only)
  B  LIF vs GELU, paired by cell and seed: r = 100 (LIF - GELU) / GELU on test MSE; R = mean over the 16 cells of the seed-mean r;
     'LIF worse' = R >= +1 % and >= 32 of the 48 pairs higher, 'LIF better' = R <= -1 % and >= 32 of 48 lower, otherwise 'no clear
     difference'. The same rule on test MAE (reported). Energy: per cell LIF / GELU ratio of the seed means (syops, as SpikF logs it).
  C  our models against SpikF-LIF (and, reported, against SpikF-GELU of these runs): the 11-24 rule -- on 16-cell means ref - ours
     >= 0.001 for MSE and MAE, and syops energy below the reference in all 16 cells -> PASS.
Fail closed (exit 2, nothing written): a SKIP line, a missing / duplicate run, a non-finite value, mixed evaluation settings or
training commits, a run that stopped after more epochs than allowed.
usage (from spikf_ref/forecasting/): python -B summarize_spikf_act.py [--suite spikf-act-ett]   ->   results/<suite>/summary.{md,json}
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
N = os.path.dirname(os.path.dirname(HERE))
DSS = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2')
PLS = ('96', '192', '336', '720')
SEEDS = ('42', '7', '2026')
ACTS = ('gelu', 'lif')
CELLS = [(d, p) for d in DSS for p in PLS]


def stop(msg):
    print('STOP:', msg)
    sys.exit(2)


def load(path):
    rows, skips = [], 0
    for l in open(path):
        if l.startswith('SKIP'):
            skips += 1
        elif l.startswith('RESULT'):
            rows.append(dict(t.split('=', 1) for t in l.split()[1:] if '=' in t))
    return rows, skips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='spikf-act-ett')
    ap.add_argument('--refs', default=os.path.join(N, 'cinc_mixer_v1', 'results', 'ett-full16-20261124', 'refs16.json'))
    ap.add_argument('--ours', default=os.path.join(N, 'cinc_mixer_v1', 'results', 'flif-v1-ett-spkmux-retune', 'judge.json'))
    a = ap.parse_args()
    out = os.path.join(HERE, 'results', a.suite)
    rows, skips = load(os.path.join(out, 'raw.txt'))
    if skips:
        stop(f'{skips} SKIP line(s)')
    V = {}
    for r in rows:
        k = (r['ds'], r['pl'], r['act'], r['seed'])
        if k in V:
            stop(f'duplicate {k}')
        for f in ('mse', 'mae', 'energy_uJ', 'val_loss_best', 'AC_op', 'MAC_op'):
            if not math.isfinite(float(r[f])):
                stop(f'non-finite {f} in {k}')
        if int(r['epochs_run']) > int(r['epochs_max']) or int(r['best_epoch']) < 1:
            stop(f'epochs of {k}')
        V[k] = r
    want = {(d, p, act, s) for d, p in CELLS for act in ACTS for s in SEEDS}
    if set(V) != want:
        stop(f'{len(want - set(V))} missing, {len(set(V) - want)} unexpected runs')
    for f in ('fast', 'sha'):
        if len({r[f] for r in rows}) != 1:
            stop(f'mixed {f}: {sorted({r[f] for r in rows})}')
    refs = json.load(open(a.refs))['models']['SpikF']['rows']
    ours = json.load(open(a.ours))['models']

    def mean(xs):
        return sum(xs) / len(xs)

    C = {}
    for d, p in CELLS:
        for act in ACTS:
            g = [V[(d, p, act, s)] for s in SEEDS]
            mse = [float(x['mse']) for x in g]
            C[(d, p, act)] = {'mse': mean(mse), 'mae': mean([float(x['mae']) for x in g]),
                              'energy_uJ': mean([float(x['energy_uJ']) for x in g]),
                              'mse_std': math.sqrt(sum((v - mean(mse)) ** 2 for v in mse) / 2),
                              'epochs_run': [int(x['epochs_run']) for x in g], 'best_epoch': [int(x['best_epoch']) for x in g]}
    M16 = {act: {m: mean([C[(d, p, act)][m] for d, p in CELLS]) for m in ('mse', 'mae', 'energy_uJ')} for act in ACTS}
    pub16 = {m: mean([refs[f'{d} {p}'][i] for d, p in CELLS]) for i, m in enumerate(('mse', 'mae'))}
    B = {}
    for m in ('mse', 'mae'):
        rc, hi, lo = [], 0, 0
        for d, p in CELLS:
            rs = []
            for s in SEEDS:
                g, l = float(V[(d, p, 'gelu', s)][m]), float(V[(d, p, 'lif', s)][m])
                rs.append(100 * (l - g) / g)
                hi += l > g
                lo += l < g
            rc.append(mean(rs))
        R = mean(rc)
        verdict = 'LIF worse' if R >= 1 and hi >= 32 else 'LIF better' if R <= -1 and lo >= 32 else 'no clear difference'
        B[m] = {'R_pct': R, 'pairs_higher': hi, 'pairs_lower': lo, 'cell_r_pct': dict(zip([f'{d} {p}' for d, p in CELLS], rc)), 'verdict': verdict}
    eratio = {f'{d} {p}': C[(d, p, 'lif')]['energy_uJ'] / C[(d, p, 'gelu')]['energy_uJ'] for d, p in CELLS}
    J = {}
    for mk, mv in ours.items():
        J[mk] = {}
        for act in ACTS:
            mse_m = M16[act]['mse'] - mv['mse16']
            mae_m = M16[act]['mae'] - mv['mae16']
            below = sum(mv['cells'][f'{d} {p}']['syn_energy_uJ'] < C[(d, p, act)]['energy_uJ'] for d, p in CELLS)
            acc = mse_m >= 0.001 - 1e-12 and mae_m >= 0.001 - 1e-12
            J[mk][act] = {'margin_mse': mse_m, 'margin_mae': mae_m, 'accuracy_pass': acc, 'energy_cells_below': below,
                          'verdict': 'PASS' if acc and below == 16 else 'NOT PASSED'}
    S = {'suite': a.suite, 'n_rows': len(rows), 'fast': rows[0]['fast'], 'sha': rows[0]['sha'],
         'cells': {f'{d} {p} {act}': C[(d, p, act)] for d, p in CELLS for act in ACTS},
         'mean16': M16, 'published16': pub16, 'B': B, 'energy_ratio_lif_gelu': eratio,
         'energy_ratio_lif_gelu_mean': mean(list(eratio.values())), 'C': J,
         'ours16': {mk: {'mse': mv['mse16'], 'mae': mv['mae16'], 'energy_uJ': mean([mv['cells'][f'{d} {p}']['syn_energy_uJ'] for d, p in CELLS])}
                    for mk, mv in ours.items()}}
    L = ['# SpikF head activation: GELU (released) vs LIF, input 96, ETT 16 cells', '',
         f'Rules: summarize_spikf_act.py docstring. {len(rows)} runs, evaluation setting `{rows[0]["fast"]}`, training commit `{rows[0]["sha"]}`. '
         'Ours = seed mean of 42 / 7 / 2026 (best-validation weights). Energy = syops count with 0.9 pJ (AC) / 4.6 pJ (MAC), as SpikF logs it.', '',
         '## 16-cell means', '', '| model | MSE | MAE | energy µJ |', '|---|--:|--:|--:|',
         f'| SpikF paper (single published numbers) | {pub16["mse"]:.4f} | {pub16["mae"]:.4f} | — |']
    for act, nm in (('gelu', 'SpikF-GELU (these runs)'), ('lif', 'SpikF-LIF (these runs)')):
        L.append(f'| {nm} | {M16[act]["mse"]:.4f} | {M16[act]["mae"]:.4f} | {M16[act]["energy_uJ"]:.2f} |')
    for mk, v in S['ours16'].items():
        L.append(f'| ours: {ours[mk]["name"]} (11-25) | {v["mse"]:.4f} | {v["mae"]:.4f} | {v["energy_uJ"]:.2f} |')
    L += ['', '## A. Reproduction: SpikF-GELU (these runs) − paper, per dataset (mean of 4 horizons; + = ours higher)', '',
          '| dataset | MSE | MAE |', '|---|--:|--:|']
    for d in DSS:
        g = [C[(d, p, 'gelu')] for p in PLS]
        L.append(f'| {d} | {mean([x["mse"] for x in g]) - mean([refs[f"{d} {p}"][0] for p in PLS]):+.4f} | '
                 f'{mean([x["mae"] for x in g]) - mean([refs[f"{d} {p}"][1] for p in PLS]):+.4f} |')
    L.append(f'| 16 cells | {M16["gelu"]["mse"] - pub16["mse"]:+.4f} | {M16["gelu"]["mae"] - pub16["mae"]:+.4f} |')
    L += ['', '## B. LIF vs GELU (paired by cell and seed)', '']
    for m in ('mse', 'mae'):
        b = B[m]
        L.append(f'- test {m.upper()}: R {b["R_pct"]:+.2f} %, pairs higher {b["pairs_higher"]} / lower {b["pairs_lower"]} of 48 -> **{b["verdict"]}**')
    L.append(f'- energy LIF / GELU: mean of cell ratios {S["energy_ratio_lif_gelu_mean"]:.4f}')
    L += ['', '| cell | GELU MSE / MAE | LIF MSE / MAE | r MSE % | GELU µJ | LIF µJ | LIF / GELU | epochs run G / L |', '|---|---|---|--:|--:|--:|--:|---|']
    for d, p in CELLS:
        g, l = C[(d, p, 'gelu')], C[(d, p, 'lif')]
        L.append(f'| {d} H{p} | {g["mse"]:.4f} / {g["mae"]:.4f} | {l["mse"]:.4f} / {l["mae"]:.4f} | {B["mse"]["cell_r_pct"][f"{d} {p}"]:+.2f} | '
                 f'{g["energy_uJ"]:.2f} | {l["energy_uJ"]:.2f} | {eratio[f"{d} {p}"]:.4f} | {g["epochs_run"]} / {l["epochs_run"]} |')
    L += ['', '## C. Our models against SpikF-LIF (and SpikF-GELU) of these runs (margin = ref − ours, ≥ +0.001 needed for both)', '',
          '| our model | reference | margin MSE / MAE | accuracy | energy cells below | verdict |', '|---|---|---|---|--:|---|']
    for mk in J:
        for act in ('lif', 'gelu'):
            j = J[mk][act]
            L.append(f'| {ours[mk]["name"]} | SpikF-{act.upper()} | {j["margin_mse"]:+.4f} / {j["margin_mae"]:+.4f} | '
                     f'{"yes" if j["accuracy_pass"] else "no"} | {j["energy_cells_below"]} / 16 | **{j["verdict"]}** |')
    L += ['', f'Seed spread (std of test MSE over 3 seeds, mean over 16 cells): GELU {mean([C[(d, p, "gelu")]["mse_std"] for d, p in CELLS]):.4f}, '
          f'LIF {mean([C[(d, p, "lif")]["mse_std"] for d, p in CELLS]):.4f}']
    open(os.path.join(out, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    json.dump(S, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
