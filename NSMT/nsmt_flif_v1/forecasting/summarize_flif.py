"""Aggregate a nsmt_flif_v1 suite and apply the first-stage criteria of MODI_PROJECT_LOG (2026-10-15, §C).

Reads results/<suite>/raw.txt (RESULT lines of scripts/run_ett.sh) and, optionally, a future.csv written by
ett_future.py and the memory-probe CSVs. Writes results/<suite>/summary.md.

Criteria (cells = data x horizon, each the mean of the seeds; pairs = same seed):
  (2) vs hippo_only   route A: every cell lower MSE, mean relative improvement >= 2 %, >= 10/12 pairs same
                      direction, total_fft_uJ <= 1.3 x hippo_only;
                      route B: mean relative MSE difference <= +1 % and total_fft_uJ <= 0.7 x hippo_only;
                      plus: f-LIF below the LIF control of the same placement in >= 3/4 cells.
  (1) memory          M1: S(12)/S(1) >= 0.10 and >= 3 x the LIF control's ratio; M2: dMSE(keep 12) >= 1 % and
                      larger than the LIF control's.
usage: python summarize_flif.py --suite flif-v1-main [--future results/<dir>/future.csv] [--probe_dir DIR]
"""
import os
import re
import csv
import glob
import argparse
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONDS = ['base', 'neo_lif_both', 'neo_lif_mlp', 'neo_flif_both', 'neo_flif_mlp', 'hippo_only']
CONTROL = {'neo_flif_both': 'neo_lif_both', 'neo_flif_mlp': 'neo_lif_mlp'}


def parse_raw(path):
    rows = []
    for line in open(path):
        if not line.startswith('RESULT'):
            continue
        kv = dict(re.findall(r'(\w+)=(\S+)', line))
        r = {'cond': kv['cond'], 'seed': int(kv['seed']), 'dir': kv.get('dir', '')}
        r['data'], pl = kv['dataset'].split('_p'); r['pl'] = int(pl)
        for k in ('mse', 'mae', 'syn_energy', 'neuron_lif_uJ', 'neuron_fft_uJ', 'neuron_direct_uJ', 'total_fft_uJ',
                  'total_direct_uJ', 'rho_neo', 'rho_hippo', 'time_s', 'params', 'epochs_run'):
            try:
                r[k] = float(kv[k])
            except (KeyError, ValueError):
                r[k] = float('nan')
        rows.append(r)
    return rows


def table(rows, metric='mse', fmt='{:.6f}'):
    cells = sorted({(r['data'], r['pl']) for r in rows})
    out = ['| cond | ' + ' | '.join(f'{d} H{p}' for d, p in cells) + ' |', '|---|' + '---|' * len(cells)]
    for c in CONDS:
        vals = []
        for d, p in cells:
            v = [r[metric] for r in rows if r['cond'] == c and r['data'] == d and r['pl'] == p]
            vals.append('–' if not v else (fmt.format(np.mean(v)) + (f' ±{np.std(v):.4f} (n={len(v)})' if metric in ('mse', 'mae') else '')))
        out.append(f'| {c} | ' + ' | '.join(vals) + ' |')
    return '\n'.join(out)


def paired(rows, a, b):
    """Per-seed pairs of cond a vs b; returns per-cell relative difference and pair directions."""
    cells = sorted({(r['data'], r['pl']) for r in rows})
    by = {(r['cond'], r['data'], r['pl'], r['seed']): r for r in rows}
    per_cell, pairs = {}, []
    for d, p in cells:
        seeds = sorted({r['seed'] for r in rows if r['cond'] == a and r['data'] == d and r['pl'] == p}
                       & {r['seed'] for r in rows if r['cond'] == b and r['data'] == d and r['pl'] == p})
        if not seeds:
            continue
        ma = np.mean([by[(a, d, p, s)]['mse'] for s in seeds]); mb = np.mean([by[(b, d, p, s)]['mse'] for s in seeds])
        ea = np.mean([by[(a, d, p, s)]['total_fft_uJ'] for s in seeds]); eb = np.mean([by[(b, d, p, s)]['total_fft_uJ'] for s in seeds])
        per_cell[(d, p)] = {'rel': (ma - mb) / mb, 'mse_a': ma, 'mse_b': mb, 'energy_ratio': ea / eb, 'n': len(seeds)}
        pairs += [by[(a, d, p, s)]['mse'] < by[(b, d, p, s)]['mse'] for s in seeds]
    return per_cell, pairs


def judge(rows):
    lines = []
    for f in ('neo_flif_both', 'neo_flif_mlp'):
        if not any(r['cond'] == f for r in rows) or not any(r['cond'] == 'hippo_only' for r in rows):
            continue
        pc, pairs = paired(rows, f, 'hippo_only')
        if not pc:
            continue
        rel = np.mean([v['rel'] for v in pc.values()]); er = np.mean([v['energy_ratio'] for v in pc.values()])
        all_lower = all(v['rel'] < 0 for v in pc.values()); same = sum(pairs)
        route_a = all_lower and rel <= -0.02 and same >= 10 and er <= 1.3
        route_b = rel <= 0.01 and er <= 0.7
        ctrl = CONTROL[f]
        pcc, _ = paired(rows, f, ctrl)
        cells_better = sum(v['rel'] < 0 for v in pcc.values())
        plus = cells_better >= 3
        lines.append(f'**{f} vs hippo_only:** cells = ' + ', '.join(f'{d} H{p}: {v["rel"]*100:+.2f}% (E×{v["energy_ratio"]:.3f}, n={v["n"]})' for (d, p), v in pc.items())
                     + f'; mean {rel*100:+.2f}%, pairs lower {same}/{len(pairs)}, energy ×{er:.3f} → route A {"PASS" if route_a else "fail"}, route B {"PASS" if route_b else "fail"}')
        lines.append(f'**{f} vs {ctrl} (same placement):** lower MSE in {cells_better}/{len(pcc)} cells ('
                     + ', '.join(f'{d} H{p}: {v["rel"]*100:+.2f}%' for (d, p), v in pcc.items()) + f') → {"PASS" if plus else "fail"} (need ≥ 3/4)')
        lines.append(f'→ (2) {"**PASS**" if (route_a or route_b) and plus else "**not passed**"} for {f}')
    for c in ('neo_lif_both', 'neo_lif_mlp', 'neo_flif_both', 'neo_flif_mlp', 'hippo_only'):
        pc, pairs = paired(rows, c, 'base')
        if pc:
            lines.append(f'(report only) {c} vs base: mean {np.mean([v["rel"] for v in pc.values()])*100:+.2f}%, pairs lower {sum(pairs)}/{len(pairs)}')
    return '\n\n'.join(lines)


def probe_summary(probe_dir):
    """S(12)/S(1) and dMSE(keep 12) per (data, H, cond), seed-averaged, from neo_memory_probe CSVs."""
    acc = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(probe_dir, '*_probe.csv')):
        rows = list(csv.DictReader(open(path)))
        if not rows or rows[0]['control'] != 'none':
            continue
        key = (rows[0]['data'], int(rows[0]['step_n']) + 1, rows[0]['cond'])
        lag = {int(r['x']): float(r['spike_delta_n']) for r in rows if r['curve'] == 'lag'}
        keep = {int(r['x']): float(r['dmse']) for r in rows if r['curve'] == 'keep_last'}
        cm = float(rows[0]['clean_mse'])
        if 1 in lag and 12 in lag and lag[1] > 0:
            acc[key]['ratio'].append(lag[12] / lag[1])
        if 12 in keep:
            acc[key]['dmse12_rel'].append(keep[12] / cm)
        acc[key]['S1'].append(lag.get(1, float('nan')))
    out = ['| data | N | cond | S(1) | S(12)/S(1) | ΔMSE(keep 12)/MSE | n |', '|---|---|---|---|---|---|---|']
    for (d, n, c), v in sorted(acc.items()):
        out.append(f"| {d} | {n} | {c} | {np.mean(v['S1']):.4f} | {np.mean(v['ratio']) if v['ratio'] else float('nan'):.4f} | "
                   f"{np.mean(v['dmse12_rel'])*100 if v['dmse12_rel'] else float('nan'):+.2f}% | {len(v['S1'])} |")
    verdict = []
    for (d, n, c), v in sorted(acc.items()):
        if c in CONTROL and (d, n, CONTROL[c]) in acc:
            ctl = acc[(d, n, CONTROL[c])]
            r, rc = np.mean(v['ratio']), np.mean(ctl['ratio']); m, mc = np.mean(v['dmse12_rel']), np.mean(ctl['dmse12_rel'])
            m1 = r >= 0.10 and (rc <= 0 or r >= 3 * rc); m2 = m >= 0.01 and m > mc
            verdict.append(f'{d} {c}: M1 {"PASS" if m1 else "fail"} (ratio {r:.3f} vs control {rc:.3f}); M2 {"PASS" if m2 else "fail"} (ΔMSE(keep12) {m*100:+.2f}% vs control {mc*100:+.2f}%)')
    return '\n'.join(out) + '\n\n' + '\n'.join(verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='flif-v1-main'); ap.add_argument('--future', default=None); ap.add_argument('--probe_dir', default=None)
    a = ap.parse_args()
    rows = parse_raw(os.path.join(HERE, 'results', a.suite, 'raw.txt'))
    md = [f'# {a.suite} — summary ({len(rows)} RESULT rows)', '', '## test MSE (mean ± std over seeds)', table(rows, 'mse'), '',
          '## test MAE', table(rows, 'mae'), '', '## total energy, paper convention (µJ; syops synaptic + neuron eq. 60/64)', table(rows, 'total_fft_uJ', '{:.3f}'), '',
          '## total energy, direct-sum convention (µJ)', table(rows, 'total_direct_uJ', '{:.3f}'), '',
          '## synaptic energy only (µJ, syops)', table(rows, 'syn_energy', '{:.3f}'), '',
          '## neocortex firing rate', table(rows, 'rho_neo', '{:.4f}'), '', '## hippocampus firing rate', table(rows, 'rho_hippo', '{:.4f}'), '',
          '## training time (s)', table(rows, 'time_s', '{:.0f}'), '', '## criterion (2) — standard ETT test', judge(rows)]
    if a.future and os.path.exists(a.future):
        frows = []
        for r in csv.DictReader(open(a.future)):
            if r['split'] != 'future':
                continue
            frows.append({'cond': r['cond'], 'seed': int(r['seed']), 'data': r['data'], 'pl': int(r['pred_len']), 'mse': float(r['mse']),
                          'mae': float(r['mae']), 'total_fft_uJ': float('nan'), 'syn_energy': float('nan'), 'rho_neo': float('nan'),
                          'rho_hippo': float('nan'), 'time_s': float('nan'), 'total_direct_uJ': float('nan')})
        md += ['', f'## future interval ({len(frows)} rows) — MSE', table(frows, 'mse'), '', '## future interval — MAE', table(frows, 'mae'), '',
               '## future interval — direction check (energy columns not available here)', judge(frows)]
    if a.probe_dir and os.path.isdir(a.probe_dir):
        md += ['', '## criterion (1) — memory probe', probe_summary(a.probe_dir)]
    out = os.path.join(HERE, 'results', a.suite, 'summary.md')
    open(out, 'w').write('\n'.join(md) + '\n'); print('\n'.join(md)); print('->', out)


if __name__ == '__main__':
    main()
