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
Stage 2 (MODI_PROJECT_LOG 2026-10-15, regimes L<lead>g<neo_grad>): conditions are named <cond>@<regime>; for
each regime R (pre-fixed, exploratory):
  Q-A neocortex contributes:  <n>@R vs hippo_only, n in {neo_lif_both, neo_flif_both}: mean relative MSE <= -1 %
                              and >= 8/12 pairs lower
  Q-B f-LIF helps:            neo_flif_both@R vs neo_lif_both@R: >= 3/4 cells lower and mean <= -0.5 %
  plus the stage-1 criteria (2) and (1) for neo_flif_both@R (control = neo_lif_both@R).
  Confirmation candidate: regimes passing (2) [A or B, with the LIF-control condition] and M2; among several, the
  lowest mean validation MSE of the evaluated (last-epoch) weights.
usage: python summarize_flif.py --suite flif-v1-main [--suite more ...] [--out_suite NAME]
                                [--future results/<dir>/future.csv] [--probe_dir DIR ...] [--stage2]
"""
import os
import re
import csv
import glob
import argparse
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONDS = ['base', 'neo_lif_both', 'neo_lif_mlp', 'neo_flif_both', 'neo_flif_mlp', 'serial_lif', 'serial_flif',
         'serial_lif@Tb1', 'serial_flif@Tb1', 'hippo_only@Tb1', 'hippo_only', 'linear']
CONTROL = {'neo_flif_both': 'neo_lif_both', 'neo_flif_mlp': 'neo_lif_mlp'}


def control_of(cond):
    """LIF control of an f-LIF condition, regime suffix kept (neo_flif_both@L2g1.0 -> neo_lif_both@L2g1.0)."""
    return cond.replace('neo_flif_', 'neo_lif_', 1) if cond.startswith('neo_flif_') else None


def ordered(conds):
    base = [c for c in CONDS if c in conds and c != 'hippo_only']
    extra = sorted(c for c in conds if c not in CONDS)
    return base + extra + (['hippo_only'] if 'hippo_only' in conds else [])


def last_val_mse(run_dir, epoch=None):
    """val_mse of the evaluated weights: the last logged epoch (train.py saves the final state), or `epoch`
    (the best-validation checkpoint) when given."""
    p = os.path.join(run_dir, 'log', 'best_log_0.csv')
    try:
        rows = list(csv.DictReader(open(p)))
        if epoch is not None:
            return float(next(r['val_mse'] for r in rows if int(r['epoch']) == int(epoch)))
        return float(rows[-1]['val_mse'])
    except Exception:
        return float('nan')


WEIGHTS = 'last'                                                     # set from --weights in main()
BEST_FIELDS = ('mse', 'mae', 'syn_energy', 'total_fft_uJ', 'total_direct_uJ', 'rho_neo', 'rho_hippo')


def parse_raw(path):
    rows = []
    for line in open(path):
        if not line.startswith('RESULT'):
            continue
        kv = dict(re.findall(r'(\w+)=(\S+)', line))
        reg = kv.get('regime', 'L0g0.05')
        r = {'cond': kv['cond'] if reg == 'L0g0.05' else f"{kv['cond']}@{reg}", 'regime': reg, 'seed': int(kv['seed']), 'dir': kv.get('dir', '')}
        r['data'], pl = kv['dataset'].split('_p'); r['pl'] = int(pl)
        for k in ('mse', 'mae', 'syn_energy', 'neuron_lif_uJ', 'neuron_fft_uJ', 'neuron_direct_uJ', 'total_fft_uJ',
                  'total_direct_uJ', 'rho_neo', 'rho_hippo', 'time_s', 'params', 'epochs_run', 'patience', 'lr_drops', 'best_epoch'):
            try:
                r[k] = float(kv[k])
            except (KeyError, ValueError):
                r[k] = float('nan')
        if WEIGHTS == 'best':                                        # best-validation checkpoint (run_ett.sh TEST_BEST=1)
            if 'mse_best' not in kv:
                raise SystemExit(f'--weights best but no mse_best in a RESULT line of {path}')
            for k in BEST_FIELDS:
                try:
                    r[k] = float(kv[k + '_best'])
                except (KeyError, ValueError):
                    r[k] = float('nan')
            r['val_mse'] = last_val_mse(r['dir'], kv.get('best_epoch'))
        else:
            r['val_mse'] = last_val_mse(r['dir'])
        rows.append(r)
    return rows


def table(rows, metric='mse', fmt='{:.6f}'):
    cells = sorted({(r['data'], r['pl']) for r in rows})
    out = ['| cond | ' + ' | '.join(f'{d} H{p}' for d, p in cells) + ' |', '|---|' + '---|' * len(cells)]
    for c in ordered({r['cond'] for r in rows}):
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
    for f in [c for c in ordered({r['cond'] for r in rows}) if c.startswith('neo_flif_')]:
        if not any(r['cond'] == 'hippo_only' for r in rows):
            continue
        pc, pairs = paired(rows, f, 'hippo_only')
        if not pc:
            continue
        rel = np.mean([v['rel'] for v in pc.values()]); er = np.mean([v['energy_ratio'] for v in pc.values()])
        all_lower = all(v['rel'] < 0 for v in pc.values()); same = sum(pairs)
        route_a = all_lower and rel <= -0.02 and same >= 10 and er <= 1.3
        route_b = rel <= 0.01 and er <= 0.7
        ctrl = control_of(f)
        pcc, _ = paired(rows, f, ctrl)
        cells_better = sum(v['rel'] < 0 for v in pcc.values())
        plus = cells_better >= 3
        lines.append(f'**{f} vs hippo_only:** cells = ' + ', '.join(f'{d} H{p}: {v["rel"]*100:+.2f}% (E×{v["energy_ratio"]:.3f}, n={v["n"]})' for (d, p), v in pc.items())
                     + f'; mean {rel*100:+.2f}%, pairs lower {same}/{len(pairs)}, energy ×{er:.3f} → route A {"PASS" if route_a else "fail"}, route B {"PASS" if route_b else "fail"}')
        lines.append(f'**{f} vs {ctrl} (same placement):** lower MSE in {cells_better}/{len(pcc)} cells ('
                     + ', '.join(f'{d} H{p}: {v["rel"]*100:+.2f}%' for (d, p), v in pcc.items()) + f') → {"PASS" if plus else "fail"} (need ≥ 3/4)')
        lines.append(f'→ (2) {"**PASS**" if (route_a or route_b) and plus else "**not passed**"} for {f}')
    for c in [c for c in ordered({r['cond'] for r in rows}) if c != 'base']:
        pc, pairs = paired(rows, c, 'base')
        if pc:
            lines.append(f'(report only) {c} vs base: mean {np.mean([v["rel"] for v in pc.values()])*100:+.2f}%, pairs lower {sum(pairs)}/{len(pairs)}')
    return '\n\n'.join(lines)


def stage2(rows, probe_rows):
    """Pre-fixed stage-2 questions per regime (see module docstring). Returns markdown."""
    regs = sorted({r['regime'] for r in rows if r['regime'] != 'L0g0.05'})
    out = ['| regime | Q-A lif vs hippo (mean, pairs) | Q-A flif vs hippo | Q-B flif vs lif (cells, mean) | (2) flif | M2 flif (ETTh2 / ETTm2) | val MSE flif / lif |',
           '|---|---|---|---|---|---|---|']
    cands = []
    for R in ['L0g0.05'] + regs:
        suf = '' if R == 'L0g0.05' else f'@{R}'
        lif, fl = f'neo_lif_both{suf}', f'neo_flif_both{suf}'
        if not any(r['cond'] == fl for r in rows):
            continue
        res = {}
        for n in (lif, fl):
            pc, pairs = paired(rows, n, 'hippo_only')
            m = np.mean([v['rel'] for v in pc.values()]) if pc else float('nan')
            res[n] = (m, sum(pairs), len(pairs), m <= -0.01 and sum(pairs) >= 8)
        pcc, _ = paired(rows, fl, lif)
        cb = sum(v['rel'] < 0 for v in pcc.values()); mb = np.mean([v['rel'] for v in pcc.values()]) if pcc else float('nan')
        qb = cb >= 3 and mb <= -0.005
        pc, pairs = paired(rows, fl, 'hippo_only')
        rel = np.mean([v['rel'] for v in pc.values()]); er = np.mean([v['energy_ratio'] for v in pc.values()])
        ra = all(v['rel'] < 0 for v in pc.values()) and rel <= -0.02 and sum(pairs) >= 10 and er <= 1.3
        rb = rel <= 0.01 and er <= 0.7
        crit2 = (ra or rb) and cb >= 3
        m2 = {}
        for d in ('ETTh2', 'ETTm2'):
            a = probe_rows.get((d, fl)); c = probe_rows.get((d, lif))
            m2[d] = None if (a is None or c is None) else (a >= 0.01 and a > c, a, c)
        m2_ok = all(v is not None and v[0] for v in m2.values())
        vf = np.nanmean([r['val_mse'] for r in rows if r['cond'] == fl]); vl = np.nanmean([r['val_mse'] for r in rows if r['cond'] == lif])
        fmt = lambda t: f"{t[0]*100:+.2f}%, {t[1]}/{t[2]} {'**yes**' if t[3] else 'no'}"
        m2s = ' / '.join('n/a' if v is None else f"{'PASS' if v[0] else 'fail'} ({v[1]*100:+.1f}% vs {v[2]*100:+.1f}%)" for v in m2.values())
        out.append(f"| {R} | {fmt(res[lif])} | {fmt(res[fl])} | {cb}/4, {mb*100:+.2f}% {'**yes**' if qb else 'no'} | "
                   f"{'**PASS**' if crit2 else 'fail'} (A {'y' if ra else 'n'}, B {'y' if rb else 'n'}) | {m2s} | {vf:.5f} / {vl:.5f} |")
        if crit2 and m2_ok:
            cands.append((vf, R))
    out.append('')
    out.append('Confirmation candidate: ' + (f'**{min(cands)[1]}** (lowest validation MSE among {[c[1] for c in cands]})' if cands else '**none** (no regime passes criterion (2) with M2)'))
    return '\n'.join(out)


def stage3(rows, probe_rows):
    """Pre-fixed stage-3 questions (MODI_PROJECT_LOG 2026-10-17, serial 'aligned spiking embedding').
    Conditions: hippo_only@Tb1, serial_{lif,flif}@Tb1, serial_{lif,flif} (backbone T = N); reference hippo_only (T = N,
    suite flif-v1-s2p10, same training rule)."""
    out = []
    def pc_(a, b):
        pc, pairs = paired(rows, a, b)
        if not pc:
            return None
        return (np.mean([v['rel'] for v in pc.values()]), sum(pairs), len(pairs), pc,
                np.mean([v['energy_ratio'] for v in pc.values()]))
    def cells(pc):
        return ', '.join(f"{d} H{p}: {v['rel']*100:+.2f}%" for (d, p), v in pc.items())
    q1 = pc_('serial_flif@Tb1', 'hippo_only@Tb1')
    if q1:
        ok = q1[0] <= -0.01 and q1[1] >= 8
        out.append(f"**Q1 aligned embedding vs plain embedding (both backbone T=1):** serial_flif@Tb1 vs hippo_only@Tb1 -- "
                   f"{cells(q1[3])}; mean {q1[0]*100:+.2f}%, pairs lower {q1[1]}/{q1[2]} -> {'**PASS**' if ok else 'not passed'} "
                   f"(need mean <= -1 % and >= 8/12)")
    q2 = pc_('serial_flif@Tb1', 'hippo_only')
    if q2:
        ok = q2[0] <= 0.01 and q2[4] <= 0.7
        out.append(f"**Q2 vs the hippocampus-only model with the paper repetition (T=24):** serial_flif@Tb1 vs hippo_only -- "
                   f"{cells(q2[3])}; mean {q2[0]*100:+.2f}%, pairs lower {q2[1]}/{q2[2]}, energy x{q2[4]:.3f} -> "
                   f"{'**PASS**' if ok else 'not passed'} (need mean <= +1 % and energy <= 0.7x)")
    for tb in ('@Tb1', ''):
        q3 = pc_(f'serial_flif{tb}', f'serial_lif{tb}')
        if q3:
            nb = sum(v['rel'] < 0 for v in q3[3].values())
            ok = nb >= 3 and q3[0] <= -0.005
            out.append(f"**Q3 f-LIF vs LIF (serial, backbone {'T=1' if tb else 'T=N'}):** {cells(q3[3])}; lower in {nb}/{len(q3[3])} "
                       f"cells, mean {q3[0]*100:+.2f}% -> {'**PASS**' if ok else 'not passed'} (need >= 3/4 and mean <= -0.5 %)")
    for tb in ('@Tb1', ''):
        res = []
        for d in ('ETTh2', 'ETTm2'):
            a = probe_rows.get((d, f'serial_flif{tb}')); c = probe_rows.get((d, f'serial_lif{tb}'))
            res.append(None if a is None or c is None else (a >= 0.01 and a > c, a, c))
        txt = ' / '.join('n/a' if r is None else f"{'PASS' if r[0] else 'fail'} ({r[1]*100:+.1f}% vs {r[2]*100:+.1f}%)" for r in res)
        out.append(f"**Q4 memory M2 (serial f-LIF vs LIF, backbone {'T=1' if tb else 'T=N'}; ETTh2 / ETTm2):** {txt}")
    out.append('')
    out.append('Reported only:')
    for a, b in (('serial_flif', 'serial_flif@Tb1'), ('serial_lif', 'serial_lif@Tb1'), ('hippo_only', 'hippo_only@Tb1'),
                 ('serial_flif@Tb1', 'base'), ('serial_flif@Tb1', 'neo_flif_both'), ('serial_flif', 'hippo_only'),
                 ('serial_lif@Tb1', 'hippo_only@Tb1')):
        r = pc_(a, b)
        if r:
            out.append(f"- {a} vs {b}: {cells(r[3])}; mean {r[0]*100:+.2f}%, pairs lower {r[1]}/{r[2]}, energy x{r[4]:.3f}")
    return '\n'.join(out)


def stage4(rows, probe_rows):
    """Pre-fixed stage-4 questions (MODI_PROJECT_LOG 2026-10-18): the neocortex path alone as a forecaster."""
    out = []
    def pc_(a, b):
        pc, pairs = paired(rows, a, b)
        if not pc:
            return None
        return (np.mean([v['rel'] for v in pc.values()]), sum(pairs), len(pairs), pc,
                np.mean([v['energy_ratio'] for v in pc.values()]))
    def cells(pc):
        return ', '.join(f"{d} H{p}: {v['rel']*100:+.2f}%" for (d, p), v in pc.items())
    def verdict(r, need_mean):
        nb = sum(v['rel'] < 0 for v in r[3].values())
        return nb, (nb >= 3 and r[0] <= need_mean)
    for u in ('step', 'patch'):
        for n in ('lif', 'flif'):
            r = pc_(f'neo_{u}_on_flat_{n}', f'neo_{u}_off_flat_{n}')
            if r:
                nb, ok = verdict(r, -0.01)
                out.append(f"**R1 alignment ({u}, {n}, flat head): on vs off** -- {cells(r[3])}; lower in {nb}/{len(r[3])}, mean "
                           f"{r[0]*100:+.2f}%, pairs {r[1]}/{r[2]}, energy x{r[4]:.2f} -> {'**PASS**' if ok else 'not passed'} (>= 3/4, mean <= -1 %)")
    for n in ('lif', 'flif'):
        for ro in ('flat', 'last'):
            r = pc_(f'neo_step_on_{ro}_{n}', f'neo_patch_on_{ro}_{n}')
            if r:
                nb, ok = verdict(r, -0.01)
                out.append(f"**R2 unit ({n}, {ro}): step (T=96) vs patch (T=24)** -- {cells(r[3])}; lower in {nb}/{len(r[3])}, mean "
                           f"{r[0]*100:+.2f}%, energy x{r[4]:.2f} -> {'**PASS**' if ok else 'not passed'} (>= 3/4, mean <= -1 %)")
    for u in ('step', 'patch'):
        for ro in ('flat', 'last'):
            r = pc_(f'neo_{u}_on_{ro}_flif', f'neo_{u}_on_{ro}_lif')
            if r:
                nb, ok = verdict(r, -0.005)
                out.append(f"**R3 neuron ({u}, {ro}): f-LIF vs LIF** -- {cells(r[3])}; lower in {nb}/{len(r[3])}, mean "
                           f"{r[0]*100:+.2f}% -> {'**PASS**' if ok else 'not passed'} (>= 3/4, mean <= -0.5 %)")
    out += ['', 'R4 (reported only) -- each neocortex-only condition against the references:']
    conds = sorted({r['cond'] for r in rows if r['cond'].startswith('neo_step_') or r['cond'].startswith('neo_patch_')})
    for c in conds:
        parts = []
        for ref in ('hippo_only', 'base', 'linear'):
            r = pc_(c, ref)
            if r:
                parts.append(f"vs {ref} {r[0]*100:+.2f}% ({r[1]}/{r[2]}, E x{r[4]:.2f})")
        out.append(f"- {c}: " + '; '.join(parts))
    for u in ('step', 'patch'):
        for n in ('lif', 'flif'):
            r = pc_(f'neo_{u}_on_last_{n}', f'neo_{u}_on_flat_{n}')
            if r:
                out.append(f"- last vs flat ({u}, {n}): {cells(r[3])}; mean {r[0]*100:+.2f}%, pairs {r[1]}/{r[2]}")
    r = pc_('linear', 'hippo_only')
    if r:
        out.append(f"- linear vs hippo_only: {cells(r[3])}; mean {r[0]*100:+.2f}%, pairs {r[1]}/{r[2]}")
    out += ['', 'Memory probe (aligned runs; dMSE when only the last 12 TOKENS are kept -- 12 steps for step, 12 patches for patch):']
    for c in conds:
        if '_on_' in c:
            vals = [f"{d} {probe_rows[(d, c)]*100:+.1f}%" for d in ('ETTh2', 'ETTm2') if (d, c) in probe_rows]
            if vals:
                out.append(f"- {c}: " + ', '.join(vals))
    return '\n'.join(out)


def probe_dmse(probe_dirs):
    """(data, cond) -> mean dMSE(keep 12)/clean MSE over the probe CSVs (control none)."""
    acc = defaultdict(list)
    for pd_ in probe_dirs:
        for path in glob.glob(os.path.join(pd_, '*_probe.csv')):
            rows = list(csv.DictReader(open(path)))
            if not rows or rows[0]['control'] != 'none':
                continue
            keep = {int(r['x']): float(r['dmse']) for r in rows if r['curve'] == 'keep_last'}
            if 12 in keep:
                acc[(rows[0]['data'], rows[0]['cond'])].append(keep[12] / float(rows[0]['clean_mse']))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def probe_summary(probe_dir):
    """S(12)/S(1) and dMSE(keep 12) per (data, H, cond), seed-averaged, from neo_memory_probe CSVs."""
    acc = defaultdict(lambda: defaultdict(list))
    for path in [p for d in (probe_dir if isinstance(probe_dir, list) else [probe_dir]) for p in glob.glob(os.path.join(d, '*_probe.csv'))]:
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
        if control_of(c) and (d, n, control_of(c)) in acc:
            ctl = acc[(d, n, control_of(c))]
            r, rc = np.mean(v['ratio']), np.mean(ctl['ratio']); m, mc = np.mean(v['dmse12_rel']), np.mean(ctl['dmse12_rel'])
            m1 = r >= 0.10 and (rc <= 0 or r >= 3 * rc); m2 = m >= 0.01 and m > mc
            verdict.append(f'{d} {c}: M1 {"PASS" if m1 else "fail"} (ratio {r:.3f} vs control {rc:.3f}); M2 {"PASS" if m2 else "fail"} (ΔMSE(keep12) {m*100:+.2f}% vs control {mc*100:+.2f}%)')
    return '\n'.join(out) + '\n\n' + '\n'.join(verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', action='append', default=None); ap.add_argument('--future', default=None)
    ap.add_argument('--probe_dir', action='append', default=None); ap.add_argument('--out_suite', default=None)
    ap.add_argument('--stage2', action='store_true')
    ap.add_argument('--stage3', action='store_true')
    ap.add_argument('--stage4', action='store_true')
    ap.add_argument('--weights', choices=['last', 'best'], default='last', help='which test row: last-epoch or best-validation weights')
    ap.add_argument('--summary_name', default='summary.md')
    a = ap.parse_args()
    global WEIGHTS
    WEIGHTS = a.weights
    suites = a.suite or ['flif-v1-main']
    rows = [r for s_ in suites for r in parse_raw(os.path.join(HERE, 'results', s_, 'raw.txt'))]
    name = a.out_suite or suites[0]
    md = [f'# {name} — summary ({len(rows)} RESULT rows from {suites}; test weights: {a.weights})', '',
          '## epochs run / best-validation epoch / lr reductions (mean over seeds)', table(rows, 'epochs_run', '{:.1f}'), '',
          table(rows, 'best_epoch', '{:.1f}'), '', table(rows, 'lr_drops', '{:.2f}'), '',
          '## validation MSE of the evaluated weights (mean over seeds)', table(rows, 'val_mse'), '', '## test MSE (mean ± std over seeds)', table(rows, 'mse'), '',
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
    if a.probe_dir:
        md += ['', '## criterion (1) — memory probe', probe_summary(a.probe_dir)]
    if a.stage2:
        md += ['', '## stage 2 — pre-fixed questions per regime', stage2(rows, probe_dmse(a.probe_dir or []))]
    if a.stage4:
        md += ['', '## stage 4 — pre-fixed questions (neocortex path alone)', stage4(rows, probe_dmse(a.probe_dir or []))]
    if a.stage3:
        md += ['', '## stage 3 — pre-fixed questions (serial aligned spiking embedding)', stage3(rows, probe_dmse(a.probe_dir or []))]
    os.makedirs(os.path.join(HERE, 'results', name), exist_ok=True)
    out = os.path.join(HERE, 'results', name, a.summary_name)
    open(out, 'w').write('\n'.join(md) + '\n'); print('\n'.join(md)); print('->', out)


if __name__ == '__main__':
    main()
