"""Stage 9 decisions and summary (MODI_PROJECT_LOG 2026-10-19).

--decide : from results/<lr_suite>/raw.txt and results/<crop_suite>/raw.txt write results/<crop_suite>/decision.json
           LR    = the lr in {0.001, 0.003} with the higher best-validation score of nsmt_lif seed 42 (tie -> 0.001)
           gate  = passes when best-val score(mix_lif_bidir, full) - best-val score(mix_lif_bidir, 8 tokens) >= 1 %p;
                   a missing run fails the gate (fail-closed)
--summary: results/<suite>/summary.md with per-condition means, energy and the pre-registered verdicts R1-R4.
--aux --suite <flif-v1-rhythm-aux-cinc|-shd>: (MODI 2026-10-20) NSMT aux-loss options against the stage-9 baseline
           (alpha 0.1, DCT low-pass target; same seeds, same code) and against spk_ssa, both read from the stage-9 suite:
           X1 auxoff - base, X2 full - base, X3 a05 / a09 - base (per neuron), X4 every option - spk_ssa (criterion (2)).
--fix --suite <flif-v1-rhythm-fix-cinc|-shd>: (MODI 2026-10-21) cause-separation conditions against the stage-9 suite:
           Y (position) Y1 spk_ssa_nope - spk_ssa, Y2 nsmt_lif_pe - nsmt_lif, Y3 mix_lif_bidir_pe - mix_lif_bidir, Y4 the pe models -
           spk_ssa_nope and - spk_ssa (criterion (2)); Z (reach) Z1 mix_lif_rec_bidir - mix_sjlif_bidir (same neuron), Z2
           mix_pop_bidir_pt16 - mix_pop_bidir, Z3 both - spk_ssa; W (hippocampus T=4) W1 nsmt_*_t4 - nsmt_*, W2 - spk_ssa;
           V (CinC only) psi 1.5 / 2 - psi 1.0. Report: mix_sjlif_bidir - mix_lif_bidir (neuron implementation), val-test gap.
Verdict: A better than B if mean(A-B) >= +0.5 %p and A > B in >= 2 of 3 seeds (CinC) / >= 4 of 5 seeds (SHD); worse if the
mirror holds; otherwise no clear difference; a missing seed or non-finite value -> incomplete.
"""
import argparse
import json
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COND_LABEL = {'nsmt_lif': 'NSMT, spikeDE LIF', 'nsmt_flif': 'NSMT, f-LIF', 'nsmt_pop': 'NSMT, population f-LIF (calibrated)',
              'nsmt_lif_ce': 'NSMT, spikeDE LIF, plain CE', 'spk_ssa': 'Hippo-only, Spikformer SSA (T=4)',
              'mix_flif_bidir': 'Neo-mixer, bidirectional f-LIF', 'mix_pop_bidir': 'Neo-mixer, bidirectional population f-LIF',
              'mix_lif_bidir': 'Neo-mixer, bidirectional spikeDE LIF', 'mix_lif_bidir_ce': 'Neo-mixer, bidirectional LIF, plain CE',
              'mix_none': 'no token mixing (floor)'}


def parse(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path):
        if line.startswith('RESULT '):
            d = dict(re.findall(r'(\S+?)=(\S+)', line))
            for k, v in list(d.items()):
                try:
                    d[k] = float(v)
                except ValueError:
                    pass
            rows.append(d)
    return rows


def best_val(d):
    v = str(d.get('best_epoch_val', '')).split(':')
    return float(v[1]) if len(v) == 2 else float('nan')


def decide(a):
    lr_rows = parse(os.path.join(HERE, 'results', a.lr_suite, 'raw.txt'))
    crop_rows = parse(os.path.join(HERE, 'results', a.crop_suite, 'raw.txt'))
    out = {}
    for data in ('CINC2017', 'SHD'):
        cand = {}
        for r in lr_rows:
            if r.get('data') == data and r.get('cond') == 'nsmt_lif' and int(r.get('seed', -1)) == 42:
                cand[r['lr']] = best_val(r)
        lr = 0.001
        if len(cand) == 2 and all(math.isfinite(v) for v in cand.values()):
            lr = 0.003 if cand.get(0.003, -1) > cand.get(0.001, -1) else 0.001
        sc = {}
        for r in crop_rows:
            if r.get('data') == data and r.get('cond') in ('mix_lif_bidir', 'mix_none'):
                sc[(r['cond'], int(r['crop']))] = best_val(r)
        full, short = sc.get(('mix_lif_bidir', 0), float('nan')), sc.get(('mix_lif_bidir', 8), float('nan'))
        gap = full - short
        out[data] = dict(lr=lr, lr_candidates={str(k): v for k, v in cand.items()}, lr_complete=len(cand) == 2,
                         crop_scores={f'{c}@{k}': v for (c, k), v in sorted(sc.items())}, gate_gap_pp=100 * gap if math.isfinite(gap) else None,
                         gate_pass=bool(math.isfinite(gap) and gap >= 0.01))
    path = os.path.join(HERE, 'results', a.crop_suite, 'decision.json')
    json.dump(out, open(path, 'w'), indent=1)
    print(json.dumps(out, indent=1))
    return out


def verdict(rows, a_cond, b_cond, seeds, need):
    da = {int(r['seed']): float(r['score']) for r in rows if r['cond'] == a_cond}
    db = {int(r['seed']): float(r['score']) for r in rows if r['cond'] == b_cond}
    if not all(s in da and s in db and math.isfinite(da[s]) and math.isfinite(db[s]) for s in seeds):
        return 'incomplete', [], float('nan')
    d = [100 * (da[s] - db[s]) for s in seeds]
    m = float(np.mean(d))
    if m >= 0.5 and sum(x > 0 for x in d) >= need:
        return 'better', d, m
    if m <= -0.5 and sum(x < 0 for x in d) >= need:
        return 'worse', d, m
    return 'no clear difference', d, m


def summary(a):
    R = os.path.join(HERE, 'results', a.suite)
    rows = [r for r in parse(os.path.join(R, 'raw.txt')) if int(r.get('crop', 0)) == 0]
    data = rows[0]['data'] if rows else '?'
    seeds = (42, 7, 2026) if data == 'CINC2017' else (42, 7, 2026, 123, 1024)
    need = 2 if data == 'CINC2017' else 4
    metric = 'CinC score (mean F1 of N, AF, Other)' if data == 'CINC2017' else 'test accuracy'
    L = [f'# Stage 9 `{a.suite}` ({data}) — summary', '', f'Best-validation weights on the test set; {metric}; mean ± sd over seeds {seeds} (%).', '',
         '| condition | n | score | macro-F1 (all classes) | accuracy | last-epoch score | synaptic µJ | total µJ (FFT / direct) | neocortex firing | params | time s |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    def ms(v, s=100., nd=2):
        v = np.asarray([x for x in v], dtype=float)
        return 'n/a' if len(v) == 0 or not np.all(np.isfinite(v)) else f'{s * v.mean():.{nd}f} ± {s * v.std(ddof=1) if len(v) > 1 else 0:.{nd}f}'
    energy = {}
    for c, lab in COND_LABEL.items():
        rr = [r for r in rows if r['cond'] == c]
        if not rr:
            continue
        energy[c] = np.mean([float(r['total_fft_uJ']) for r in rr])
        rho = [r.get('rho_neo') for r in rr]
        rho_s = ms(rho, 1., 4) if all(isinstance(x, float) for x in rho) else 'n/a'
        L.append(f"| {lab} | {len(rr)} | {ms([r['score'] for r in rr])} | {ms([r['macro_f1'] for r in rr])} | {ms([r['acc'] for r in rr])} | "
                 f"{ms([r['last_score'] for r in rr])} | {ms([r['syn_energy_uJ'] for r in rr], 1.)} | {ms([r['total_fft_uJ'] for r in rr], 1.)} / "
                 f"{ms([r['total_direct_uJ'] for r in rr], 1.)} | {rho_s} | {int(rr[0]['params'])} | {np.mean([r['time_s'] for r in rr]):.0f} |")
    Q = [('R1 memory requirement', c, 'mix_none') for c in ('nsmt_lif', 'nsmt_flif', 'nsmt_pop', 'spk_ssa', 'mix_flif_bidir', 'mix_pop_bidir', 'mix_lif_bidir')]
    Q += [('R2 neuron in NSMT', 'nsmt_flif', 'nsmt_lif'), ('R2 neuron in NSMT', 'nsmt_pop', 'nsmt_lif'),
          ('R3 neuron in Neo-mixer', 'mix_flif_bidir', 'mix_lif_bidir'), ('R3 neuron in Neo-mixer', 'mix_pop_bidir', 'mix_lif_bidir')]
    Q += [('R4 vs Hippo-only (Spikformer)', c, 'spk_ssa') for c in ('mix_flif_bidir', 'mix_pop_bidir', 'mix_lif_bidir', 'nsmt_lif', 'nsmt_flif', 'nsmt_pop')]
    if data == 'CINC2017':
        Q += [('R5 CAL vs plain CE (report)', 'nsmt_lif', 'nsmt_lif_ce'), ('R5 CAL vs plain CE (report)', 'mix_lif_bidir', 'mix_lif_bidir_ce')]
    L += ['', '## Pre-registered comparisons (%p, paired by seed)', '', '| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |',
          '|---|---|---|---|---|---|---|---|']
    for q, x, y in Q:
        v, d, m = verdict(rows, x, y, seeds, need)
        er = energy.get(x, float('nan')) / energy.get(y, float('nan')) if energy.get(y) else float('nan')
        crit = ''
        if q.startswith('R4') and v != 'incomplete':
            crit = 'met' if (v == 'better') or (m >= -0.5 and math.isfinite(er) and er <= 0.5) else 'not met'
        L.append(f"| {q} | {COND_LABEL.get(x, x)} | {COND_LABEL.get(y, y)} | {' / '.join(f'{z:+.2f}' for z in d)} | {m:+.2f} | {v} | {er:.2f} | {crit} |")
    open(os.path.join(R, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


AUX_OPTS = (('auxoff', 'aux off (alpha 0, plain CE)'), ('', 'alpha 0.1, DCT target (stage 9 baseline)'), ('a05', 'alpha 0.5, DCT target'),
            ('a09', 'alpha 0.9, DCT target'), ('full', 'alpha 0.1, full-signal target'))
AUX_NEURONS = (('nsmt_lif', 'NSMT, spikeDE LIF'), ('nsmt_pop', 'NSMT, population f-LIF (calibrated)'), ('nsmt_flif', 'NSMT, f-LIF'))


def _last_train(d):
    """final-epoch train CE and MSE from the run's log/best_log_0.csv (nan if missing)."""
    try:
        import csv
        r = list(csv.DictReader(open(os.path.join(str(d), 'log', 'best_log_0.csv'))))[-1]
        return float(r['train_ce']), float(r['train_mse'])
    except Exception:
        return float('nan'), float('nan')


def aux_summary(a):
    R = os.path.join(HERE, 'results', a.suite)
    base_suite = a.suite.replace('-aux', '')
    rows = [r for r in parse(os.path.join(R, 'raw.txt')) if int(r.get('crop', 0)) == 0]
    rows += [r for r in parse(os.path.join(HERE, 'results', base_suite, 'raw.txt'))
             if int(r.get('crop', 0)) == 0 and r['cond'] in ('nsmt_lif', 'nsmt_pop', 'nsmt_flif', 'spk_ssa')]
    data = 'CINC2017' if 'cinc' in a.suite else 'SHD'
    seeds = (42, 7, 2026) if data == 'CINC2017' else (42, 7, 2026, 123, 1024)
    need = 2 if data == 'CINC2017' else 4
    metric = 'CinC score (mean F1 of N, AF, Other)' if data == 'CINC2017' else 'test accuracy'
    def ms(v, s=100., nd=2):
        v = np.asarray(list(v), dtype=float)
        return 'n/a' if len(v) == 0 or not np.all(np.isfinite(v)) else f'{s * v.mean():.2f} ± {s * v.std(ddof=1) if len(v) > 1 else 0:.2f}' if nd == 2 else f'{s * v.mean():.{nd}f}'
    L = [f'# NSMT aux-loss options `{a.suite}` ({data}) — summary', '',
         f'Best-validation weights on the test set; {metric}; mean ± sd over seeds {seeds} (%). Baseline rows and spk_ssa come from '
         f'`results/{base_suite}/raw.txt`. Final-epoch train CE / MSE from each run\'s log/best_log_0.csv (MSE is logged even when the aux loss is off).', '',
         '| neuron | aux setting | n | score | macro-F1 | last-epoch score | final train CE | final train MSE | total µJ (FFT) | neocortex firing | time s |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    energy = {}
    for n, nl in AUX_NEURONS:
        for o, ol in AUX_OPTS:
            c = n if not o else f'{n}_{o}'
            rr = [r for r in rows if r['cond'] == c]
            if not rr:
                L.append(f'| {nl} | {ol} | 0 | n/a | | | | | | | |'); continue
            energy[c] = np.mean([float(r['total_fft_uJ']) for r in rr])
            tr = [_last_train(r['dir']) for r in rr]
            rho = [r.get('rho_neo') for r in rr]
            L.append(f"| {nl} | {ol} | {len(rr)} | {ms([r['score'] for r in rr])} | {ms([r['macro_f1'] for r in rr])} | {ms([r['last_score'] for r in rr])} | "
                     f"{ms([t[0] for t in tr], 1., 4)} | {ms([t[1] for t in tr], 1., 4)} | {energy[c]:.1f} | "
                     f"{ms(rho, 1., 4) if all(isinstance(x, float) for x in rho) else 'n/a'} | {np.mean([r['time_s'] for r in rr]):.0f} |")
    rr = [r for r in rows if r['cond'] == 'spk_ssa']
    if rr:
        energy['spk_ssa'] = np.mean([float(r['total_fft_uJ']) for r in rr])
        L.append(f"| Hippo-only, Spikformer SSA (T=4) | – | {len(rr)} | {ms([r['score'] for r in rr])} | {ms([r['macro_f1'] for r in rr])} | "
                 f"{ms([r['last_score'] for r in rr])} | | | {energy['spk_ssa']:.1f} | n/a | {np.mean([r['time_s'] for r in rr]):.0f} |")
    Q = []
    for n, _ in AUX_NEURONS:
        Q += [('X1 aux loss off', f'{n}_auxoff', n), ('X2 full-signal target', f'{n}_full', n),
              ('X3 alpha 0.5', f'{n}_a05', n), ('X3 alpha 0.9', f'{n}_a09', n)]
    for n, _ in AUX_NEURONS:
        Q += [('X4 vs Hippo-only (Spikformer)', f'{n}_{o}', 'spk_ssa') for o in ('auxoff', 'full', 'a05', 'a09')]
    lab = {n if not o else f'{n}_{o}': f'{nl} / {ol}' for n, nl in AUX_NEURONS for o, ol in AUX_OPTS}
    lab['spk_ssa'] = 'Hippo-only, Spikformer SSA (T=4)'
    L += ['', '## Pre-registered comparisons (%p, paired by seed)', '', '| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |',
          '|---|---|---|---|---|---|---|---|']
    for q, x, y in Q:
        v, d, m = verdict(rows, x, y, seeds, need)
        er = energy.get(x, float('nan')) / energy.get(y, float('nan')) if energy.get(y) else float('nan')
        crit = ''
        if q.startswith('X4') and v != 'incomplete':
            crit = 'met' if (v == 'better') or (m >= -0.5 and math.isfinite(er) and er <= 0.5) else 'not met'
        L.append(f"| {q} | {lab.get(x, x)} | {lab.get(y, y)} | {' / '.join(f'{z:+.2f}' for z in d)} | {m:+.2f} | {v} | {er:.2f} | {crit} |")
    open(os.path.join(R, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


FIX_LABEL = {'spk_ssa': 'Hippo-only, Spikformer SSA (T=4), relative bias', 'spk_ssa_nope': 'Hippo-only, Spikformer SSA (T=4), no position',
             'nsmt_lif': 'NSMT LIF (T=N)', 'nsmt_lif_pe': 'NSMT LIF + absolute position', 'nsmt_pop': 'NSMT population (T=N, psi 1)',
             'mix_lif_bidir': 'Neo-mixer bidir spikeDE LIF', 'mix_lif_bidir_pe': 'Neo-mixer bidir LIF + absolute position',
             'mix_sjlif_bidir': 'Neo-mixer bidir spikingjelly LIF (control)', 'mix_lif_rec_bidir': 'Neo-mixer bidir recurrent spikingjelly LIF',
             'mix_pop_bidir': 'Neo-mixer bidir population (taus 4-32, psi 1)', 'mix_pop_bidir_pt16': 'Neo-mixer bidir population (taus 16-128)',
             'nsmt_lif_t4': 'NSMT LIF, hippocampus T=4 + token cross-attention', 'nsmt_pop_t4': 'NSMT population, hippocampus T=4 + token cross-attention',
             'nsmt_pop_psi1.5': 'NSMT population psi 1.5', 'nsmt_pop_psi2': 'NSMT population psi 2', 'mix_pop_bidir_psi1.5': 'Neo-mixer bidir population psi 1.5',
             'mix_pop_bidir_psi2': 'Neo-mixer bidir population psi 2', 'nsmt_pop_pe': 'NSMT population + absolute position',
             'mix_pop_bidir_pe': 'Neo-mixer bidir population + absolute position', 'mix_pop_rec_bidir': 'Neo-mixer bidir recurrent population f-LIF'}
FIX_ORDER = ('spk_ssa', 'spk_ssa_nope', 'nsmt_lif', 'nsmt_lif_pe', 'nsmt_lif_t4', 'nsmt_pop', 'nsmt_pop_pe', 'nsmt_pop_t4', 'nsmt_pop_psi1.5', 'nsmt_pop_psi2',
             'mix_lif_bidir', 'mix_lif_bidir_pe', 'mix_sjlif_bidir', 'mix_lif_rec_bidir', 'mix_pop_bidir', 'mix_pop_bidir_pe', 'mix_pop_rec_bidir',
             'mix_pop_bidir_pt16', 'mix_pop_bidir_psi1.5', 'mix_pop_bidir_psi2')


def fix_summary(a):
    R = os.path.join(HERE, 'results', a.suite)
    base_suite = a.suite.replace('-fix', '')
    rows = [r for r in parse(os.path.join(R, 'raw.txt')) if int(r.get('crop', 0)) == 0]
    rows += [r for r in parse(os.path.join(HERE, 'results', base_suite, 'raw.txt'))
             if int(r.get('crop', 0)) == 0 and r['cond'] in ('spk_ssa', 'nsmt_lif', 'nsmt_pop', 'mix_lif_bidir', 'mix_pop_bidir')]
    data = 'CINC2017' if 'cinc' in a.suite else 'SHD'
    seeds = (42, 7, 2026) if data == 'CINC2017' else (42, 7, 2026, 123, 1024)
    need = 2 if data == 'CINC2017' else 4
    metric = 'CinC score (mean F1 of N, AF, Other)' if data == 'CINC2017' else 'test accuracy'
    def ms(v, s=100., nd=2):
        v = np.asarray(list(v), dtype=float)
        return 'n/a' if len(v) == 0 or not np.all(np.isfinite(v)) else f'{s * v.mean():.{nd}f} ± {s * v.std(ddof=1) if len(v) > 1 else 0:.{nd}f}'
    L = [f'# Cause-separation `{a.suite}` ({data}) — summary', '',
         f'Best-validation weights on the test set; {metric}; mean ± sd over seeds {seeds} (%). Stage-9 rows come from `results/{base_suite}/raw.txt`.', '',
         '| condition | n | score | best val | val − test | macro-F1 | last-epoch score | total µJ (FFT) | neocortex / mixer firing | params | time s |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    energy = {}
    for c in FIX_ORDER:
        rr = [r for r in rows if r['cond'] == c]
        if not rr:
            continue
        energy[c] = np.mean([float(r['total_fft_uJ']) for r in rr])
        val = [best_val(r) for r in rr]; sc = [float(r['score']) for r in rr]
        rho = [r.get('rho_neo') for r in rr]
        L.append(f"| {FIX_LABEL.get(c, c)} | {len(rr)} | {ms(sc)} | {ms(val)} | {ms([v - s for v, s in zip(val, sc)])} | {ms([r['macro_f1'] for r in rr])} | "
                 f"{ms([r['last_score'] for r in rr])} | {energy[c]:.1f} | {ms(rho, 1., 4) if all(isinstance(x, float) for x in rho) else 'n/a'} | "
                 f"{int(rr[0]['params'])} | {np.mean([r['time_s'] for r in rr]):.0f} |")
    Q = [('Y1 relative bias removed', 'spk_ssa_nope', 'spk_ssa'), ('Y2 absolute position in NSMT', 'nsmt_lif_pe', 'nsmt_lif'),
         ('Y3 absolute position in Neo-mixer', 'mix_lif_bidir_pe', 'mix_lif_bidir'),
         ('Y4 position matched (none): NSMT+pos vs SSA no-pos', 'nsmt_lif_pe', 'spk_ssa_nope'), ('Y4 position matched (none): mixer+pos vs SSA no-pos', 'mix_lif_bidir_pe', 'spk_ssa_nope'),
         ('Y4 vs Hippo-only (criterion 2)', 'nsmt_lif_pe', 'spk_ssa'), ('Y4 vs Hippo-only (criterion 2)', 'mix_lif_bidir_pe', 'spk_ssa'),
         ('Z1 recurrence (same neuron)', 'mix_lif_rec_bidir', 'mix_sjlif_bidir'), ('Z1 report: spikingjelly vs spikeDE LIF', 'mix_sjlif_bidir', 'mix_lif_bidir'),
         ('Z2 slower population branches', 'mix_pop_bidir_pt16', 'mix_pop_bidir'),
         ('Z3 vs Hippo-only (criterion 2)', 'mix_lif_rec_bidir', 'spk_ssa'), ('Z3 vs Hippo-only (criterion 2)', 'mix_pop_bidir_pt16', 'spk_ssa'),
         ('Y2p absolute position in NSMT (population)', 'nsmt_pop_pe', 'nsmt_pop'), ('Y3p absolute position in Neo-mixer (population)', 'mix_pop_bidir_pe', 'mix_pop_bidir'),
         ('Z1p recurrence (population)', 'mix_pop_rec_bidir', 'mix_pop_bidir'),
         ('W1 hippocampus T=4 + token xattn', 'nsmt_lif_t4', 'nsmt_lif'), ('W1 hippocampus T=4 + token xattn', 'nsmt_pop_t4', 'nsmt_pop'),
         ('W2 vs Hippo-only (criterion 2)', 'nsmt_lif_t4', 'spk_ssa'), ('W2 vs Hippo-only (criterion 2)', 'nsmt_pop_t4', 'spk_ssa')]
    if data == 'CINC2017':
        Q += [('V1 soma init', 'nsmt_pop_psi1.5', 'nsmt_pop'), ('V1 soma init', 'nsmt_pop_psi2', 'nsmt_pop'),
              ('V1 soma init', 'mix_pop_bidir_psi1.5', 'mix_pop_bidir'), ('V1 soma init', 'mix_pop_bidir_psi2', 'mix_pop_bidir')]
    L += ['', '## Pre-registered comparisons (%p, paired by seed)', '', '| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |',
          '|---|---|---|---|---|---|---|---|']
    for q, x, y in Q:
        v, d, m = verdict(rows, x, y, seeds, need)
        er = energy.get(x, float('nan')) / energy.get(y, float('nan')) if energy.get(y) else float('nan')
        crit = ''
        if 'criterion 2' in q and v != 'incomplete':
            crit = 'met' if (v == 'better') or (m >= -0.5 and math.isfinite(er) and er <= 0.5) else 'not met'
        L.append(f"| {q} | {FIX_LABEL.get(x, x)} | {FIX_LABEL.get(y, y)} | {' / '.join(f'{z:+.2f}' for z in d)} | {m:+.2f} | {v} | {er:.2f} | {crit} |")
    open(os.path.join(R, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


BIX_NEURONS = ('lif', 'flif', 'pop_lif', 'pop_flif', 'popg_lif', 'popg_flif', 'popk_lif', 'popk_flif')
BIX_NAME = {'lif': 'spikeDE LIF', 'flif': 'f-LIF', 'pop_lif': 'population LIF (psi 0.5)', 'pop_flif': 'population f-LIF (psi 1)',
            'popg_lif': 'soma-less tau groups, LIF', 'popg_flif': 'soma-less tau groups, f-LIF', 'popk_lif': 'soma-less streams, LIF',
            'popk_flif': 'soma-less streams, f-LIF'}


def _mix_cond(n):
    """Neo-mixer (OR) condition of each neuron: stage-9 names for lif / flif / pop_flif."""
    return {'lif': 'mix_lif_bidir', 'flif': 'mix_flif_bidir', 'pop_flif': 'mix_pop_bidir'}.get(n, f'mix_{n}_bidir')


def bix_summary(a):
    """MODI 2026-10-21 idea batch: B1-B4, P1, F1, A1 (see the pre-registration)."""
    R = os.path.join(HERE, 'results', a.suite)
    base_suite = a.suite.replace('-bix', '')
    rows = [r for r in parse(os.path.join(R, 'raw.txt')) if int(r.get('crop', 0)) == 0]
    rows += [r for r in parse(os.path.join(HERE, 'results', base_suite, 'raw.txt'))
             if int(r.get('crop', 0)) == 0 and r['cond'] in ('spk_ssa', 'mix_lif_bidir', 'mix_flif_bidir', 'mix_pop_bidir', 'mix_none')]
    data = 'CINC2017' if 'cinc' in a.suite else 'SHD'
    seeds = (42, 7, 2026) if data == 'CINC2017' else (42, 7, 2026, 123, 1024)
    need = 2 if data == 'CINC2017' else 4
    metric = 'CinC score (mean F1 of N, AF, Other)' if data == 'CINC2017' else 'test accuracy'
    lab = {'spk_ssa': 'Spikformer-style SSA (T=4)', 'spk_ssa_abn': 'Spikformer-style SSA (T=4) + attention BN', 'mix_none': 'no mixing (floor)'}
    for n in BIX_NEURONS:
        lab[f'bix_{n}'] = f'BiX (head), {BIX_NAME[n]}'; lab[f'bixch_{n}'] = f'BiX-ch (per channel), {BIX_NAME[n]}'; lab[_mix_cond(n)] = f'Neo-mixer (OR), {BIX_NAME[n]}'
    order = ['spk_ssa', 'spk_ssa_abn', 'mix_none'] + [_mix_cond(n) for n in BIX_NEURONS] + [f'bix_{n}' for n in BIX_NEURONS] + [f'bixch_{n}' for n in BIX_NEURONS[:4]]
    def ms(v, s=100., nd=2):
        v = np.asarray(list(v), dtype=float)
        return 'n/a' if len(v) == 0 or not np.all(np.isfinite(v)) else f'{s * v.mean():.{nd}f} ± {s * v.std(ddof=1) if len(v) > 1 else 0:.{nd}f}'
    L = [f'# Idea batch `{a.suite}` ({data}) — summary', '',
         f'Best-validation weights on the test set; {metric}; mean ± sd over seeds {seeds} (%). Stage-9 rows from `results/{base_suite}/raw.txt`. '
         'SHD rows compare structures within native-time processing only (no T = N claim on SHD).', '',
         '| condition | n | score | best val | val − test | macro-F1 | total µJ (FFT / direct) | neocortex firing | params | time s |', '|---|---|---|---|---|---|---|---|---|---|']
    energy = {}
    for c in order:
        rr = [r for r in rows if r['cond'] == c]
        if not rr:
            L.append(f'| {lab.get(c, c)} | 0 | n/a | | | | | | | |'); continue
        energy[c] = np.mean([float(r['total_fft_uJ']) for r in rr])
        val = [best_val(r) for r in rr]; sc = [float(r['score']) for r in rr]; rho = [r.get('rho_neo') for r in rr]
        L.append(f"| {lab.get(c, c)} | {len(rr)} | {ms(sc)} | {ms(val)} | {ms([v - s for v, s in zip(val, sc)])} | {ms([r['macro_f1'] for r in rr])} | "
                 f"{energy[c]:.1f} / {np.mean([float(r['total_direct_uJ']) for r in rr]):.1f} | {ms(rho, 1., 4) if all(isinstance(x, float) for x in rho) else 'n/a'} | "
                 f"{int(rr[0]['params'])} | {np.mean([r['time_s'] for r in rr]):.0f} |")
    Q = [('B1 OR -> cross-attention', f'bix_{n}', _mix_cond(n)) for n in BIX_NEURONS]
    Q += [('B2 vs Spikformer (criterion 2)', f'bix_{n}', 'spk_ssa') for n in BIX_NEURONS]
    Q += [('B2 vs Spikformer + attention BN (criterion 2)', f'bix_{n}', 'spk_ssa_abn') for n in BIX_NEURONS]
    Q += [('B3 neuron inside BiX', f'bix_{n}', 'bix_lif') for n in BIX_NEURONS[1:]]
    Q += [('B3 fractional order inside BiX', f'bix_{f}', f'bix_{l}') for l, f in (('lif', 'flif'), ('pop_lif', 'pop_flif'), ('popg_lif', 'popg_flif'), ('popk_lif', 'popk_flif'))]
    Q += [('B4 per-channel vs head form', f'bixch_{n}', f'bix_{n}') for n in BIX_NEURONS[:4]]
    Q += [('B4 per-channel vs Neo-mixer (OR)', f'bixch_{n}', _mix_cond(n)) for n in BIX_NEURONS[:4]]
    for f in ('lif', 'flif'):
        Q += [('P1 soma -> tau groups (Neo-mixer)', _mix_cond(f'popg_{f}'), _mix_cond(f'pop_{f}')), ('P1 soma -> streams (Neo-mixer)', _mix_cond(f'popk_{f}'), _mix_cond(f'pop_{f}')),
              ('P1 groups -> streams (separation, Neo-mixer)', _mix_cond(f'popk_{f}'), _mix_cond(f'popg_{f}')),
              ('P1 soma -> tau groups (BiX)', f'bix_popg_{f}', f'bix_pop_{f}'), ('P1 soma -> streams (BiX)', f'bix_popk_{f}', f'bix_pop_{f}'),
              ('P1 groups -> streams (separation, BiX)', f'bix_popk_{f}', f'bix_popg_{f}')]
    Q += [('F1 fractional order, Neo-mixer population', 'mix_pop_bidir', 'mix_pop_lif_bidir'), ('A1 attention BN', 'spk_ssa_abn', 'spk_ssa')]
    L += ['', '## Pre-registered comparisons (%p, paired by seed)', '', '| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |',
          '|---|---|---|---|---|---|---|---|']
    for q, x, y in Q:
        v, d, m = verdict(rows, x, y, seeds, need)
        er = energy.get(x, float('nan')) / energy.get(y, float('nan')) if energy.get(y) else float('nan')
        crit = ''
        if 'criterion 2' in q and v != 'incomplete':
            crit = 'met' if (v == 'better') or (m >= -0.5 and math.isfinite(er) and er <= 0.5) else 'not met'
        L.append(f"| {q} | {lab.get(x, x)} | {lab.get(y, y)} | {' / '.join(f'{z:+.2f}' for z in d)} | {m:+.2f} | {v} | {er:.2f} | {crit} |")
    open(os.path.join(R, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def lin_summary(a):
    """MODI 2026-10-22: linear-fusion Neo-mixer (mixlin_*) vs OR (stage 9 / idea batch), BiX (idea batch) and Spikformer-style."""
    R = os.path.join(HERE, 'results', a.suite)
    rows = [r for r in parse(os.path.join(R, 'raw.txt')) if int(r.get('crop', 0)) == 0]
    rows += [r for r in parse(os.path.join(HERE, 'results', 'flif-v1-rhythm-bix-cinc', 'raw.txt')) if int(r.get('crop', 0)) == 0]
    rows += [r for r in parse(os.path.join(HERE, 'results', 'flif-v1-rhythm-cinc', 'raw.txt'))
             if int(r.get('crop', 0)) == 0 and r['cond'] in ('spk_ssa', 'mix_lif_bidir', 'mix_flif_bidir', 'mix_pop_bidir', 'mix_none')]
    seeds, need = (42, 7, 2026), 2
    ns = ('lif', 'flif', 'pop_lif', 'pop_flif')
    lab = {'spk_ssa': 'Spikformer-style SSA (T=4)', 'spk_ssa_abn': 'Spikformer-style SSA + attention BN', 'mix_none': 'no mixing (floor)'}
    for n in ns:
        lab[f'mixlin_{n}'] = f'Neo-mixer linear fusion, {BIX_NAME[n]}'; lab[_mix_cond(n)] = f'Neo-mixer OR, {BIX_NAME[n]}'; lab[f'bix_{n}'] = f'BiX, {BIX_NAME[n]}'
    order = ['spk_ssa', 'spk_ssa_abn', 'mix_none'] + [c for n in ns for c in (_mix_cond(n), f'mixlin_{n}', f'bix_{n}')]
    def ms(v, s=100., nd=2):
        v = np.asarray(list(v), dtype=float)
        return 'n/a' if len(v) == 0 or not np.all(np.isfinite(v)) else f'{s * v.mean():.{nd}f} ± {s * v.std(ddof=1) if len(v) > 1 else 0:.{nd}f}'
    L = [f'# Linear-fusion Neo-mixer `{a.suite}` (CINC2017) — summary', '', 'Best-validation weights on the test set; CinC score; mean ± sd over seeds (42, 7, 2026) (%).', '',
         '| condition | n | score | best val | val − test | total µJ (FFT) | neocortex firing | params |', '|---|---|---|---|---|---|---|---|']
    energy = {}
    for c in order:
        rr = [r for r in rows if r['cond'] == c]
        if not rr:
            L.append(f'| {lab.get(c, c)} | 0 | n/a | | | | | |'); continue
        energy[c] = np.mean([float(r['total_fft_uJ']) for r in rr])
        val = [best_val(r) for r in rr]; sc = [float(r['score']) for r in rr]; rho = [r.get('rho_neo') for r in rr]
        L.append(f"| {lab.get(c, c)} | {len(rr)} | {ms(sc)} | {ms(val)} | {ms([v - s for v, s in zip(val, sc)])} | {energy[c]:.1f} | "
                 f"{ms(rho, 1., 4) if all(isinstance(x, float) for x in rho) else 'n/a'} | {int(rr[0]['params'])} |")
    Q = [('L1 OR -> linear fusion', f'mixlin_{n}', _mix_cond(n)) for n in ns] + [('L2 linear fusion vs BiX', f'mixlin_{n}', f'bix_{n}') for n in ns]
    Q += [('L3 vs Spikformer (criterion 2)', f'mixlin_{n}', 'spk_ssa') for n in ns]
    L += ['', '## Pre-registered comparisons (%p, paired by seed)', '', '| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |', '|---|---|---|---|---|---|---|---|']
    for q, x, y in Q:
        v, d, m = verdict(rows, x, y, seeds, need)
        er = energy.get(x, float('nan')) / energy.get(y, float('nan')) if energy.get(y) else float('nan')
        crit = ''
        if 'criterion 2' in q and v != 'incomplete':
            crit = 'met' if (v == 'better') or (m >= -0.5 and math.isfinite(er) and er <= 0.5) else 'not met'
        L.append(f"| {q} | {lab.get(x, x)} | {lab.get(y, y)} | {' / '.join(f'{z:+.2f}' for z in d)} | {m:+.2f} | {v} | {er:.2f} | {crit} |")
    open(os.path.join(R, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def _table_and_verdicts(a, title, rows, order, lab, Q, note):
    R = os.path.join(HERE, 'results', a.suite)
    seeds, need = (42, 7, 2026), 2
    def ms(v, sc=100., nd=2):
        v = np.asarray(list(v), dtype=float)
        return 'n/a' if len(v) == 0 or not np.all(np.isfinite(v)) else f'{sc * v.mean():.{nd}f} ± {sc * v.std(ddof=1) if len(v) > 1 else 0:.{nd}f}'
    L = [f'# {title} `{a.suite}` (CINC2017) — summary', '', note, '',
         '| condition | n | score | macro-F1 | best val | val − test | total µJ (FFT) | neocortex firing | params |', '|---|---|---|---|---|---|---|---|---|']
    energy = {}
    for c in order:
        rr = [r for r in rows if r['cond'] == c]
        if not rr:
            L.append(f'| {lab.get(c, c)} | 0 | n/a | | | | | | |'); continue
        energy[c] = np.mean([float(r['total_fft_uJ']) for r in rr])
        val = [best_val(r) for r in rr]; sc = [float(r['score']) for r in rr]; mf = [float(r['macro_f1']) for r in rr]; rho = [r.get('rho_neo') for r in rr]
        L.append(f"| {lab.get(c, c)} | {len(rr)} | {ms(sc)} | {ms(mf)} | {ms(val)} | {ms([v - x for v, x in zip(val, sc)])} | {energy[c]:.1f} | "
                 f"{ms(rho, 1., 4) if all(isinstance(x, float) for x in rho) else 'n/a'} | {int(rr[0]['params'])} |")
    L += ['', '## Pre-registered comparisons (%p, paired by seed; better = mean >= +0.5 and A > B in >= 2 of 3 seeds)', '',
          '| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |', '|---|---|---|---|---|---|---|---|']
    for q, x, y in Q:
        v, d, m = verdict(rows, x, y, seeds, need)
        er = energy.get(x, float('nan')) / energy.get(y, float('nan')) if energy.get(y) else float('nan')
        crit = ''
        if 'criterion 2' in q and v != 'incomplete':
            crit = 'met' if (v == 'better') or (m >= -0.5 and math.isfinite(er) and er <= 0.5) else 'not met'
        L.append(f"| {q} | {lab.get(x, x)} | {lab.get(y, y)} | {' / '.join(f'{z:+.2f}' for z in d)} | {m:+.2f} | {v} | {er:.2f} | {crit} |")
    open(os.path.join(R, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def mem_summary(a):
    """MODI 2026-10-23: membrane fusion (mixlinmem_* / mixlinmx_* / bixmem_*) against the spike versions (lin, BiX), OR and Spikformer."""
    rows = [r for r in parse(os.path.join(HERE, 'results', a.suite, 'raw.txt')) if int(r.get('crop', 0)) == 0]
    for sub in ('flif-v1-rhythm-lin-cinc', 'flif-v1-rhythm-bix-cinc'):
        rows += [r for r in parse(os.path.join(HERE, 'results', sub, 'raw.txt')) if int(r.get('crop', 0)) == 0]
    rows += [r for r in parse(os.path.join(HERE, 'results', 'flif-v1-rhythm-cinc', 'raw.txt'))
             if int(r.get('crop', 0)) == 0 and r['cond'] in ('spk_ssa', 'mix_lif_bidir', 'mix_flif_bidir', 'mix_pop_bidir')]
    ns = ('lif', 'flif', 'pop_lif', 'pop_flif')
    lab = {'spk_ssa': 'Spikformer-style SSA (T=4)'}
    for n in ns:
        lab.update({_mix_cond(n): f'Neo-mixer OR, {BIX_NAME[n]}', f'mixlin_{n}': f'linear fusion of spikes, {BIX_NAME[n]}',
                    f'mixlinmem_{n}': f'linear fusion of charges, {BIX_NAME[n]}', f'mixlinmx_{n}': f'linear fusion of charges + |H_f - H_r|, {BIX_NAME[n]}',
                    f'bix_{n}': f'BiX (spike q / k), {BIX_NAME[n]}', f'bixmem_{n}': f'BiX with charge q / k, {BIX_NAME[n]}'})
    order = ['spk_ssa'] + [c for n in ns for c in (_mix_cond(n), f'mixlin_{n}', f'mixlinmem_{n}', f'mixlinmx_{n}', f'bix_{n}', f'bixmem_{n}')]
    Q = [('M1 charges vs spikes (linear fusion)', f'mixlinmem_{n}', f'mixlin_{n}') for n in ns]
    Q += [('M2 + |H_f - H_r|', f'mixlinmx_{n}', f'mixlinmem_{n}') for n in ns]
    Q += [('M3 charges + |H_f - H_r| vs spikes', f'mixlinmx_{n}', f'mixlin_{n}') for n in ns]
    Q += [('M4 charge q / k vs spike q / k (BiX)', f'bixmem_{n}', f'bix_{n}') for n in ns]
    Q += [('M5 vs OR', f'{c}_{n}', _mix_cond(n)) for n in ns for c in ('mixlinmem', 'mixlinmx', 'bixmem')]
    Q += [('M6 vs Spikformer (criterion 2)', f'{c}_{n}', 'spk_ssa') for n in ns for c in ('mixlinmem', 'mixlinmx', 'bixmem')]
    _table_and_verdicts(a, 'Membrane-fusion Neo-mixer', rows, order, lab, Q,
                        'Best-validation weights on the test set; CinC score and macro-F1; mean ± sd over seeds (42, 7, 2026) (%). '
                        'Energy counts the real-valued fusion inputs as MAC (syops: AC only for 0/1 inputs); attention products are not counted, as before.')


def t4k_summary(a):
    """MODI 2026-10-23: NSMT T_h = 4 with the neocortex as K = 4 soma-less streams aligned with the hippocampus steps."""
    rows = [r for r in parse(os.path.join(HERE, 'results', a.suite, 'raw.txt')) if int(r.get('crop', 0)) == 0]
    rows += [r for r in parse(os.path.join(HERE, 'results', 'flif-v1-rhythm-fix-cinc', 'raw.txt')) if r['cond'] in ('nsmt_lif_t4', 'nsmt_pop_t4')]
    rows += [r for r in parse(os.path.join(HERE, 'results', 'flif-v1-rhythm-cinc', 'raw.txt'))
             if int(r.get('crop', 0)) == 0 and r['cond'] in ('spk_ssa', 'nsmt_lif', 'nsmt_flif', 'nsmt_pop')]
    lab = {'spk_ssa': 'Spikformer-style SSA (T=4)', 'nsmt_lif': 'NSMT T = N, spikeDE LIF', 'nsmt_flif': 'NSMT T = N, f-LIF',
           'nsmt_pop': 'NSMT T = N, population f-LIF', 'nsmt_lif_t4': 'NSMT T_h 4, spikeDE LIF (copied to the 4 steps)',
           'nsmt_flif_t4': 'NSMT T_h 4, f-LIF (copied)', 'nsmt_poplif_t4': 'NSMT T_h 4, population LIF with soma (copied)',
           'nsmt_pop_t4': 'NSMT T_h 4, population f-LIF with soma (copied)', 'nsmt_popk_lif_t4': 'NSMT T_h 4, 4 LIF streams = 4 steps',
           'nsmt_popk_flif_t4': 'NSMT T_h 4, 4 f-LIF streams = 4 steps'}
    order = ['spk_ssa', 'nsmt_lif', 'nsmt_flif', 'nsmt_pop', 'nsmt_lif_t4', 'nsmt_flif_t4', 'nsmt_poplif_t4', 'nsmt_pop_t4', 'nsmt_popk_lif_t4', 'nsmt_popk_flif_t4']
    Q = [('K1 streams = steps vs population with soma (copied)', 'nsmt_popk_lif_t4', 'nsmt_poplif_t4'),
         ('K1 streams = steps vs population with soma (copied)', 'nsmt_popk_flif_t4', 'nsmt_pop_t4'),
         ('K2 streams = steps vs one neuron (copied)', 'nsmt_popk_lif_t4', 'nsmt_lif_t4'),
         ('K2 streams = steps vs one neuron (copied)', 'nsmt_popk_flif_t4', 'nsmt_flif_t4'),
         ('K3 vs Spikformer (criterion 2)', 'nsmt_popk_lif_t4', 'spk_ssa'), ('K3 vs Spikformer (criterion 2)', 'nsmt_popk_flif_t4', 'spk_ssa'),
         ('K4 (report) new baselines vs T = N', 'nsmt_flif_t4', 'nsmt_flif'), ('K4 (report) population LIF vs LIF, copied', 'nsmt_poplif_t4', 'nsmt_lif_t4')]
    _table_and_verdicts(a, 'NSMT T_h = 4 with neocortex streams aligned to the hippocampus steps', rows, order, lab, Q,
                        'Best-validation weights on the test set; CinC score and macro-F1; mean ± sd over seeds (42, 7, 2026) (%).')


def _fold0(path):
    return [r for r in parse(path) if int(r.get('crop', 0)) == 0 and int(r.get('fold', 0)) == 0]


def _all_fold0():
    rows = []
    for sub in ('flif-v1-rhythm-cinc', 'flif-v1-rhythm-lin-cinc', 'flif-v1-rhythm-bix-cinc', 'flif-v1-rhythm-mem-cinc', 'flif-v1-rhythm-xor-cinc',
                'flif-v1-rhythm-popg-cinc', 'flif-v1-rhythm-t4k-cinc', 'flif-v1-rhythm-t4k2-cinc', 'flif-v1-rhythm-fix-cinc'):
        rows += _fold0(os.path.join(HERE, 'results', sub, 'raw.txt'))
    return rows


def xor_summary(a):
    """MODI 2026-10-24: spike-xor fusion mixlinxor_* -- does the |F - R| (= F xor R) term reproduce the gain of linmx without charges?"""
    rows = _all_fold0()
    ns = ('lif', 'flif', 'pop_lif', 'pop_flif')
    lab = {'spk_ssa': 'Spikformer-style SSA (T=4)'}
    for n in ns:
        lab.update({_mix_cond(n): f'Neo-mixer OR, {BIX_NAME[n]}', f'mixlin_{n}': f'linear fusion of spikes, {BIX_NAME[n]}',
                    f'mixlinmx_{n}': f'linear fusion of charges + |H_f - H_r|, {BIX_NAME[n]}', f'mixlinxor_{n}': f'linear fusion of spikes + F xor R, {BIX_NAME[n]}'})
    order = ['spk_ssa'] + [c for n in ns for c in (_mix_cond(n), f'mixlin_{n}', f'mixlinxor_{n}', f'mixlinmx_{n}')]
    Q = [('X1 + F xor R (spikes)', f'mixlinxor_{n}', f'mixlin_{n}') for n in ns]
    Q += [('X2 spike xor vs charge difference', f'mixlinxor_{n}', f'mixlinmx_{n}') for n in ns]
    Q += [('X3 vs OR', f'mixlinxor_{n}', _mix_cond(n)) for n in ns]
    Q += [('X4 vs Spikformer (criterion 2)', f'mixlinxor_{n}', 'spk_ssa') for n in ns]
    _table_and_verdicts(a, 'Spike-xor fusion Neo-mixer', rows, order, lab, Q,
                        'Best-validation weights on the test set; CinC fold 0; mean ± sd over seeds (42, 7, 2026) (%).')


def popg_summary(a):
    """MODI 2026-10-24: tau-group (soma-less) Neo-mixer with the linear fusions (spikes, charges, charges + difference, spikes + xor)."""
    rows = _all_fold0()
    ns = ('popg_lif', 'popg_flif')
    name = {'popg_lif': 'soma-less tau groups, LIF', 'popg_flif': 'soma-less tau groups, f-LIF'}
    lab = {'spk_ssa': 'Spikformer-style SSA (T=4)'}
    for n in ns:
        lab.update({f'mix_{n}_bidir': f'Neo-mixer OR, {name[n]}', f'mixlin_{n}': f'linear fusion of spikes, {name[n]}',
                    f'mixlinmem_{n}': f'linear fusion of charges, {name[n]}', f'mixlinmx_{n}': f'charges + |H_f - H_r|, {name[n]}',
                    f'mixlinxor_{n}': f'spikes + F xor R, {name[n]}'})
    order = ['spk_ssa'] + [c for n in ns for c in (f'mix_{n}_bidir', f'mixlin_{n}', f'mixlinmem_{n}', f'mixlinmx_{n}', f'mixlinxor_{n}')]
    Q = [('G0 OR -> linear fusion', f'mixlin_{n}', f'mix_{n}_bidir') for n in ns]
    Q += [('G1 charges vs spikes (M1)', f'mixlinmem_{n}', f'mixlin_{n}') for n in ns]
    Q += [('G2 + |H_f - H_r| (M2)', f'mixlinmx_{n}', f'mixlinmem_{n}') for n in ns]
    Q += [('G3 charges + difference vs spikes (M3)', f'mixlinmx_{n}', f'mixlin_{n}') for n in ns]
    Q += [('G4 + F xor R (X1)', f'mixlinxor_{n}', f'mixlin_{n}') for n in ns]
    Q += [('G5 soma-less vs soma (same fusion)', f'{c}_popg_{f}', f'{c}_pop_{f}') for f in ('lif', 'flif') for c in ('mixlin', 'mixlinmem', 'mixlinmx', 'mixlinxor')]
    Q += [('G6 vs Spikformer (criterion 2)', f'{c}_{n}', 'spk_ssa') for n in ns for c in ('mixlin', 'mixlinmem', 'mixlinmx', 'mixlinxor')]
    for f in ('lif', 'flif'):
        for c in ('mixlin', 'mixlinmem', 'mixlinmx', 'mixlinxor'):
            lab.setdefault(f'{c}_pop_{f}', f'{c}, population {f} with soma')
    _table_and_verdicts(a, 'Tau-group Neo-mixer with linear fusions', rows, order, lab, Q,
                        'Best-validation weights on the test set; CinC fold 0; mean ± sd over seeds (42, 7, 2026) (%).')


def t4k2_summary(a):
    """MODI 2026-10-24: stream-NSMT controls -- reversed stream order (srev) and the fused bidirectional streams (bimx)."""
    rows = _all_fold0()
    lab = {'spk_ssa': 'Spikformer-style SSA (T=4)', 'nsmt_lif_t4': 'T_h 4, LIF copied', 'nsmt_flif_t4': 'T_h 4, f-LIF copied',
           'nsmt_popk_lif_t4': 'T_h 4, LIF streams = steps (tau 4 -> 32)', 'nsmt_popk_flif_t4': 'T_h 4, f-LIF streams = steps (tau 4 -> 32)',
           'nsmt_popk_lif_t4_srev': 'T_h 4, LIF streams, reversed (tau 32 -> 4)', 'nsmt_popk_flif_t4_srev': 'T_h 4, f-LIF streams, reversed',
           'nsmt_popk_lif_t4_bimx': 'T_h 4, LIF streams both directions fused (charges + difference)',
           'nsmt_popk_flif_t4_bimx': 'T_h 4, f-LIF streams both directions fused'}
    order = list(lab)
    Q = [('S1 order reversed', f'nsmt_popk_{f}_t4_srev', f'nsmt_popk_{f}_t4') for f in ('lif', 'flif')]
    Q += [('S2 both directions fused', f'nsmt_popk_{f}_t4_bimx', f'nsmt_popk_{f}_t4') for f in ('lif', 'flif')]
    Q += [('S3 vs one neuron copied', f'nsmt_popk_{f}_t4_{v}', f'nsmt_{f}_t4') for f in ('lif', 'flif') for v in ('srev', 'bimx')]
    Q += [('S4 vs Spikformer (criterion 2)', f'nsmt_popk_{f}_t4_{v}', 'spk_ssa') for f in ('lif', 'flif') for v in ('srev', 'bimx')]
    _table_and_verdicts(a, 'Stream-NSMT controls', rows, order, lab, Q,
                        'Best-validation weights on the test set; CinC fold 0; mean ± sd over seeds (42, 7, 2026) (%).')


def folds_summary(a):
    """MODI 2026-10-24: charges + |H_f - H_r| LIF (mixlinmx_lif), spike linear fusion LIF (mixlin_lif) and Spikformer-style (spk_ssa) on
    CinC folds 0-4 (fold 0 from the earlier suites). Per fold: the usual verdict (3 seeds); pooled: the 15 (fold, seed) pairs, mean,
    95% t-interval (14 df), and the number of pairs with A > B."""
    from scipy import stats
    R = os.path.join(HERE, 'results', a.suite)
    rows = [r for r in parse(os.path.join(R, 'raw.txt')) if int(r.get('crop', 0)) == 0]
    rows += [r for r in parse(os.path.join(R, 'raw_recovered.txt')) if int(r.get('crop', 0)) == 0]   # recover_fold_results.py (10-25)
    keys = [(r['cond'], int(r['fold']), int(r['seed'])) for r in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit('duplicate (cond, fold, seed) rows')
    rows += [r for r in _all_fold0() if r['cond'] in ('mixlinmx_lif', 'mixlin_lif', 'spk_ssa')]
    conds = ('spk_ssa', 'mixlin_lif', 'mixlinmx_lif')
    lab = {'spk_ssa': 'Spikformer-style SSA (T=4)', 'mixlin_lif': 'linear fusion of spikes, LIF', 'mixlinmx_lif': 'charges + |H_f - H_r|, LIF'}
    seeds = (42, 7, 2026)
    L = [f'# CinC folds 0-4 `{a.suite}` — summary', '', 'Best-validation weights on each fold\'s test set; CinC score (%); fold 0 from the earlier suites.', '',
         '| condition | ' + ' | '.join(f'fold {f}' for f in range(5)) + ' | all (15) | macro-F1 all | total µJ |', '|---|' + '---|' * 8]
    def get(c, f, s, key='score'):
        v = [float(r[key]) for r in rows if r['cond'] == c and int(r.get('fold', 0)) == f and int(r['seed']) == s]
        return v[0] if len(v) == 1 else float('nan')
    for c in conds:
        cells = []
        for f in range(5):
            v = [get(c, f, s) for s in seeds]
            cells.append(f'{100 * np.mean(v):.2f}' if np.all(np.isfinite(v)) else 'n/a')
        allv = [get(c, f, s) for f in range(5) for s in seeds]; allm = [get(c, f, s, 'macro_f1') for f in range(5) for s in seeds]
        en = [float(r['total_fft_uJ']) for r in rows if r['cond'] == c]
        L.append(f"| {lab[c]} | " + ' | '.join(cells) + f" | {100 * np.mean(allv):.2f} ± {100 * np.std(allv, ddof=1):.2f} | {100 * np.mean(allm):.2f} | {np.mean(en):.1f} |")
    L += ['', '## Comparisons (%p)', '', '| question | A | B | per fold (mean of 3 seeds) and verdict | pooled mean [95% CI] | A > B pairs |', '|---|---|---|---|---|---|']
    for q, x, y in (('F1 charges + difference vs Spikformer', 'mixlinmx_lif', 'spk_ssa'), ('F2 charges + difference vs spike linear fusion', 'mixlinmx_lif', 'mixlin_lif'),
                    ('F3 spike linear fusion vs Spikformer', 'mixlin_lif', 'spk_ssa')):
        per = []
        for f in range(5):
            fr = [r for r in rows if int(r.get('fold', 0)) == f]
            v, d, m = verdict(fr, x, y, seeds, 2)
            per.append(f'f{f} {m:+.2f} {v}')
        d = np.array([100 * (get(x, f, s) - get(y, f, s)) for f in range(5) for s in seeds])
        if np.all(np.isfinite(d)):
            half = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
            pooled = f'{d.mean():+.2f} [{d.mean() - half:+.2f}, {d.mean() + half:+.2f}]'; wins = f'{int((d > 0).sum())}/15'
        else:
            pooled, wins = 'incomplete', 'n/a'
        L.append(f'| {q} | {lab[x]} | {lab[y]} | ' + '; '.join(per) + f' | {pooled} | {wins} |')
    open(os.path.join(R, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--decide', action='store_true')
    ap.add_argument('--lr_suite', default='flif-v1-rhythm-lr')
    ap.add_argument('--crop_suite', default='flif-v1-rhythm-crop')
    ap.add_argument('--suite', default=None)
    ap.add_argument('--aux', action='store_true')
    ap.add_argument('--fix', action='store_true')
    ap.add_argument('--bix', action='store_true')
    ap.add_argument('--lin', action='store_true')
    ap.add_argument('--mem', action='store_true')
    ap.add_argument('--t4k', action='store_true')
    ap.add_argument('--xor', action='store_true')
    ap.add_argument('--popg', action='store_true')
    ap.add_argument('--t4k2', action='store_true')
    ap.add_argument('--folds', action='store_true')
    a = ap.parse_args()
    if a.xor:
        xor_summary(a)
    elif a.popg:
        popg_summary(a)
    elif a.t4k2:
        t4k2_summary(a)
    elif a.folds:
        folds_summary(a)
    elif a.mem:
        mem_summary(a)
    elif a.t4k:
        t4k_summary(a)
    else:
        decide(a) if a.decide else (aux_summary(a) if a.aux else (fix_summary(a) if a.fix else (bix_summary(a) if a.bix else (lin_summary(a) if a.lin else summary(a)))))
