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
             'mix_pop_bidir_psi2': 'Neo-mixer bidir population psi 2'}
FIX_ORDER = ('spk_ssa', 'spk_ssa_nope', 'nsmt_lif', 'nsmt_lif_pe', 'nsmt_lif_t4', 'nsmt_pop', 'nsmt_pop_t4', 'nsmt_pop_psi1.5', 'nsmt_pop_psi2',
             'mix_lif_bidir', 'mix_lif_bidir_pe', 'mix_sjlif_bidir', 'mix_lif_rec_bidir', 'mix_pop_bidir', 'mix_pop_bidir_pt16',
             'mix_pop_bidir_psi1.5', 'mix_pop_bidir_psi2')


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


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--decide', action='store_true')
    ap.add_argument('--lr_suite', default='flif-v1-rhythm-lr')
    ap.add_argument('--crop_suite', default='flif-v1-rhythm-crop')
    ap.add_argument('--suite', default=None)
    ap.add_argument('--aux', action='store_true')
    ap.add_argument('--fix', action='store_true')
    a = ap.parse_args()
    decide(a) if a.decide else (aux_summary(a) if a.aux else (fix_summary(a) if a.fix else summary(a)))
