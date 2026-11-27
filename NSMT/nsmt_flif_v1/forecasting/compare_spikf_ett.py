"""MODI 2026-11-27 (user request 11-27 "SpikF와도 비교해줘"; adoption bar of 11-27): report-only comparison of the current ETT base
(suite flif-v1-ett-r1: query pop LIF / pop f-LIF with the Neocortex membrane shortcut, all SpikeDE, R1 settings, bias off; and
Spikformer(DE)) against SpikF (ICML 2025) and the other SNN rows of SpikF Table 12, on the 4 ETT cells of this workstream
(ETTh2 / ETTm2 x H96 / H720, input 96).

References (read from the f_lif_pop_v3 branch, not retyped; the branch name / commit are printed):
  - published rows: NSMT/cinc_mixer_v1/results/ett-full16-20261124/refs16.json (SpikF Table 12, transcription checked there, 0 mismatches)
  - reruns of the released SpikF code, 3 seeds (42 / 7 / 2026), with energy (syops count, 0.9 / 4.6 pJ, as SpikF logs it):
    NSMT/spikf_ref/forecasting/results/spikf-act-ett/summary.json (SpikF-GELU = released; SpikF-LIF = LIF head activation)
Ours: mean over seeds 42 / 7 / 2026 of test MSE / MAE (best-validation weights) and syn_energy_uJ. Fail closed: every cell x seed once,
finite, no SKIP, one sha.
Adoption bar (user 11-27): 4-cell mean MSE AND MAE each at least 0.001 lower than the reference; energy at most 1.5x (only the reruns
carry an energy).
usage (forecasting folder): python compare_spikf_ett.py [--branch origin/exp/f-lif-pop-v3]
"""
import argparse
import json
import math
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = [('ETTh2', '96'), ('ETTh2', '720'), ('ETTm2', '96'), ('ETTm2', '720')]
OURS = {'nsmt4q_pop_lif_msn_sde': 'ours, query pop LIF', 'nsmt4q_pop_flif_msn_sde': 'ours, query pop f-LIF',
        'spkall_lif': 'Spikformer(DE)'}


def git_json(branch, path):
    return json.loads(subprocess.check_output(['git', 'show', f'{branch}:{path}'], cwd=HERE))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--branch', default='origin/exp/f-lif-pop-v3'); a = ap.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', '--short', a.branch], cwd=HERE, text=True).strip()
    refs = git_json(a.branch, 'NSMT/cinc_mixer_v1/results/ett-full16-20261124/refs16.json')['models']
    rer = git_json(a.branch, 'NSMT/spikf_ref/forecasting/results/spikf-act-ett/summary.json')
    R = {f'{m} (paper)': [(*refs[m]['rows'][f'{d} {h}'], None) for d, h in CELLS] for m in ('SpikF', 'SpikeRNN', 'iSpikformer')}
    for act, name in (('gelu', 'SpikF rerun, released (3 seeds)'), ('lif', 'SpikF-LIF rerun (3 seeds)')):
        R[name] = [(rer['cells'][f'{d} {h} {act}']['mse'], rer['cells'][f'{d} {h} {act}']['mae'], rer['cells'][f'{d} {h} {act}']['energy_uJ'])
                   for d, h in CELLS]
    lines = open(os.path.join(HERE, 'results', 'flif-v1-ett-r1', 'raw.txt')).read().splitlines()
    if any(l.startswith('SKIP') for l in lines):
        raise SystemExit('STOP: SKIP line in flif-v1-ett-r1')
    rows = [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in lines if l.startswith('RESULT')]
    if len({r['sha'] for r in rows}) != 1:
        raise SystemExit('STOP: more than one code version')
    O = {}
    for c in OURS:
        per = []
        for d, h in CELLS:
            rr = [r for r in rows if r['cond'] == c and r['ds'] == d and r['pl'] == h]
            if sorted(r['seed'] for r in rr) != ['2026', '42', '7']:
                raise SystemExit(f'STOP: {c} {d} H{h} seeds {[r["seed"] for r in rr]}')
            v = [[float(r[k]) for k in ('mse', 'mae', 'syn_energy_uJ')] for r in rr]
            if not all(math.isfinite(x) for t in v for x in t):
                raise SystemExit(f'STOP: {c} {d} H{h} not finite')
            per.append(tuple(sum(t[i] for t in v) / 3 for i in range(3)))
        O[c] = per
    mean = lambda per, i: sum(p[i] for p in per) / 4
    out = ['# ETT (ETTh2 / ETTm2 x H96 / H720, input 96): current base vs SpikF and the SNN rows of SpikF Table 12\n',
           f'References from `{a.branch}` (commit {commit}); SpikF reruns: code commit {rer["sha"]}, {rer["n_rows"]} runs. Ours: flif-v1-ett-r1, '
           f'sha {rows[0]["sha"]}. Values: mean of seeds; 4-cell mean in the last columns. Energy = syops count (µJ).\n',
           '| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | 4-cell MSE | 4-cell MAE | energy µJ |', '|---|---|---|---|---|---|---|---|']
    for name, per in list(R.items()) + [(OURS[c], O[c]) for c in OURS]:
        e = f'{mean(per, 2):.2f}' if per[0][2] is not None else '—'
        out.append(f'| {name} | ' + ' | '.join(f'{m:.4f} / {x:.4f}' for m, x, _ in per) + f' | {mean(per, 0):.4f} | {mean(per, 1):.4f} | {e} |')
    out += ['', '## Adoption bar (reference − ours ≥ 0.001 for both 4-cell MSE and MAE; energy ≤ 1.5x)', '',
            '| ours | reference | margin MSE | margin MAE | energy × | cells passing both | adoption |', '|---|---|---|---|---|---|---|']
    dec = {}
    for c in OURS:
        for name, per in R.items():
            dm, da = mean(per, 0) - mean(O[c], 0), mean(per, 1) - mean(O[c], 1)
            ex = mean(O[c], 2) / mean(per, 2) if per[0][2] is not None else None
            cells = sum(1 for o, r in zip(O[c], per) if r[0] - o[0] >= 0.001 and r[1] - o[1] >= 0.001)
            acc = dm >= 0.001 and da >= 0.001
            v = 'pass' if acc and (ex is None or ex <= 1.5) else ('energy over' if acc else 'not passed')
            dec[f'{c} vs {name}'] = dict(margin_mse=dm, margin_mae=da, energy_x=ex, cells=cells, adoption=v)
            out.append(f'| {OURS[c]} | {name} | {dm:+.4f} | {da:+.4f} | {"—" if ex is None else f"{ex:.3f}"} | {cells}/4 | **{v}** |')
    txt = '\n'.join(out) + '\n'
    print(txt)
    od = os.path.join(HERE, 'results', 'flif-v1-ett-vs-spikf'); os.makedirs(od, exist_ok=True)
    open(os.path.join(od, 'summary.md'), 'w').write(txt)
    json.dump(dict(branch=a.branch, commit=commit, decisions=dec), open(os.path.join(od, 'decision.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
