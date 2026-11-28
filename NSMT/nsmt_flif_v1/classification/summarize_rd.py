"""MODI 2026-11-27 (approved plan 1-snuggly-galaxy.md; user decisions 11-27 "1. 둘다", "2. 일단은 빼고 성능 변화만"): pre-registered
summary of the Neocortex readout x accumulation-aux runs on MIT-BIH, Sleep-EDF and ETT (docs/MODI_PROJECT_LOG.md 2026-11-28
pre-registration).

Models (18 per task): Neocortex neuron n in {lif, flif, pop_lif} x readout r in {spk (spikes), mem ((가): membrane -> q_linear -> BN ->
query LIF), memd ((나): membrane -> BN = the real-valued queries)} x accumulation aux a in {off, on}. Spikformer(DE) is the comparison model.
Rows (fail closed: every needed row once, settings below, finite values, no SKIP, one sha per suite, no unexpected row in a new suite):
  MIT-BIH (seeds 42 / 7 / 2026): new flif-v1-mitbih-rd (rd_<n>_<r>[_acc], 51 rows; epochs 30, lr 0.01, val 0, wdx 1, pe 0, aux 0 / 0.1
          matching _acc); pop_lif spk off = nsmt4q_pop_lif_msn_sde and Spikformer(DE) = spkall_lif from flif-v1-mitbih-r1. Metric: f1
          (macro-F1, %).
  Sleep-EDF (lr 0.003, folds 0-4, seed 42): new flif-v1-sleep5 (8 fill-in models, 40 rows); the other 10 models and spkall_lif from
          flif-v1-sleep-lr / -sleep / -sleep2 / -sleep3 / -sleep4 (select_sleep.check: epochs 30, batch 64). Metric: macro_f1 (%).
  ETT (ETTh2 / ETTm2 x H96 / H720, seeds 42 / 7 / 2026): new flif-v1-ett-rd (204 rows; seq_len 96, wdx 1, vpe 4, pe 0, runner-default
          lr / wd / batch); pop_lif spk off and spkall_lif from flif-v1-ett-r1. Metrics: test MSE, MAE.
Clear-difference verdicts (as before): ETT 12 pairs (cell x seed), R = mean relative MSE change %, better = R <= -1 and >= 8/12 pairs
lower, worse = R >= +1 and >= 8/12 higher; MIT-BIH 3 seed pairs, macro-F1 %p, better = mean >= +0.5 and >= 2/3 positive (worse mirrored);
Sleep 5 fold pairs, better = mean >= +1.0 and >= 4/5 positive (worse mirrored).
Adoption bar (user 11-27): ETT 4-cell mean MSE AND MAE each >= 0.001 lower than the comparison; classification macro-F1 mean >= +0.1 %p;
and syops energy (syn_energy_uJ, mean over runs) <= 1.5x the comparison's.
Tests: E = each variant - the same neuron's spk-off model (5 per neuron) + factor contrasts (memd - mem at the same aux; aux on - off at the
same readout); S = each of the 18 models - Spikformer(DE) (verdict + adoption); F (ETT only) = adoption vs SpikF paper, SpikF rerun and
SpikF-LIF rerun (read from the f_lif_pop_v3 branch as compare_spikf_ett.py); N = lif - pop_lif and flif - pop_lif at every readout x aux.
--task all: adoption candidates = models passing S adoption on all three tasks, ordered by the mean energy ratio over the tasks.
usage (classification folder): python summarize_rd.py --task mitbih|sleep|ett|all [--root <nsmt_flif_v1 copy, for tests>]
"""
import argparse
import json
import math
import os
import subprocess

import numpy as np

import select_sleep as SS

HERE = os.path.dirname(os.path.abspath(__file__))
NEU, RDS, AUX = ('lif', 'flif', 'pop_lif'), ('spk', 'mem', 'memd'), (0, 1)
MODELS = [(n, r, a) for n in NEU for r in RDS for a in AUX]
BASE_REF = ('pop_lif', 'spk', 0)
SEEDS = ('42', '7', '2026')
CELLS = (('ETTh2', '96'), ('ETTh2', '720'), ('ETTm2', '96'), ('ETTm2', '720'))
GAIN = {'ett': 0.001, 'mitbih': 0.1, 'sleep': 0.1}
EMAX = 1.5
SLEEP_REF = ('flif-v1-sleep-lr', 'flif-v1-sleep', 'flif-v1-sleep2', 'flif-v1-sleep3', 'flif-v1-sleep4')
FILL = {('lif', 'memd', 0), ('lif', 'memd', 1), ('flif', 'memd', 0), ('flif', 'memd', 1), ('flif', 'spk', 1), ('flif', 'mem', 1),
        ('pop_lif', 'memd', 0), ('pop_lif', 'memd', 1)}
SUITE = {'mitbih': ('classification', 'flif-v1-mitbih-rd'), 'sleep': ('classification', 'flif-v1-sleep5'),
         'ett': ('forecasting', 'flif-v1-ett-rd')}


def name(m):
    return f'{m[0]} {m[1]}{"+acc" if m[2] else ""}'


def cond_rd(m):
    return f'rd_{m[0]}_{m[1]}{"_acc" if m[2] else ""}'


def cond_sleep(m):
    return f'q_{m[0]}' + ('' if m[1] == 'spk' else f'_{m[1]}') + ('_acc' if m[2] else '')


def kv(line):
    return dict(t.split('=', 1) for t in line.split()[1:] if '=' in t)


def read(path):
    lines = open(path).read().splitlines()
    sk = [l for l in lines if l.startswith('SKIP')]
    if sk:
        raise SystemExit(f'STOP: {len(sk)} SKIP lines in {path}: {sk[0][:160]}')
    return [kv(l) for l in lines if l.startswith('RESULT')]


def finite(r, keys, key):
    for k in keys:
        if k not in r or not math.isfinite(float(r[k])):
            raise SystemExit(f'STOP: {key} {k} = {r.get(k)}')


class Task:
    """values(m, field) -> array over the pairs (the same pair order for every model); sf(field) the same for Spikformer(DE)."""
    def __init__(self, kind, T, cond_of, pair_keys, shas):
        self.kind, self.T, self.cond_of, self.pairs, self.shas = kind, T, cond_of, pair_keys, shas

    def values(self, m, field):
        c = 'spkall_lif' if m == 'SF' else self.cond_of(m)
        return np.array([float(self.T[(c,) + p][field]) for p in self.pairs], dtype=np.float64)


def load_mitbih(root):
    new = read(os.path.join(root, 'classification', 'results', 'flif-v1-mitbih-rd', 'raw.txt'))
    r1 = read(os.path.join(root, 'classification', 'results', 'flif-v1-mitbih-r1', 'raw.txt'))
    cond_of = lambda m: 'nsmt4q_pop_lif_msn_sde' if m == BASE_REF else cond_rd(m)
    want_new = {(cond_rd(m), s) for m in MODELS if m != BASE_REF for s in SEEDS}
    want_ref = {(c, s) for c in ('nsmt4q_pop_lif_msn_sde', 'spkall_lif') for s in SEEDS}
    T, shas = {}, {}

    def check(r, key):
        bad = {k: r.get(k) for k, v in (('epochs', '30'), ('lr', '0.01'), ('val', '0'), ('wdx', '1'), ('pe', '0'), ('dataset', 'MITBIH'))
               if r.get(k) != v}
        if r['cond'].startswith('rd_') and r.get('aux') != ('0.1' if r['cond'].endswith('_acc') else '0'):
            bad['aux'] = r.get('aux')
        if bad:
            raise SystemExit(f'STOP: {key} settings {bad}')
        finite(r, ('f1', 'overall_acc', 'syn_energy_uJ', 'firing_rate'), key)
    for suite, rows, want, strict in (('flif-v1-mitbih-rd', new, want_new, True), ('flif-v1-mitbih-r1', r1, want_ref, False)):
        sh = set()
        for r in rows:
            key = (r['cond'], r['seed'])
            if key not in want:
                if strict:
                    raise SystemExit(f'STOP: unexpected row {key} in {suite}')
                continue
            if key in T:
                raise SystemExit(f'STOP: duplicate {key} ({suite})')
            check(r, key); T[key] = r; sh.add(r['sha'])
        if len(sh) > 1:
            raise SystemExit(f'STOP: {suite} code versions {sorted(sh)}')
        shas[suite] = sorted(sh)
    miss = sorted((want_new | want_ref) - set(T))
    if miss:
        raise SystemExit(f'STOP: missing {miss[:4]} ({len(miss)})')
    return Task('mitbih', T, cond_of, [(s,) for s in SEEDS], shas)


def load_sleep(root):
    SS.HERE = os.path.join(root, 'classification')
    lr = '0.003'
    want_new = {(cond_sleep(m), lr, f) for m in FILL for f in range(5)}
    want_ref = {(cond_sleep(m), lr, f) for m in MODELS if m not in FILL for f in range(5)} | {('spkall_lif', lr, f) for f in range(5)}
    T, shas = {}, {}
    for suite in SLEEP_REF + ('flif-v1-sleep5',):
        strict = suite == 'flif-v1-sleep5'
        want = want_new if strict else want_ref
        sh = set()
        for r in SS.rows(suite):
            key = (r['cond'], r['lr'], int(r['fold']))
            if key not in want:
                if strict:
                    raise SystemExit(f'STOP: unexpected row {key} in {suite}')
                continue
            if key in T:
                raise SystemExit(f'STOP: duplicate {key} ({suite})')
            SS.check(r, key); finite(r, ('syops_firing_rate',), key); T[key] = r; sh.add(r['sha'])
        if len(sh) > 1:
            raise SystemExit(f'STOP: {suite} code versions {sorted(sh)}')
        shas[suite] = sorted(sh)
    miss = sorted((want_new | want_ref) - set(T))
    if miss:
        raise SystemExit(f'STOP: missing {miss[:4]} ({len(miss)})')
    return Task('sleep', T, cond_sleep, [(lr, f) for f in range(5)], shas)


def load_ett(root):
    new = read(os.path.join(root, 'forecasting', 'results', 'flif-v1-ett-rd', 'raw.txt'))
    r1 = read(os.path.join(root, 'forecasting', 'results', 'flif-v1-ett-r1', 'raw.txt'))
    cond_of = lambda m: 'nsmt4q_pop_lif_msn_sde' if m == BASE_REF else cond_rd(m)
    want_new = {(cond_rd(m), d, p, s) for m in MODELS if m != BASE_REF for d, p in CELLS for s in SEEDS}
    want_ref = {(c, d, p, s) for c in ('nsmt4q_pop_lif_msn_sde', 'spkall_lif') for d, p in CELLS for s in SEEDS}
    T, shas = {}, {}

    def check(r, key):
        bad = {k: r.get(k) for k, v in (('seq_len', '96'), ('wdx', '1'), ('vpe', '4'), ('pe', '0')) if r.get(k) != v}
        bad.update({k: r[k] for k in ('lr', 'wd', 'drop', 'bs', 'wu', 'auxs') if k in r})
        if bad:
            raise SystemExit(f'STOP: {key} settings {bad}')
        finite(r, ('mse', 'mae', 'syn_energy_uJ', 'syops_firing_rate'), key)
    for suite, rows, want, strict in (('flif-v1-ett-rd', new, want_new, True), ('flif-v1-ett-r1', r1, want_ref, False)):
        sh = set()
        for r in rows:
            key = (r['cond'], r['ds'], r['pl'], r['seed'])
            if key not in want:
                if strict:
                    raise SystemExit(f'STOP: unexpected row {key} in {suite}')
                continue
            if key in T:
                raise SystemExit(f'STOP: duplicate {key} ({suite})')
            check(r, key); T[key] = r; sh.add(r['sha'])
        if len(sh) > 1:
            raise SystemExit(f'STOP: {suite} code versions {sorted(sh)}')
        shas[suite] = sorted(sh)
    miss = sorted((want_new | want_ref) - set(T))
    if miss:
        raise SystemExit(f'STOP: missing {miss[:4]} ({len(miss)})')
    return Task('ett', T, cond_of, [(d, p, s) for d, p in CELLS for s in SEEDS], shas)


def spikf_refs(branch='origin/exp/f-lif-pop-v3'):
    def git_json(path):
        return json.loads(subprocess.check_output(['git', 'show', f'{branch}:{path}'], cwd=HERE))
    refs = git_json('NSMT/cinc_mixer_v1/results/ett-full16-20261124/refs16.json')['models']
    rer = git_json('NSMT/spikf_ref/forecasting/results/spikf-act-ett/summary.json')
    R = {'SpikF (paper)': [(*refs['SpikF']['rows'][f'{d} {h}'], None) for d, h in CELLS]}
    for act, nm in (('gelu', 'SpikF rerun'), ('lif', 'SpikF-LIF rerun')):
        R[nm] = [tuple(rer['cells'][f'{d} {h} {act}'][k] for k in ('mse', 'mae', 'energy_uJ')) for d, h in CELLS]
    return R


def clear(task, A, B, pos='better', neg='worse'):
    """A, B: per-pair metric arrays (ETT: MSE, lower is better; classification: macro-F1 in %)."""
    if task == 'ett':
        rel = (A / B - 1.) * 100.; R = float(rel.mean()); lo, hi = int((A < B).sum()), int((A > B).sum())
        v = pos if (R <= -1. and lo >= 8) else neg if (R >= 1. and hi >= 8) else 'no clear difference'
        return dict(stat=R, unit='% (R, MSE)', pos=lo, n=len(A), verdict=v)
    thr, k = (0.5, 2) if task == 'mitbih' else (1.0, 4)
    d = A - B; m = float(d.mean())
    v = pos if (m >= thr and (d > 0).sum() >= k) else neg if (m <= -thr and (d < 0).sum() >= k) else 'no clear difference'
    return dict(stat=m, unit='%p', pos=int((d > 0).sum()), n=len(A), verdict=v)


def adoption(task, t, m, ref='SF'):
    e = t.values(m, 'syn_energy_uJ').mean() / t.values(ref, 'syn_energy_uJ').mean()
    if task == 'ett':
        dm = t.values(ref, 'mse').mean() - t.values(m, 'mse').mean(); da = t.values(ref, 'mae').mean() - t.values(m, 'mae').mean()
        acc = dm >= GAIN['ett'] - 1e-12 and da >= GAIN['ett'] - 1e-12
        res = dict(margin_mse=float(dm), margin_mae=float(da))
    else:
        d = t.values(m, metric(task)).mean() * 100 - t.values(ref, metric(task)).mean() * 100
        acc = d >= GAIN[task] - 1e-9
        res = dict(gain=float(d))
    res.update(energy_x=float(e), adoption='pass' if acc and e <= EMAX else ('energy over' if acc else 'below'))
    return res


def metric(task):
    return {'mitbih': 'f1', 'sleep': 'macro_f1', 'ett': 'mse'}[task]


def summarize(task, t):
    met = metric(task)
    val = (lambda m: t.values(m, met)) if task == 'ett' else (lambda m: t.values(m, met) * 100)
    fr = 'firing_rate' if task == 'mitbih' else 'syops_firing_rate'
    e0 = t.values('SF', 'syn_energy_uJ').mean()
    out = [f'# {task}: Neocortex readout x accumulation aux (3 neurons) vs Spikformer(DE)\n',
           f'Pre-registered: docs/MODI_PROJECT_LOG.md 2026-11-28. Code versions: {t.shas}. Pairs: {len(t.pairs)}.\n']
    if task == 'ett':
        out += ['| model | 4-cell MSE | 4-cell MAE | energy µJ | × Spikformer(DE) | firing rate |', '|---|---|---|---|---|---|']
        rows = [('Spikformer(DE)', 'SF')] + [(name(m), m) for m in MODELS]
        for nm, m in rows:
            out.append(f'| {nm} | {t.values(m, "mse").mean():.4f} | {t.values(m, "mae").mean():.4f} | {t.values(m, "syn_energy_uJ").mean():.2f} | '
                       f'{t.values(m, "syn_energy_uJ").mean() / e0:.2f} | {t.values(m, fr).mean():.4f} |')
    else:
        acc = 'overall_acc' if task == 'mitbih' else 'acc'
        out += ['| model | macro-F1 | accuracy | energy µJ | × Spikformer(DE) | firing rate |', '|---|---|---|---|---|---|']
        for nm, m in [('Spikformer(DE)', 'SF')] + [(name(m), m) for m in MODELS]:
            v = val(m)
            out.append(f'| {nm} | {v.mean():.2f} ± {v.std():.2f} | {100 * t.values(m, acc).mean():.2f} | {t.values(m, "syn_energy_uJ").mean():.2f} | '
                       f'{t.values(m, "syn_energy_uJ").mean() / e0:.2f} | {t.values(m, fr).mean():.4f} |')
    dec = {'E': {}, 'Efactor': {}, 'S': {}, 'N': {}, 'shas': t.shas}
    out += ['', '| test | statistic | positive / lower pairs | verdict |', '|---|---|---|---|']

    def line(group, key, A, B, pos='better', neg='worse'):
        r = clear(task, val(A) if A != 'SF' else val('SF'), val(B) if B != 'SF' else val('SF'), pos, neg)
        dec[group][key] = r
        out.append(f'| {group} {key} | {r["stat"]:+.2f} {r["unit"]} | {r["pos"]}/{r["n"]} | **{r["verdict"]}** |')
    for n in NEU:
        base = (n, 'spk', 0)
        for m in MODELS:
            if m[0] == n and m != base:
                line('E', f'{name(m)} − {name(base)}', m, base)
        for a in AUX:
            line('Efactor', f'{name((n, "memd", a))} − {name((n, "mem", a))}', (n, 'memd', a), (n, 'mem', a))
        for r in RDS:
            line('Efactor', f'{name((n, r, 1))} − {name((n, r, 0))} (aux on − off)', (n, r, 1), (n, r, 0))
    for m in MODELS:
        line('S', f'{name(m)} − Spikformer(DE)', m, 'SF')
        dec['S'][f'{name(m)} − Spikformer(DE)']['adoption'] = adoption(task, t, m)
    for r in RDS:
        for a in AUX:
            for n in ('lif', 'flif'):
                line('N', f'{name((n, r, a))} − {name(("pop_lif", r, a))}', (n, r, a), ('pop_lif', r, a),
                     f'{"f-LIF" if n == "flif" else "LIF"} better', 'pop LIF better')
    out += ['', '## Adoption bar vs Spikformer(DE) (user 11-27)', '',
            '| model | ' + ('margin MSE | margin MAE' if task == 'ett' else 'gain %p') + ' | energy × | adoption |',
            '|---|' + ('---|---|' if task == 'ett' else '---|') + '---|---|']
    for m in MODELS:
        ad = dec['S'][f'{name(m)} − Spikformer(DE)']['adoption']
        acc_cols = f'{ad["margin_mse"]:+.4f} | {ad["margin_mae"]:+.4f}' if task == 'ett' else f'{ad["gain"]:+.2f}'
        out.append(f'| {name(m)} | {acc_cols} | {ad["energy_x"]:.2f} | **{ad["adoption"]}** |')
    if task == 'ett':
        R = spikf_refs()
        dec['F'] = {}
        out += ['', '## Adoption bar vs SpikF (ETT; reference − ours ≥ 0.001 for both MSE and MAE; energy ≤ 1.5x where the reference has one)', '',
                '| model | reference | margin MSE | margin MAE | energy × | adoption |', '|---|---|---|---|---|---|']
        for m in MODELS:
            om, oa, oe = t.values(m, 'mse').mean(), t.values(m, 'mae').mean(), t.values(m, 'syn_energy_uJ').mean()
            for nm, per in R.items():
                rm, ra = np.mean([p[0] for p in per]), np.mean([p[1] for p in per])
                ex = oe / np.mean([p[2] for p in per]) if per[0][2] is not None else None
                accp = rm - om >= 0.001 - 1e-12 and ra - oa >= 0.001 - 1e-12
                v = 'pass' if accp and (ex is None or ex <= EMAX) else ('energy over' if accp else 'not passed')
                dec['F'][f'{name(m)} vs {nm}'] = dict(margin_mse=float(rm - om), margin_mae=float(ra - oa), energy_x=ex, adoption=v)
                out.append(f'| {name(m)} | {nm} | {rm - om:+.4f} | {ra - oa:+.4f} | {"—" if ex is None else f"{ex:.3f}"} | **{v}** |')
    return '\n'.join(out) + '\n', dec


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--task', required=True, choices=['mitbih', 'sleep', 'ett', 'all'])
    ap.add_argument('--root', default=os.path.dirname(HERE)); a = ap.parse_args(argv)
    load = {'mitbih': load_mitbih, 'sleep': load_sleep, 'ett': load_ett}
    tasks = ['mitbih', 'sleep', 'ett'] if a.task == 'all' else [a.task]
    decs, texts = {}, {}
    for task in tasks:
        texts[task], decs[task] = summarize(task, load[task](a.root))
        if a.task != 'all':
            sub, suite = SUITE[task]
            od = os.path.join(a.root, sub, 'results', suite)
            open(os.path.join(od, 'summary.md'), 'w').write(texts[task])
            json.dump(decs[task], open(os.path.join(od, 'decision.json'), 'w'), indent=1)
            print(texts[task])
            return decs
    out = ['# Neocortex readout x accumulation aux: three tasks (adoption bar vs Spikformer(DE), user 11-27)\n',
           '| model | MIT-BIH | Sleep-EDF | ETT | mean energy × | all three pass | ETT vs SpikF (paper / rerun / SpikF-LIF) |', '|---|---|---|---|---|---|---|']
    cand = []
    for m in MODELS:
        ads = [decs[k]['S'][f'{name(m)} − Spikformer(DE)']['adoption'] for k in ('mitbih', 'sleep', 'ett')]
        ex = float(np.mean([x['energy_x'] for x in ads])); allp = all(x['adoption'] == 'pass' for x in ads)
        f = ' / '.join(decs['ett']['F'][f'{name(m)} vs {nm}']['adoption'] for nm in ('SpikF (paper)', 'SpikF rerun', 'SpikF-LIF rerun'))
        cell = lambda x, k: (f'{x["gain"]:+.2f} %p' if k != 'ett' else f'{x["margin_mse"]:+.4f} / {x["margin_mae"]:+.4f}') + f' ({x["energy_x"]:.2f}×) {x["adoption"]}'
        out.append(f'| {name(m)} | {cell(ads[0], "mitbih")} | {cell(ads[1], "sleep")} | {cell(ads[2], "ett")} | {ex:.2f} | {"yes" if allp else "no"} | {f} |')
        if allp:
            cand.append((ex, name(m)))
    cand.sort()
    rec = ('candidates (lowest mean energy first): ' + ', '.join(f'{nm} ({ex:.2f}×)' for ex, nm in cand)) if cand else \
        'no model passes the adoption bar on all three tasks; per-task passes: ' + '; '.join(
            f'{k}: ' + ', '.join(name(m) for m in MODELS if decs[k]['S'][f'{name(m)} − Spikformer(DE)']['adoption']['adoption'] == 'pass')
            for k in ('mitbih', 'sleep', 'ett'))
    out += ['', f'**Reading (pre-registered):** {rec}.']
    txt = '\n'.join(out) + '\n'
    od = os.path.join(a.root, 'classification', 'results', 'flif-v1-rd-summary'); os.makedirs(od, exist_ok=True)
    open(os.path.join(od, 'summary.md'), 'w').write(txt)
    json.dump(dict(candidates=[nm for _, nm in cand], reading=rec, tasks=decs), open(os.path.join(od, 'decision.json'), 'w'), indent=1)
    print(txt)
    return dict(candidates=[nm for _, nm in cand], reading=rec, tasks=decs)


if __name__ == '__main__':
    main()
