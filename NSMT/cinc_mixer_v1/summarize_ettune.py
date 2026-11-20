"""ETT tuning of the LIF SpkMixer (2026-11-20, exp/f-lif-pop-v3; PROJECT_LOG ETT tuning pre-registration; user decisions 11-19: plan C +
patch length, the structure adopted by the mixer-only population ETT part = LIF SpkMixer, SF-de not tuned, selection on validation only).

Stepwise, one knob per stage, the winners of the earlier stages carried over per dataset (scripts/make_queue_ettune.sh):
    patch  PATCH  {4, 8*, 16}        dmodel EMB {32, 64*, 128, 256}    mlp MLPR {0.5, 1, 2, 4} (* ETTh2 0.5, ETTm2 1)
    blocks BLOCKS {1*, 2, 3}         wd     WD  {0, 0.01, 0.06*, 0.1}  (* = Table 13 value)
Fixed: muxlin + IAND, 1 mixer layer per direction, lr ETTh2 5e-4 / ETTm2 1e-4, input 96, patience 10, tau 2, T_h 4, batch 64, L1 loss.
Rows: the new runs in results/flif-v1-ett-spkmux-ettune/raw.txt (RESULT fields patch= emb= mlpr= blocks= wd=, tag suffixes _pt _e _mlp
_blk _wd); the all-default configuration is the tuned LIF SpkMixer of 11-01 (results/flif-v1-ett-spkmux-tune, tag
spkmx_lif_muxlin_iand_lr<lr>_L1), never re-run.

Selection (--stage patch|dmodel|mlp|blocks|wd; validation only, the 11-01 rule): per dataset,
    score(v) = mean over H96 / H720 of [seed-mean best validation L1 at v] / [seed-mean best validation L1 at the current value]
the lowest score wins; ties (< 1e-12) keep the current value; between two other values with the same score the earlier one in the
candidate list. Writes selected_<stage>.json (refuses to overwrite a different selection) and selection_<stage>.md (no test column).
Final (--stage final, after all five): the selected configuration's runs (the final rows are the tuning runs, as on 11-01):
    test table (tuned vs the untuned LIF SpkMixer vs SF-de; also the configuration after each stage, report),
    comparisons (ETT rule: r = 100 (A - B) / B per seed on test MSE, R = 4-cell mean, better = R <= -1 % and >= 8 of 12 pairs lower,
    worse = mirror) tuned vs untuned and tuned vs SF-de, energy A/B under the syops rule (logged total_fft_uJ; LIF / SF-de rows have
    no population neuron) and the old rule (energy_recount.csv total_fft_uJ_corr, report),
    the user's pass criteria (results/ett-pass-20261119/pass_ett.py judge) -> summary_final.md, final.json.
Fail closed (exit 2, nothing written): a SKIP line, a duplicate, a row whose tag / fields disagree with its configuration, a row of a
configuration no stage asked for, a missing run, a non-finite value, a missing earlier selection; final also: a missing or
non-reproducing energy recount row (recomputed test MSE within 1e-5, syops body + head = logged synaptic energy within 1e-3).
env ET_DRY=<folder>: read flif-v1-ett-spkmux-ettune from <folder> (dry runs).
env ET_CHAIN=x (2026-11-20 extension, PROJECT_LOG; user decisions 11-20): the same chain and rules with the wider lists patch
{4, 8*, 16, 24, 32} and wd {0, 0.005, 0.01, 0.02, 0.05, 0.06*} (0.1 dropped); selections / final / new runs in
flif-v1-ett-spkmux-ettune2 (under ET_DRY when set); the rows of flif-v1-ett-spkmux-ettune (results/, read only) are reused (src 'ettune'),
the new rows carry src 'ettune2'; a configuration present in both suites stops (duplicate); only new-suite rows must belong to a requested
configuration; final also reports the extended result vs the first chain's final configuration (results/flif-v1-ett-spkmux-ettune/final.json).
Without ET_CHAIN the behaviour and every output are unchanged.
usage: python summarize_ettune.py --stage patch|dmodel|mlp|blocks|wd|final
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
DRY = os.environ.get('ET_DRY')
SUITE = 'flif-v1-ett-spkmux-ettune'
OUT = os.path.join(DRY, SUITE) if DRY else os.path.join(RES, SUITE)
TUNE = os.path.join(RES, 'flif-v1-ett-spkmux-tune')
SFDE = os.path.join(RES, 'flif-v1-ett-sfde')
sys.path.insert(0, os.path.join(RES, 'ett-pass-20261119'))
import pass_ett as pe                                                     # noqa: E402

COND = 'spkmx_lif_muxlin_iand'
LR = {'ETTh2': '0.0005', 'ETTm2': '0.0001'}
LYS = {'ETTh2': '1', 'ETTm2': '2'}                                        # SF-de layers (Table 13)
STAGES = (('patch', 'PATCH', ('4', '8', '16')), ('dmodel', 'EMB', ('32', '64', '128', '256')), ('mlp', 'MLPR', ('0.5', '1', '2', '4')),
          ('blocks', 'BLOCKS', ('1', '2', '3')), ('wd', 'WD', ('0', '0.01', '0.06', '0.1')))
NAMES = [s[0] for s in STAGES]
ORDER = ('PATCH', 'EMB', 'MLPR', 'BLOCKS', 'WD')
FIELD = {'PATCH': 'patch', 'EMB': 'emb', 'MLPR': 'mlpr', 'BLOCKS': 'blocks', 'WD': 'wd'}
SUF = {'PATCH': '_pt', 'EMB': '_e', 'MLPR': '_mlp', 'BLOCKS': '_blk', 'WD': '_wd'}
DEFAULT = {'ETTh2': {'PATCH': '8', 'EMB': '64', 'MLPR': '0.5', 'BLOCKS': '1', 'WD': '0.06'},
           'ETTm2': {'PATCH': '8', 'EMB': '64', 'MLPR': '1', 'BLOCKS': '1', 'WD': '0.06'}}
DSS, PLS, SEEDS = ('ETTh2', 'ETTm2'), (96, 720), (42, 7, 2026)
CELLS = tuple((d, p) for d in DSS for p in PLS)
EPS = 1e-12
CHAIN = os.environ.get('ET_CHAIN', '')
if CHAIN not in ('', 'x'):
    sys.exit(f'bad ET_CHAIN {CHAIN}')
OLD = None                                                                # first-chain suite reused by the extension (read only)
NEWSRC = 'ettune'
if CHAIN == 'x':
    OLD = os.path.join(RES, SUITE)
    SUITE, NEWSRC = 'flif-v1-ett-spkmux-ettune2', 'ettune2'
    OUT = os.path.join(DRY, SUITE) if DRY else os.path.join(RES, SUITE)
    STAGES = (('patch', 'PATCH', ('4', '8', '16', '24', '32')), ('dmodel', 'EMB', ('32', '64', '128', '256')),
              ('mlp', 'MLPR', ('0.5', '1', '2', '4')), ('blocks', 'BLOCKS', ('1', '2', '3')),
              ('wd', 'WD', ('0', '0.005', '0.01', '0.02', '0.05', '0.06')))
    NAMES = [s[0] for s in STAGES]


def stop(m):
    print(f'STOP: {m}')
    sys.exit(2)


def key(cfg):
    return tuple(cfg[k] for k in ORDER)


def tag(ds, cfg):
    return f'{COND}_lr{LR[ds]}_L1' + ''.join(f'{SUF[k]}{cfg[k]}' for k in ORDER if cfg[k] != DEFAULT[ds][k])


def cfgtxt(ds, cfg):
    d = [f'{k}={cfg[k]}' for k in ORDER if cfg[k] != DEFAULT[ds][k]]
    return ', '.join(d) if d else 'default (Table 13)'


def raw(path):
    out = []
    for l in open(path):
        if l.startswith('SKIP'):
            stop(f'SKIP line in {path}: {l.strip()[:160]}')
        if l.startswith('RESULT'):
            out.append(dict(t.split('=', 1) for t in l.split()[1:] if '=' in t))
    return out


def finite_row(r, where):
    be = str(r.get('best_epoch_val_epochs', '')).split(':')
    if len(be) != 3:
        stop(f'{where}: best_epoch_val_epochs {r.get("best_epoch_val_epochs")}')
    r['val'] = float(be[1])
    for k in ('mse', 'mae', 'val', 'total_fft_uJ', 'syn_energy_uJ'):
        r[k] = float(r[k])
        if not math.isfinite(r[k]):
            stop(f'{where}: non-finite {k}')
    if 'val_loss_best' in r and abs(float(r['val_loss_best']) - r['val']) > 1e-6:
        stop(f'{where}: val_loss_best {r["val_loss_best"]} != best-epoch validation {r["val"]}')
    return r


def load():
    """-> D[(ds, pl, seed, cfgkey)] = row; rows of the new suite carry r['src'] = 'ettune', the default rows 'tune'."""
    D = {}
    for r in raw(os.path.join(TUNE, 'raw.txt')):
        if r.get('cond') != COND or r['ds'] not in DSS or r.get('tag') != tag(r['ds'], DEFAULT[r['ds']]):
            continue
        if int(r['pl']) not in PLS or int(r['seed']) not in SEEDS:
            continue
        k = (r['ds'], int(r['pl']), int(r['seed']), key(DEFAULT[r['ds']]))
        if k in D:
            stop(f'duplicate default row {k}')
        r['src'] = 'tune'
        D[k] = finite_row(r, f'tune {k}')
    srcs = [(OUT, SUITE, NEWSRC)] if OLD is None else [(OLD, 'flif-v1-ett-spkmux-ettune', 'ettune'), (OUT, SUITE, NEWSRC)]
    for folder, sname, src in srcs:
        p = os.path.join(folder, 'raw.txt')
        for r in (raw(p) if os.path.exists(p) else []):
            ds = r.get('ds')
            if ds not in DSS or r.get('cond') != COND or r.get('lr') != LR[ds] or r.get('layers') != '1' or int(r['pl']) not in PLS \
                    or int(r['seed']) not in SEEDS:
                stop(f'unexpected row in {sname}: {r.get("tag")} {ds} {r.get("pl")} {r.get("seed")} lr {r.get("lr")} layers {r.get("layers")}')
            try:
                cfg = {k: r[FIELD[k]] for k in ORDER}
            except KeyError as e:
                stop(f'{sname} row without field {e}: {r.get("tag")}')
            if r['tag'] != tag(ds, cfg):
                stop(f'{sname} row tag {r["tag"]} != {tag(ds, cfg)} from its fields')
            if cfg == DEFAULT[ds]:
                stop(f'{sname} holds a default-configuration row ({ds} {r["pl"]} {r["seed"]}); the default rows come from the 11-01 tuning')
            k = (ds, int(r['pl']), int(r['seed']), key(cfg))
            if k in D:
                stop(f'duplicate row {k}')
            r['src'] = src
            D[k] = finite_row(r, f'{sname} {k}')
    return D


def selections(upto):
    """configuration per dataset before stage index `upto` (from selected_<stage>.json of the earlier stages) + the expected
    configurations of stages < upto (keys per dataset)."""
    base = {ds: dict(DEFAULT[ds]) for ds in DSS}
    expected = {ds: set() for ds in DSS}
    for name, knob, vals in STAGES[:upto]:
        p = os.path.join(OUT, f'selected_{name}.json')
        if not os.path.exists(p):
            stop(f'{p} missing')
        s = json.load(open(p))
        for ds in DSS:
            for v in vals:
                expected[ds].add(key(dict(base[ds], **{knob: v})))
            v = str(s['selected'][ds])
            if v not in vals:
                stop(f'{p}: {v} not in {vals}')
            base[ds][knob] = v
    return base, expected


def seed_mean(D, ds, pl, cfg, k):
    v = [D.get((ds, pl, s, key(cfg))) for s in SEEDS]
    return None if any(x is None for x in v) else float(np.mean([x[k] for x in v]))


def check_extra(D, expected):
    for (ds, pl, s, ck), r in D.items():
        if r['src'] == NEWSRC and ck not in expected[ds]:
            stop(f'{SUITE} row of a configuration no stage asked for: {r["tag"]} {ds} {pl} {s}')


def stage(name):
    k = NAMES.index(name)
    _, knob, vals = STAGES[k]
    D = load()
    base, expected = selections(k)
    for ds in DSS:
        for v in vals:
            expected[ds].add(key(dict(base[ds], **{knob: v})))
    check_extra(D, expected)
    sel, cur, scores, vt, miss = {}, {}, {}, {}, []
    for ds in DSS:
        cur[ds] = base[ds][knob]
        vt[ds] = {v: {pl: seed_mean(D, ds, pl, dict(base[ds], **{knob: v}), 'val') for pl in PLS} for v in vals}
        for v in vals:
            for pl in PLS:
                if vt[ds][v][pl] is None:
                    miss.append(f'{ds} H{pl} {knob}={v} ({tag(ds, dict(base[ds], **{knob: v}))})')
    if miss:
        stop('missing runs: ' + '; '.join(miss))
    for ds in DSS:
        scores[ds] = {v: float(np.mean([vt[ds][v][pl] / vt[ds][cur[ds]][pl] for pl in PLS])) for v in vals}
        best = min(scores[ds].values())
        sel[ds] = cur[ds] if abs(scores[ds][cur[ds]] - best) < EPS else min(vals, key=lambda v: (scores[ds][v], vals.index(v)))
    new = {'stage': name, 'knob': knob, 'selected': sel, 'current': cur,
           'config': {ds: dict(base[ds], **{knob: sel[ds]}) for ds in DSS}, 'scores': scores,
           'val': {ds: {v: {str(pl): vt[ds][v][pl] for pl in PLS} for v in vals} for ds in DSS}}
    p = os.path.join(OUT, f'selected_{name}.json')
    if os.path.exists(p):
        old = json.load(open(p))
        if old.get('selected') != sel:
            stop(f'{p} already selects {old.get("selected")}, now {sel}; not overwritten')
    L = [f'# ETT tuning of the LIF SpkMixer, stage {name} ({knob}); validation only', '',
         'Rule: summarize_ettune.py docstring (score = mean over H96 / H720 of the seed-mean best validation L1 relative to the current '
         'value; lowest wins; ties keep the current value). No test values are shown at this stage.', '']
    for ds in DSS:
        L += [f'## {ds} (before this stage: {cfgtxt(ds, base[ds])})', '',
              '| value | validation L1 H96 | validation L1 H720 | score | |', '|---|--:|--:|--:|---|']
        for v in vals:
            mark = ('**selected**' if v == sel[ds] else '') + (' (current)' if v == cur[ds] else '')
            L.append(f'| {knob}={v} | {vt[ds][v][96]:.6f} | {vt[ds][v][720]:.6f} | {scores[ds][v]:.6f} | {mark.strip()} |')
        L.append('')
    L.append(f'Selected: ' + '; '.join(f'{ds} {knob}={sel[ds]}' for ds in DSS))
    json.dump(new, open(p, 'w'), indent=1)
    open(os.path.join(OUT, f'selection_{name}.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def energy_rows():
    E = {}
    paths = [(os.path.join(OUT, 'energy_recount.csv'), NEWSRC), (os.path.join(TUNE, 'energy_recount.csv'), 'tune'),
             (os.path.join(SFDE, 'energy_recount.csv'), 'sfde')]
    if OLD is not None:
        paths.insert(1, (os.path.join(OLD, 'energy_recount.csv'), 'ettune'))
    for path, src in paths:
        if not os.path.exists(path):
            stop(f'{path} missing')
        for e in csv.DictReader(open(path)):
            k = (src, e.get('tag', ''), e['ds'], int(e['pl']), int(e['seed']))
            if k in E:
                stop(f'duplicate energy row {k}')
            E[k] = e
    return E


def attach(r, E, src):
    e = E.get((src, r['tag'], r['ds'], int(r['pl']), int(r['seed'])))
    if e is None:
        stop(f'no energy recount row for {src} {r["tag"]} {r["ds"]} {r["pl"]} {r["seed"]}')
    if abs(float(e['mse_recomputed']) - r['mse']) > 1e-5:
        stop(f'recomputed MSE {e["mse_recomputed"]} != logged {r["mse"]} ({r["tag"]} {r["ds"]} {r["pl"]} {r["seed"]})')
    syn = r['syn_energy_uJ']
    body = float(e['body_syn_uJ']) + float(e['head_MAC']) * 4.6e-6
    if abs(body - syn) > 1e-3 * syn:
        stop(f'syops body + head {body} != logged synaptic energy {syn} ({r["tag"]} {r["ds"]} {r["pl"]} {r["seed"]})')
    r['Es'], r['Eo'] = r['total_fft_uJ'], float(e['total_fft_uJ_corr'])
    return r


def sfde_rows(E):
    lrs = json.load(open(os.path.join(SFDE, 'selected_lr.json')))['selected']
    T = {}
    for r in raw(os.path.join(SFDE, 'raw.txt')):
        if r.get('cond') != 'spk_ssa' or r['ds'] not in DSS:
            continue
        if r['tag'] != f'spk_ssa_lr{lrs[r["ds"]]}_L{LYS[r["ds"]]}_nalif' or int(r['pl']) not in PLS or int(r['seed']) not in SEEDS:
            continue
        k = (r['ds'], int(r['pl']), int(r['seed']))
        if k in T:
            stop(f'duplicate SF-de row {k}')
        T[k] = attach(finite_row(r, f'SF-de {k}'), E, 'sfde')
    if len(T) != 12:
        stop(f'{len(T)} SF-de rows, want 12')
    return T


def arm(D, E, cfgs):
    T = {}
    for ds, pl in CELLS:
        for s in SEEDS:
            r = D.get((ds, pl, s, key(cfgs[ds])))
            if r is None:
                stop(f'missing run {ds} H{pl} seed {s} {tag(ds, cfgs[ds])}')
            T[(ds, pl, s)] = attach(r, E, r['src'])
    return T


def cmp(A, B):
    cell, pairs, er, eo, mae = [], [], [], [], []
    for ds, pl in CELLS:
        r = [100 * (A[(ds, pl, s)]['mse'] - B[(ds, pl, s)]['mse']) / B[(ds, pl, s)]['mse'] for s in SEEDS]
        cell.append(float(np.mean(r))); pairs += r
        mae.append(float(np.mean([100 * (A[(ds, pl, s)]['mae'] - B[(ds, pl, s)]['mae']) / B[(ds, pl, s)]['mae'] for s in SEEDS])))
        er.append(np.mean([A[(ds, pl, s)]['Es'] for s in SEEDS]) / np.mean([B[(ds, pl, s)]['Es'] for s in SEEDS]))
        eo.append(np.mean([A[(ds, pl, s)]['Eo'] for s in SEEDS]) / np.mean([B[(ds, pl, s)]['Eo'] for s in SEEDS]))
    R = float(np.mean(cell)); lo, hi = sum(z < 0 for z in pairs), sum(z > 0 for z in pairs)
    v = 'better' if R <= -1 and lo >= 8 else 'worse' if R >= 1 and hi >= 8 else 'no clear difference'
    return dict(R=R, lo=lo, verdict=v, Es=float(np.mean(er)), Eo=float(np.mean(eo)), cells=cell, R_mae=float(np.mean(mae)), pairs=pairs)


def cm(T, k):
    return float(np.mean([np.mean([T[(ds, pl, s)][k] for s in SEEDS]) for ds, pl in CELLS]))


def final():
    D = load()
    base, expected = selections(len(STAGES))
    check_extra(D, expected)
    E = energy_rows()
    n_new = sum(r['src'] == NEWSRC for r in D.values())
    for r in D.values():                                                  # every new run must have a reproducing recount row
        if r['src'] == NEWSRC:
            attach(r, E, NEWSRC)
    path = [{ds: dict(DEFAULT[ds]) for ds in DSS}]
    for name, knob, _ in STAGES:
        s = json.load(open(os.path.join(OUT, f'selected_{name}.json')))
        path.append({ds: dict(path[-1][ds], **{knob: str(s['selected'][ds])}) for ds in DSS})
    tuned, untuned, sf = arm(D, E, base), arm(D, E, DEFAULT), sfde_rows(E)
    steps = [arm(D, E, p) for p in path]
    L = ['# ETT tuning of the LIF SpkMixer: final (test values; selection was validation only)'
         + (' -- extended chain (ET_CHAIN=x: patch up to 32, wd within 0-0.06)' if CHAIN == 'x' else ''), '',
         f'Rule and sources: summarize_ettune.py docstring. New runs in {SUITE}: {n_new}. Seeds 42 / 7 / 2026, input 96.', '',
         '## Selection path (validation score of the winner; 1 = kept the current value)', '',
         '| stage | ' + ' | '.join(DSS) + ' |', '|---|' + '---|' * len(DSS)]
    for name, knob, _ in STAGES:
        s = json.load(open(os.path.join(OUT, f'selected_{name}.json')))
        L.append(f'| {name} ({knob}) | ' + ' | '.join(f'{s["current"][ds]} → **{s["selected"][ds]}** ({s["scores"][ds][s["selected"][ds]]:.4f})'
                                                     for ds in DSS) + ' |')
    L += ['', 'Final configuration: ' + '; '.join(f'{ds}: {cfgtxt(ds, base[ds])} (tag {tag(ds, base[ds])})' for ds in DSS), '',
          '## Test MSE (seed mean), 4-cell means, energy', '',
          '| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean MSE | mean MAE | mean best validation L1 | energy µJ syops | energy µJ old rule | params ETTh2 H96 / ETTm2 H96 |',
          '|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|']

    def trow(nm, T):
        cells = ' | '.join(f'{np.mean([T[(ds, pl, s)]["mse"] for s in SEEDS]):.4f}' for ds, pl in CELLS)
        pa = ' / '.join(str(T[(ds, 96, 42)].get('params', '?')) for ds in DSS)
        return f'| {nm} | {cells} | {cm(T, "mse"):.4f} | {cm(T, "mae"):.4f} | {cm(T, "val"):.5f} | {cm(T, "Es"):.2f} | {cm(T, "Eo"):.2f} | {pa} |'
    L += [trow('**tuned LIF SpkMixer**', tuned), trow('LIF SpkMixer (untuned, 11-01)', untuned), trow('SF-de (lr only)', sf), '',
          'Report: the configuration after each stage (test values were not used for any selection):', '',
          '| after stage | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean MSE | mean MAE | mean best validation L1 | energy µJ syops | energy µJ old rule | params ETTh2 H96 / ETTm2 H96 |',
          '|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|']
    for nm, T in zip(['(none: untuned)'] + NAMES, steps):
        L.append(trow(nm, T))
    L += ['', '## Comparisons (ETT rule; test MSE)', '',
          '| comparison | R % | lower pairs | verdict | energy A/B syops | energy A/B old rule | per cell r % | MAE R % |', '|---|--:|--:|---|--:|--:|---|--:|']
    res = {}
    pairs = [('tuned vs untuned LIF SpkMixer', tuned, untuned), ('tuned LIF SpkMixer vs SF-de', tuned, sf),
             ('(report) untuned LIF SpkMixer vs SF-de', untuned, sf)]
    if OLD is not None:                                                   # the first chain's final configuration (report only)
        first = json.load(open(os.path.join(OLD, 'final.json')))['config']
        pairs.append(('(report) tuned (extended) vs tuned (first chain, 11-20 04:02)', tuned, arm(D, E, first)))
    for nm, A, B in pairs:
        c = cmp(A, B); res[nm] = {k: v for k, v in c.items() if k != 'pairs'}
        L.append(f'| {nm} | {c["R"]:+.2f} | {c["lo"]}/12 | {c["verdict"]} | {c["Es"]:.4f} | {c["Eo"]:.4f} | '
                 f'{", ".join(f"{z:+.2f}" for z in c["cells"])} | {c["R_mae"]:+.2f} |')
    L += ['', '## User pass criteria (results/ett-pass-20261119/pass_ett.py)', '']
    passes = {}
    for nm, T in (('tuned LIF SpkMixer', tuned), ('untuned LIF SpkMixer (reference)', untuned)):
        V, VS = pe.vals(T)[0], pe.vals(sf)[0]
        EA = {c: [T[(c[0], c[1], s)]['Es'] for s in SEEDS] for c in CELLS}
        ES = {c: [sf[(c[0], c[1], s)]['Es'] for s in SEEDS] for c in CELLS}
        r, lines = pe.judge(nm, V, VS, EA, ES)
        passes[nm] = r
        L += lines
    out = {'config': base, 'tags': {ds: tag(ds, base[ds]) for ds in DSS},
           'tuned': {'mse': cm(tuned, 'mse'), 'mae': cm(tuned, 'mae'), 'Es': cm(tuned, 'Es'), 'Eo': cm(tuned, 'Eo')},
           'untuned': {'mse': cm(untuned, 'mse'), 'mae': cm(untuned, 'mae'), 'Es': cm(untuned, 'Es')},
           'sfde': {'mse': cm(sf, 'mse'), 'mae': cm(sf, 'mae'), 'Es': cm(sf, 'Es')},
           'comparisons': res, 'pass': passes, 'n_new_runs': n_new}
    json.dump(out, open(os.path.join(OUT, 'final.json'), 'w'), indent=1)
    open(os.path.join(OUT, 'summary_final.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=NAMES + ['final'])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    final() if a.stage == 'final' else stage(a.stage)


if __name__ == '__main__':
    main()
