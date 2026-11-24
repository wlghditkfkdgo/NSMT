"""ETTh1 / ETTm1 retuning of the LIF SpkMixer: selections and the 16-cell judgment (2026-11-25, exp/f-lif-pop-v3; PROJECT_LOG
pre-registration of 2026-11-25; user decision 11-25 "(나)"; plan and selection rule in rt_plan.py).

--stage lr|layers|patch|dmodel|mlp|blocks|wd   selection (validation only; no test value is printed or written):
    writes selected_<stage>.json (shared stages) or selected_<chain>_<stage>.json (chains a, x) and selection_<stage>.md;
    refuses to overwrite a different selection.
--stage judge   after the final stage and energy_recount_ett.py --suite flif-v1-ett-spkmux-retune. Two models, as on 11-24:
    first  ETTh1 / ETTm1: chain a's configuration (new);  ETTh2 / ETTm2: the first-chain final of 11-20 04:02 (unchanged)
    ext    ETTh1 / ETTm1: chain x's configuration (new);  ETTh2 / ETTm2: the extension final of 11-20 17:07 (unchanged)
    16 cells (4 datasets x H 96 / 192 / 336 / 720), seed mean per cell. ETTh1 / ETTm1 H96 / H720 = the tuning runs of the selected
    configuration, H192 / H336 = the final-stage runs; ETTh2 / ETTm2: the same rows as the 11-24 full table (old suites + full16),
    which must reproduce results/flif-v1-ett-spkmux-full16/summary.json (1e-9).
    Pre-registered rule (unchanged from 11-24 18:02, NSMT removed by the user's correction of 11-24 20:1x):
      accuracy  16-cell means; ref - ours >= 0.001 for MSE and for MAE against each of SpikF, SpikeRNN, iSpikformer (SpikF paper
                Table 12) and Informer, FEDformer, PatchTST, ETSformer, Autoformer (TimesNet Table 13 / SpikF Tables 11-12;
                results/ett-full16-20261124/refs16.json, rows with role SNN or DNN)
      energy    syops only: seed-mean syn_energy_uJ below SpikF's syops count in every one of the 16 cells (spikf16/*.csv)
      PASS = all 8 accuracy rows and the energy rule.
    Report only: selection paths, per-dataset tables, ETTh1 / ETTm1 against the 11-24 full table (settings copied from ETTh2 / ETTm2),
    per-dataset 4-horizon means against SpikF, seed spread.
Fail closed (exit 2, nothing written): SKIP line, duplicate, a row whose tag / fields / epoch count disagree, a row no stage asked for,
missing run, non-finite value, missing earlier selection; judge also: a missing or non-reproducing energy row (recomputed test MSE
within 1e-5, syops body + head = logged synaptic energy within 1e-3), ETTh2 / ETTm2 rows that do not reproduce the 11-24 summary.
usage (from cinc_mixer_v1/): python -B summarize_rt.py --stage <stage>|judge
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

import rt_plan as P

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
EPS, MARGIN = 1e-12, 0.001
ALLDS, ALLPL = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'), (96, 192, 336, 720)
CELLS = [(d, p) for d in ALLDS for p in ALLPL]
F16 = os.path.join(RES, 'flif-v1-ett-spkmux-full16')
REFDIR = os.path.join(RES, 'ett-full16-20261124')
# ETTh2 / ETTm2 rows of the two 11-24 models: (suite of H96 / H720, tag per family); H192 / H336 from full16
OLDM = {'first': ('flif-v1-ett-spkmux-ettune', {'h': 'spkmx_lif_muxlin_iand_lr0.0005_L1_pt16_wd0.1', 'm': 'spkmx_lif_muxlin_iand_lr0.0001_L1_pt16_mlp2_wd0.1'}),
        'ext': ('flif-v1-ett-spkmux-ettune2', {'h': 'spkmx_lif_muxlin_iand_lr0.0005_L1_pt24_wd0.005', 'm': 'spkmx_lif_muxlin_iand_lr0.0001_L1_pt24_e256_blk2_wd0.02'})}
CH = {'first': 'a', 'ext': 'x'}
NAME = {'first': 'first chain', 'ext': 'extension'}
REFS = ('SpikF', 'SpikeRNN', 'iSpikformer', 'Informer', 'FEDformer', 'PatchTST', 'ETSformer', 'Autoformer')
stop = P.stop


def rows(path):
    out = []
    for l in open(path):
        if l.startswith('SKIP'):
            stop(f'SKIP line in {path}: {l.strip()[:160]}')
        if l.startswith('RESULT'):
            out.append(dict(t.split('=', 1) for t in l.split()[1:] if '=' in t))
    return out


def finite(r, where):
    be = str(r.get('best_epoch_val_epochs', '')).split(':')
    if len(be) != 3:
        stop(f'{where}: best_epoch_val_epochs {r.get("best_epoch_val_epochs")}')
    r['val'] = float(be[1])
    for k in ('mse', 'mae', 'val', 'syn_energy_uJ', 'total_fft_uJ'):
        r[k] = float(r[k])
        if not math.isfinite(r[k]):
            stop(f'{where}: non-finite {k}')
    if abs(float(r['val_loss_best']) - r['val']) > 1e-6:
        stop(f'{where}: val_loss_best {r["val_loss_best"]} != best-epoch validation {r["val"]}')
    return r


def load():
    """-> D[(ds, pl, seed, cfgkey)] = row of the retune suite; checks every row against the plan."""
    p = os.path.join(P.OUT, 'raw.txt')
    if not os.path.exists(p):
        stop(f'{p} missing')
    sel, fin = P.requested()
    D = {}
    for r in rows(p):
        ds = r.get('ds')
        if ds not in P.DSS or r.get('cond') != P.COND or int(r['seed']) not in P.SEEDS:
            stop(f'unexpected row {r.get("tag")} {ds} {r.get("pl")} {r.get("seed")}')
        try:
            cfg = {k: r[P.FIELD[k]] for k in P.KN}
        except KeyError as e:
            stop(f'row without field {e}: {r.get("tag")}')
        if r['tag'] != P.tag(ds, cfg):
            stop(f'row tag {r["tag"]} != {P.tag(ds, cfg)} from its fields')
        if f'+epoch+{P.EPOCHS}+' not in r.get('dir', ''):
            stop(f'row not trained with {P.EPOCHS} epochs: {r.get("dir")}')
        pl = int(r['pl'])
        k = (ds, pl, int(r['seed']), P.key(cfg))
        if (pl in P.SELPL and (ds, k[3]) not in sel) or (pl in P.FINPL and (ds, k[3]) not in fin) or pl not in P.SELPL + P.FINPL:
            stop(f'row of a configuration no stage asked for: {r["tag"]} {ds} H{pl} seed {r["seed"]}')
        if k in D:
            stop(f'duplicate row {k}')
        D[k] = finite(r, f'{k}')
    return D


def sm(D, ds, pl, cfg, f):
    v = [D.get((ds, pl, s, P.key(cfg))) for s in P.SEEDS]
    return None if any(x is None for x in v) else float(np.mean([x[f] for x in v]))


def stage(name):
    D = load()
    E, _ = P.walk()
    es = [e for e in E if e['name'] == name]
    need = 1 if name in [x[0] for x in P.SHARED] else len(P.CHAINS)
    if len(es) != need:
        stop(f'stage {name} not reachable ({len(es)} of {need} entries)')
    L = [f'# ETTh1 / ETTm1 retuning of the LIF SpkMixer, stage {name} ({es[0]["knob"]}); validation only', '',
         'Rule: rt_plan.py docstring (score = mean over H96 / H720 of the seed-mean best validation L1 relative to the current value; '
         'lowest wins; within 1e-12 of the current value keeps it). No test values are shown at this stage.', '']
    W = []                                                                        # written only after every entry passed
    for e in es:
        knob, vals = e['knob'], e['vals']
        vt, miss = {}, []
        for ds in P.DSS:
            vt[ds] = {v: {pl: sm(D, ds, pl, dict(e['base'][ds], **{knob: v}), 'val') for pl in P.SELPL} for v in vals}
            miss += [f'{ds} H{pl} {knob}={v}' for v in vals for pl in P.SELPL if vt[ds][v][pl] is None]
        if miss:
            stop('missing runs: ' + '; '.join(miss))
        sel, cur, sc = {}, {}, {}
        for ds in P.DSS:
            cur[ds] = e['base'][ds][knob]
            sc[ds] = {v: float(np.mean([vt[ds][v][pl] / vt[ds][cur[ds]][pl] for pl in P.SELPL])) for v in vals}
            best = min(sc[ds].values())
            sel[ds] = cur[ds] if abs(sc[ds][cur[ds]] - best) < EPS else min(vals, key=lambda v: (sc[ds][v], vals.index(v)))
        new = {'stage': name, 'chain': e['chain'], 'knob': knob, 'selected': sel, 'current': cur, 'scores': sc,
               'config': {ds: dict(e['base'][ds], **{knob: sel[ds]}) for ds in P.DSS},
               'val': {ds: {v: {str(pl): vt[ds][v][pl] for pl in P.SELPL} for v in vals} for ds in P.DSS}}
        f = P.selfile(e['chain'], name)
        if os.path.exists(f) and json.load(open(f)).get('selected') != sel:
            stop(f'{f} already selects {json.load(open(f)).get("selected")}, now {sel}; not overwritten')
        W.append((f, new))
        hdr = 'shared' if e['chain'] is None else P.CNAME[e['chain']]
        for ds in P.DSS:
            L += [f'## {hdr}: {ds} (before this stage: {P.cfgtxt(e["base"][ds])})', '',
                  '| value | validation L1 H96 | validation L1 H720 | score | |', '|---|--:|--:|--:|---|']
            for v in vals:
                mark = ('**selected**' if v == sel[ds] else '') + (' (current)' if v == cur[ds] else '')
                L.append(f'| {knob}={v} | {vt[ds][v][96]:.6f} | {vt[ds][v][720]:.6f} | {sc[ds][v]:.6f} | {mark.strip()} |')
            L.append('')
        L.append(f'Selected ({hdr}): ' + '; '.join(f'{ds} {knob}={sel[ds]}' for ds in P.DSS))
        L.append('')
    for f, new in W:
        json.dump(new, open(f, 'w'), indent=1)
    open(os.path.join(P.OUT, f'selection_{name}.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def fam(ds):
    return 'h' if ds.startswith('ETTh') else 'm'


def old_rows():
    """ETTh2 / ETTm2 rows of the two 11-24 models -> O[mk][(ds, pl, seed)] = (mse, mae, syn); must equal full16 summary.json."""
    O = {mk: {} for mk in OLDM}
    for mk, (suite, tg) in OLDM.items():
        for src, pls in ((os.path.join(RES, suite, 'raw.txt'), (96, 720)), (os.path.join(F16, 'raw.txt'), (192, 336))):
            for r in rows(src):
                ds = r.get('ds')
                if ds not in ('ETTh2', 'ETTm2') or r.get('tag') != tg[fam(ds)] or int(r['pl']) not in pls or int(r['seed']) not in P.SEEDS:
                    continue
                k = (ds, int(r['pl']), int(r['seed']))
                if k in O[mk]:
                    stop(f'duplicate old row {mk} {k}')
                O[mk][k] = tuple(float(r[x]) for x in ('mse', 'mae', 'syn_energy_uJ'))
        if len(O[mk]) != 24:
            stop(f'{mk}: {len(O[mk])} ETTh2 / ETTm2 rows, want 24')
    S = json.load(open(os.path.join(F16, 'summary.json')))
    for mk in OLDM:
        for ds in ('ETTh2', 'ETTm2'):
            for pl in ALLPL:
                c = S['models'][mk]['cells'][f'{ds} {pl}']
                for j, f in enumerate(('mse', 'mae', 'syn_energy_uJ')):
                    if abs(np.mean([O[mk][(ds, pl, s)][j] for s in P.SEEDS]) - c[f]) > 1e-9:
                        stop(f'{mk} {ds} H{pl} {f} does not reproduce the 11-24 summary')
    return O, S


def judge():
    D = load()
    E, F = P.walk()
    if F is None:
        stop('judge needs every chain selection')
    en = {}
    p = os.path.join(P.OUT, 'energy_recount.csv')
    if not os.path.exists(p):
        stop(f'{p} missing')
    for e in csv.DictReader(open(p)):
        k = (e['tag'], e['ds'], int(e['pl']), int(e['seed']))
        if k in en:
            stop(f'duplicate energy row {k}')
        en[k] = e
    O, S16 = old_rows()
    V = {mk: {} for mk in OLDM}                                                    # (ds, pl, seed) -> (mse, mae, syn)
    for mk in OLDM:
        V[mk].update(O[mk])
        for ds in P.DSS:
            cfg = F[CH[mk]][ds]
            for pl in ALLPL:
                for s in P.SEEDS:
                    r = D.get((ds, pl, s, P.key(cfg)))
                    if r is None:
                        stop(f'missing run {mk} {ds} H{pl} seed {s} ({P.tag(ds, cfg)})')
                    e = en.get((r['tag'], ds, pl, s))
                    if e is None:
                        stop(f'no energy row {r["tag"]} {ds} H{pl} seed {s}')
                    if abs(float(e['mse_recomputed']) - r['mse']) > 1e-5:
                        stop(f'recomputed MSE {e["mse_recomputed"]} != {r["mse"]} ({r["tag"]} {ds} H{pl} seed {s})')
                    body = float(e['body_syn_uJ']) + float(e['head_MAC']) * 4.6e-6
                    if abs(body - r['syn_energy_uJ']) > 1e-3 * r['syn_energy_uJ']:
                        stop(f'syops body + head {body} != {r["syn_energy_uJ"]} ({r["tag"]} {ds} H{pl} seed {s})')
                    V[mk][(ds, pl, s)] = (r['mse'], r['mae'], r['syn_energy_uJ'])
        if len(V[mk]) != 48:
            stop(f'{mk}: {len(V[mk])} rows, want 48')
    REF = json.load(open(os.path.join(REFDIR, 'refs16.json')))['models']
    for m in REFS:
        if REF[m]['role'] not in ('SNN', 'DNN'):
            stop(f'{m}: role {REF[m]["role"]}')
    SP = {}
    for ds in ALLDS:
        for r in csv.DictReader(open(os.path.join(REFDIR, 'spikf16', f'spikf_syops_{ds}.csv'))):
            SP[(r['ds'], int(r['pl']))] = float(r['energy_uJ'])
    if sorted(SP) != sorted(CELLS) or not all(math.isfinite(v) and v > 0 for v in SP.values()):
        stop('SpikF energy rows incomplete')

    def cm(mk, c, j):
        return float(np.mean([V[mk][(c[0], c[1], s)][j] for s in P.SEEDS]))

    def ref(m, c, j):
        return REF[m]['rows'][f'{c[0]} {c[1]}'][j]

    L = ['# ETTh1 / ETTm1 retuned on their own validation: full ETT table, input 96', '',
         'Rules: summarize_rt.py / rt_plan.py docstrings. Ours: seed mean of 42 / 7 / 2026 (test values of the best-validation weights). '
         'ETTh2 / ETTm2 cells are the 11-24 values (unchanged). Paper rows: single published numbers.', '',
         '## Selection paths on ETTh1 / ETTm1 (validation score of the winner; 1 = kept the current value)', '']
    paths = {}
    for c in [None] + list(P.CHAINS):
        es = [e for e in E if e['chain'] == c]
        L += [f'### {"shared stages" if c is None else P.CNAME[c]}', '', '| stage | ETTh1 | ETTm1 |', '|---|---|---|']
        for e in es:
            s = json.load(open(P.selfile(c, e['name'])))
            L.append(f'| {e["name"]} ({e["knob"]}) | ' + ' | '.join(f'{s["current"][ds]} → **{s["selected"][ds]}** ({s["scores"][ds][s["selected"][ds]]:.4f})'
                                                                for ds in P.DSS) + ' |')
            paths.setdefault(str(c), {})[e['name']] = {ds: [s['current'][ds], s['selected'][ds], s['scores'][ds][s['selected'][ds]]] for ds in P.DSS}
        L.append('')
    L += ['Final configurations: ' + '; '.join(f'{NAME[mk]} {ds}: {P.cfgtxt(F[CH[mk]][ds])} (tag {P.tag(ds, F[CH[mk]][ds])})'
                                                 for mk in OLDM for ds in P.DSS), '']
    SHOW = ('SpikF', 'SpikeRNN', 'iSpikformer', 'PatchTST')
    for ds in ALLDS:
        L += [f'## {ds} (test MSE / MAE){"" if ds in P.DSS else " — unchanged from 11-24"}', '',
              '| H | first chain | extension | ' + ' | '.join(SHOW) + ' |', '|---|---|---|' + '---|' * len(SHOW)]
        for pl in ALLPL:
            c = (ds, pl)
            L.append(f'| {pl} | ' + ' | '.join(f'{cm(mk, c, 0):.4f} / {cm(mk, c, 1):.4f}' for mk in OLDM) + ' | '
                     + ' | '.join(f'{ref(m, c, 0):.3f} / {ref(m, c, 1):.3f}' for m in SHOW) + ' |')
        L.append('| Avg | ' + ' | '.join(f'{np.mean([cm(mk, (ds, q), 0) for q in ALLPL]):.4f} / {np.mean([cm(mk, (ds, q), 1) for q in ALLPL]):.4f}'
                                         for mk in OLDM) + ' | '
                 + ' | '.join(f'{REF[m]["rows"][f"{ds} Avg"][0]:.3f} / {REF[m]["rows"][f"{ds} Avg"][1]:.3f}' for m in SHOW) + ' |')
        L.append('')
    out = {'final_config': F, 'final_tags': {c: {ds: P.tag(ds, F[c][ds]) for ds in P.DSS} for c in P.CHAINS}, 'paths': paths, 'models': {}}
    L += ['## Judgment (16-cell means; margin = ref − ours, ≥ +0.0010 needed for MSE and MAE)', '']
    for mk in OLDM:
        om = float(np.mean([cm(mk, c, 0) for c in CELLS])); oa = float(np.mean([cm(mk, c, 1) for c in CELLS]))
        old = S16['models'][mk]
        L += [f'### {NAME[mk]}: 16-cell MSE / MAE **{om:.4f} / {oa:.4f}** (11-24 full table: {old["mse16"]:.4f} / {old["mae16"]:.4f})', '',
              '| reference | role | ref 16-cell MSE / MAE | margin MSE / MAE | passes | cells with lower MSE / MAE (of 16) |', '|---|---|---|---|---|---|']
        res = {}
        for m in REFS:
            rm = float(np.mean([ref(m, c, 0) for c in CELLS])); ra = float(np.mean([ref(m, c, 1) for c in CELLS]))
            dm, da = rm - om, ra - oa
            ok = dm >= MARGIN - EPS and da >= MARGIN - EPS
            lo = (sum(cm(mk, c, 0) < ref(m, c, 0) for c in CELLS), sum(cm(mk, c, 1) < ref(m, c, 1) for c in CELLS))
            L.append(f'| {m} | {REF[m]["role"]} | {rm:.4f} / {ra:.4f} | {dm:+.4f} / {da:+.4f} | {"yes" if ok else "no"} | {lo[0]} / {lo[1]} |')
            res[m] = dict(role=REF[m]['role'], ref_mse=rm, ref_mae=ra, margin_mse=dm, margin_mae=da, passes=bool(ok), lower_mse=lo[0], lower_mae=lo[1])
        er = {c: cm(mk, c, 2) / SP[c] for c in CELLS}
        below = sum(v < 1 for v in er.values())
        acc = all(r['passes'] for r in res.values())
        verdict = 'PASS' if acc and below == 16 else 'NOT PASSED'
        L += ['', f'Energy (syops): below SpikF in **{below} / 16** cells (needed 16); mean of cell ratios ours / SpikF {np.mean(list(er.values())):.4f}, '
              f'largest {max(er.values()):.4f} ({max(er, key=er.get)[0]} H{max(er, key=er.get)[1]}); 16-cell mean {np.mean([cm(mk, c, 2) for c in CELLS]):.2f} µJ '
              f'(SpikF {np.mean([SP[c] for c in CELLS]):.2f} µJ).',
              f'Accuracy rows (3 SNN + 5 DNN): {"all pass" if acc else "NOT all pass: " + ", ".join(m for m, r in res.items() if not r["passes"])}. '
              f'**Verdict: {verdict}**', '']
        out['models'][mk] = dict(name=NAME[mk], mse16=om, mae16=oa, rows=res, energy_cells_below_spikf=below,
                                 energy_ratio_spikf_mean=float(np.mean(list(er.values()))), energy_ratio_spikf_max=float(max(er.values())),
                                 energy16_uJ=float(np.mean([cm(mk, c, 2) for c in CELLS])), accuracy_all=bool(acc), verdict=verdict,
                                 cells={f'{c[0]} {c[1]}': dict(mse=cm(mk, c, 0), mae=cm(mk, c, 1), syn_energy_uJ=cm(mk, c, 2),
                                                             mse_std=float(np.std([V[mk][(c[0], c[1], s)][0] for s in P.SEEDS]))) for c in CELLS})
    L += ['## Report: ETTh1 / ETTm1 against the 11-24 full table (settings copied from ETTh2 / ETTm2), test MSE seed mean', '',
          '| model | cell | 11-24 | retuned | difference |', '|---|---|--:|--:|--:|']
    for mk in OLDM:
        for ds in P.DSS:
            for pl in ALLPL:
                a, b = S16['models'][mk]['cells'][f'{ds} {pl}']['mse'], cm(mk, (ds, pl), 0)
                L.append(f'| {NAME[mk]} | {ds} H{pl} | {a:.4f} | {b:.4f} | {b - a:+.4f} |')
    L += ['', '## Report: per dataset, mean of the 4 horizons against SpikF (+ = ours higher = worse)', '',
          '| model | dataset | MSE ours − SpikF | MAE ours − SpikF |', '|---|---|--:|--:|']
    for mk in OLDM:
        for ds in ALLDS:
            dm = np.mean([cm(mk, (ds, q), 0) - ref('SpikF', (ds, q), 0) for q in ALLPL])
            da = np.mean([cm(mk, (ds, q), 1) - ref('SpikF', (ds, q), 1) for q in ALLPL])
            L.append(f'| {NAME[mk]} | {ds} | {dm:+.4f} | {da:+.4f} |')
    L += ['', '## Energy per cell (syops µJ per input window; ours seed mean, SpikF one checkpoint)', '',
          '| cell | first chain | extension | SpikF | first / SpikF | ext / SpikF |', '|---|--:|--:|--:|--:|--:|']
    for c in CELLS:
        a, b = cm('first', c, 2), cm('ext', c, 2)
        L.append(f'| {c[0]} H{c[1]} | {a:.2f} | {b:.2f} | {SP[c]:.2f} | {a / SP[c]:.4f} | {b / SP[c]:.4f} |')
    L += ['', '## Seed spread (std of test MSE over the 3 seeds, mean over the 16 cells)', '']
    for mk in OLDM:
        L.append(f'- {NAME[mk]}: {np.mean([out["models"][mk]["cells"][f"{c[0]} {c[1]}"]["mse_std"] for c in CELLS]):.4f}')
    out['n_rows_suite'] = len(D)
    json.dump(out, open(os.path.join(P.OUT, 'judge.json'), 'w'), indent=1)
    open(os.path.join(P.OUT, 'summary_judge.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=[s for s in P.STAGES if s != 'final'] + ['judge'])
    a = ap.parse_args()
    os.makedirs(P.OUT, exist_ok=True)
    judge() if a.stage == 'judge' else stage(a.stage)


if __name__ == '__main__':
    main()
