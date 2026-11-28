"""AD hyperparameter tuning of the LIF SpkMixer, stage by stage (2026-11-29, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; user "(가)
epoch 수, 정규화, layer/block 개수, hidden_dim 개수 등등 하이퍼파라미터 튜닝", decisions "test 전체로 선택" + "AUC-PR로 선택").
Start = the 11-19 G5 setting per dataset (muxlin, one mixer layer per direction, lr 2e-3, emb 64, patch 8, mlp 1, one block, window
normalisation, 10 epochs, patience 3; residual MSL OR, SMAP OR, PSM IAND). Stages in this order; each changes one setting (or one pair)
per dataset, the others stay at the values selected so far:
  norm    ADNORM     window* | mean | none          (ad_wrap.Reconstructor: mean / std over the window, mean only, nothing)
  epochs  EPOCHS     1 | 3 | 10* | 30               (patience 3 on the validation MSE, best-validation weights tested)
  width   EMB        32 | 64* | 128
  depth   BLOCKS, MLAYERS   (1, 1)* | (1, 2) | (2, 1)
  patch   PATCH, MLP        (8, 1)* | (4, 1) | (20, 1) | (8, 2)   (patch 16 / 12 leave the last 4 steps unreconstructed: N x stride = 96)
  lr      LR         1e-3 | 2e-3* | 4e-3
  res     residual   OR | IAND | ADD               (* = the start value; the current value of a stage is whatever earlier stages chose)
Selection (user decision, stated as optimistic): per dataset the candidate with the highest seed-mean TEST AUC-PR (seeds 42 / 7 / 2026);
a candidate within 1e-9 of the current value keeps the current value; other ties keep the earlier candidate in the list. A missing
seed, a duplicate row or a SKIP line stops (exit 2). selected_<stage>.json is never overwritten with a different selection.
Rows: whole training file (holdout 0) from flif-v1-ad-energy, flif-v1-ad-iand (11-19 / 11-28 runs at the same settings; the 11-28
reproduction showed bit-exact training) and the tuning suite flif-v1-ad-tune2; one (dataset, tag, seed) may appear in one suite only.
usage (from cinc_mixer_v1/):
  python -B adtune.py --stage <s> --queue <file>     write the missing runs of stage <s> (needs the earlier selections)
  python -B adtune.py --stage <s> --select           select stage <s> -> results/flif-v1-ad-tune2/selected_<s>.json + selection_<s>.md
  (ADT_ROOT=<dir>: read / write suites under <dir> instead of results/; dry runs)
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('ADT_ROOT') or os.path.join(HERE, 'results')
SUITE = 'flif-v1-ad-tune2'
SUITES = ('flif-v1-ad-energy', 'flif-v1-ad-iand', SUITE)
DSS, SEEDS = ('SMAP', 'PSM', 'MSL'), ('42', '7', '2026')                # slowest dataset first in every queue
BASE = {ds: dict(res=r, lr='0.002', ml=1, bl=1, emb=64, pt=8, mlp='1', norm='window', ep=10) for ds, r in (('MSL', 'or'), ('SMAP', 'or'), ('PSM', 'iand'))}
STAGES = [('norm', ('norm',), [('window',), ('mean',), ('none',)]),
          ('epochs', ('ep',), [(1,), (3,), (10,), (30,)]),
          ('width', ('emb',), [(32,), (64,), (128,)]),
          ('depth', ('bl', 'ml'), [(1, 1), (1, 2), (2, 1)]),
          ('patch', ('pt', 'mlp'), [(8, '1'), (4, '1'), (20, '1'), (8, '2')]),
          ('lr', ('lr',), [('0.001',), ('0.002',), ('0.004',)]),
          ('res', ('res',), [('or',), ('iand',), ('add',)])]
NAMES = [s[0] for s in STAGES]


def stop(m):
    print(f'STOP: {m}')
    sys.exit(2)


def tag(c):
    t = f"spkmx_lif_muxlin_{c['res']}_lr{c['lr']}_L{c['ml']}"
    t += f"_b{c['bl']}" if c['bl'] != 1 else ''
    t += f"_e{c['emb']}" if c['emb'] != 64 else ''
    t += f"_p{c['pt']}" if c['pt'] != 8 else ''
    t += f"_mlp{c['mlp']}" if c['mlp'] != '1' else ''
    t += f"_nm{c['norm']}" if c['norm'] != 'window' else ''
    t += f"_ep{c['ep']}" if c['ep'] != 10 else ''
    return t


def line(ds, c, s):
    return (f"SUITE={SUITE} COND=spkmx_lif_muxlin_{c['res']} LR={c['lr']} MLAYERS={c['ml']} BLOCKS={c['bl']} EMB={c['emb']} PATCH={c['pt']} "
            f"MLP={c['mlp']} ADNORM={c['norm']} EPOCHS={c['ep']} bash cinc_mixer_v1/scripts/run_ad_spk.sh {ds} {s}")


def index():
    D = {}
    for su in SUITES:
        p = os.path.join(ROOT, su, 'raw.txt')
        if not os.path.exists(p):
            if su == SUITE:
                continue
            stop(f'{p} missing')
        L = open(p).read().splitlines()
        if su == SUITE and any(l.startswith('SKIP') for l in L):
            stop(f'SKIP line in {p}')
        for l in L:
            if not l.startswith('RESULT'):
                continue
            r = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
            if r.get('holdout') != '0' or not r.get('cond', '').startswith('spkmx_lif_muxlin_'):
                continue
            k = (r['ds'], r['tag'], r['seed'])
            if k in D:
                stop(f'duplicate row {k} ({D[k]["_suite"]}, {su})')
            r['_suite'] = su
            D[k] = r
    return D


def apply(c, keys, val):
    c = dict(c)
    for k, v in zip(keys, val):
        c[k] = v
    return c


def current(upto):
    """config per dataset after the stages before `upto` (their selections must exist)."""
    C = {ds: dict(BASE[ds]) for ds in DSS}
    for name, keys, _ in STAGES[:NAMES.index(upto)]:
        p = os.path.join(ROOT, SUITE, f'selected_{name}.json')
        if not os.path.exists(p):
            stop(f'{p} missing (stage {name} not selected)')
        sel = json.load(open(p))['selected']
        for ds in DSS:
            C[ds] = apply(C[ds], keys, tuple(sel[ds]))
    return C


def cands(stage, C):
    keys, vals = STAGES[NAMES.index(stage)][1:]
    return {ds: [(v, apply(C[ds], keys, v)) for v in vals] for ds in DSS}


def three(D, ds, c):
    v = [D.get((ds, tag(c), s)) for s in SEEDS]
    return None if any(x is None for x in v) else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=NAMES, required=True)
    ap.add_argument('--queue')
    ap.add_argument('--select', action='store_true')
    a = ap.parse_args()
    C = current(a.stage)
    K = cands(a.stage, C)
    D = index()
    if a.queue:
        out = []
        for ds in DSS:
            for v, c in K[ds]:
                for s in SEEDS:
                    if (ds, tag(c), s) not in D:
                        out.append(line(ds, c, s))
        if len(set(out)) != len(out):
            stop('duplicate queue lines')
        open(a.queue, 'w').write(''.join(l + '\n' for l in out))
        print(f'stage {a.stage}: {len(out)} runs -> {a.queue}')
        return
    if not a.select:
        stop('--queue or --select')
    keys = STAGES[NAMES.index(a.stage)][1]
    cur = {ds: tuple(C[ds][k] for k in keys) for ds in DSS}
    sel, scores, info = {}, {}, {}
    for ds in DSS:
        sc = {}
        for v, c in K[ds]:
            rr = three(D, ds, c)
            if rr is None:
                stop(f'{a.stage} {ds} {v}: missing seed rows of {tag(c)}')
            x = [float(r['aucpr']) for r in rr]
            if not all(math.isfinite(t) for t in x) or any(r.get('energy_check') != 'ok' for r in rr):
                stop(f'{a.stage} {ds} {v}: non-finite AUC-PR or energy_check != ok')
            sc[v] = sum(x) / 3
            info[f'{ds} {v}'] = dict(tag=tag(c), suites=sorted({r['_suite'] for r in rr}), aucpr=sc[v],
                                     pa_f1=sum(float(r['pa_f1']) for r in rr) / 3, raw_f1=sum(float(r['raw_f1']) for r in rr) / 3,
                                     syn_energy_uJ=sum(float(r['syn_energy_uJ']) for r in rr) / 3, epochs_run=[int(r['epochs_run']) for r in rr],
                                     time_s=[int(r['time_s']) for r in rr])
        best = max(sc.values())
        pick = cur[ds] if best - sc[cur[ds]] < 1e-9 else next(v for v, _ in K[ds] if sc[v] == best)
        sel[ds] = list(pick); scores[ds] = {json.dumps(list(v)): s for v, s in sc.items()}
    p = os.path.join(ROOT, SUITE, f'selected_{a.stage}.json')
    if os.path.exists(p):
        old = json.load(open(p))
        if old['selected'] != sel:
            stop(f'{p} exists with a different selection {old["selected"]} (new {sel}); not overwritten')
    else:
        json.dump(dict(stage=a.stage, keys=list(keys), selected=sel, scores=scores, current_before=cur), open(p, 'x'), indent=1)
    L = [f'# AD tuning stage {a.stage} ({", ".join(keys)}): selection on seed-mean test AUC-PR (user decision; optimistic)', '',
         '| dataset | value | tag | AUC-PR % | point-adjusted F1 % | raw F1 % | energy µJ (syops) | epochs run | suites |', '|---|---|---|--:|--:|--:|--:|---|---|']
    for ds in DSS:
        for v, c in K[ds]:
            i = info[f'{ds} {v}']
            mark = ' **(selected)**' if tuple(sel[ds]) == v else (' (current)' if v == cur[ds] else '')
            L.append(f"| {ds} | {v}{mark} | {i['tag']} | {100 * i['aucpr']:.3f} | {100 * i['pa_f1']:.2f} | {100 * i['raw_f1']:.2f} | "
                     f"{i['syn_energy_uJ']:.2f} | {i['epochs_run']} | {', '.join(i['suites'])} |")
    L += ['', f'selected_{a.stage}.json: {json.dumps(sel)}']
    open(os.path.join(ROOT, SUITE, f'selection_{a.stage}.md'), 'w').write('\n'.join(L) + '\n')
    json.dump(info, open(os.path.join(ROOT, SUITE, f'selection_{a.stage}_rows.json'), 'w'), indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
