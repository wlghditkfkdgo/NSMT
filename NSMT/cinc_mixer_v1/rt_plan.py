"""Plan of the ETTh1 / ETTm1 retuning of the LIF SpkMixer (2026-11-25, exp/f-lif-pop-v3; PROJECT_LOG pre-registration of 2026-11-25;
user decision 11-25 "(나)": ETTh1 and ETTm1 choose their settings on their own validation, like ETTh2 / ETTm2 did).
Shared by scripts/make_queue_rt.py (queues) and summarize_rt.py (selections, judgment); recheck_rt.py does NOT import it.

Procedure = the one ETTh2 / ETTm2 went through, applied per dataset to ETTh1 and ETTm1:
  start   the NSMT-paper Table 13 values of the dataset (as ETTh2 / ETTm2 started from theirs; run_ett_mixer.sh blob 8b0a3492):
          ETTh1 lr 5e-4, M 2, patch 8, d_model 32, MLP 1, blocks 1, wd 0.06, 50 epochs
          ETTm1 lr 1e-3, M 2, patch 8, d_model 64, MLP 1, blocks 1, wd 0.06, 50 epochs
          structure fixed: muxlin + IAND (the residual is not re-selected), tau 2, T_h 4, batch 64, patience 10, L1 loss, input 96
  shared  lr {1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3}  ->  M = mixer layers per direction {1, 2, 3}         (11-01 stages S1, S2)
  chain a (first chain, 11-20 00:14):  patch {4, 8, 16} -> d_model {32, 64, 128, 256} -> MLP {0.5, 1, 2, 4} -> blocks {1, 2, 3}
          -> wd {0, 0.01, 0.06, 0.1}
  chain x (extension, 11-20 13:08):    patch {4, 8, 16, 24, 32} -> d_model -> MLP -> blocks -> wd {0, 0.005, 0.01, 0.02, 0.05, 0.06}
  both chains start from the shared stages' winners and run stage by stage together (one queue per stage = the union of the runs
  both chains need, a run never repeated).
Selection (11-01 / 11-20 rule, validation only): per dataset and chain,
    score(v) = mean over H96 / H720 of [seed-mean best validation L1 at v] / [the same at the current value]
the lowest score wins; within 1e-12 of the current value's score the current value is kept; between two other values with the same
score the earlier one in the list. Cells H96 / H720 (as for ETTh2 / ETTm2), seeds 42 / 7 / 2026.
final   each chain's selected configuration at H192 / H336 x 3 seeds (the H96 / H720 rows are the tuning runs, as for ETTh2 / ETTm2).
env RT_DRY=<folder>: selections / raw.txt under <folder>/flif-v1-ett-spkmux-retune, queues under <folder>/queues (dry runs).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = 'flif-v1-ett-spkmux-retune'
DRY = os.environ.get('RT_DRY')
OUT = os.path.join(DRY, SUITE) if DRY else os.path.join(HERE, 'results', SUITE)
QDIR = os.path.join(DRY, 'queues') if DRY else os.path.join(HERE, 'scripts', 'queues')
COND = 'spkmx_lif_muxlin_iand'
DSS, SELPL, FINPL, SEEDS = ('ETTh1', 'ETTm1'), (96, 720), (192, 336), (42, 7, 2026)
EPOCHS = '50'
KN = ('LR', 'LAYERS', 'PATCH', 'EMB', 'MLPR', 'BLOCKS', 'WD')
FIELD = {'LR': 'lr', 'LAYERS': 'layers', 'PATCH': 'patch', 'EMB': 'emb', 'MLPR': 'mlpr', 'BLOCKS': 'blocks', 'WD': 'wd'}
START = {'ETTh1': {'LR': '0.0005', 'LAYERS': '2', 'PATCH': '8', 'EMB': '32', 'MLPR': '1', 'BLOCKS': '1', 'WD': '0.06'},
         'ETTm1': {'LR': '0.001', 'LAYERS': '2', 'PATCH': '8', 'EMB': '64', 'MLPR': '1', 'BLOCKS': '1', 'WD': '0.06'}}
RDEF = {'ETTh1': {'PATCH': '8', 'EMB': '64', 'MLPR': '0.5', 'BLOCKS': '1', 'WD': '0.06'},     # run_ett_spk.sh defaults (tag suffix rule)
        'ETTm1': {'PATCH': '8', 'EMB': '64', 'MLPR': '1', 'BLOCKS': '1', 'WD': '0.06'}}
SUF = {'PATCH': '_pt', 'EMB': '_e', 'MLPR': '_mlp', 'BLOCKS': '_blk', 'WD': '_wd'}
SHARED = (('lr', 'LR', ('0.0001', '0.00025', '0.0005', '0.001', '0.002')), ('layers', 'LAYERS', ('1', '2', '3')))
CHAINS = {'a': (('patch', 'PATCH', ('4', '8', '16')), ('dmodel', 'EMB', ('32', '64', '128', '256')), ('mlp', 'MLPR', ('0.5', '1', '2', '4')),
                ('blocks', 'BLOCKS', ('1', '2', '3')), ('wd', 'WD', ('0', '0.01', '0.06', '0.1'))),
          'x': (('patch', 'PATCH', ('4', '8', '16', '24', '32')), ('dmodel', 'EMB', ('32', '64', '128', '256')),
                ('mlp', 'MLPR', ('0.5', '1', '2', '4')), ('blocks', 'BLOCKS', ('1', '2', '3')),
                ('wd', 'WD', ('0', '0.005', '0.01', '0.02', '0.05', '0.06')))}
CHAIN_NAMES = tuple(s[0] for s in CHAINS['a'])
STAGES = tuple(s[0] for s in SHARED) + CHAIN_NAMES + ('final',)
CNAME = {'a': 'first chain (patch <= 16, wd <= 0.1)', 'x': 'extension (patch <= 32, wd <= 0.06)'}
SLOW = (('ETTm1', 96), ('ETTm1', 720), ('ETTh1', 720), ('ETTh1', 96))          # queue order: slow cells first
SLOWF = (('ETTm1', 336), ('ETTm1', 192), ('ETTh1', 336), ('ETTh1', 192))


def stop(m):
    print(f'STOP: {m}')
    sys.exit(2)


def key(cfg):
    return tuple(cfg[k] for k in KN)


def tag(ds, cfg):
    """the tag run_ett_spk.sh gives this configuration (a knob equal to the runner default adds nothing)."""
    return f'{COND}_lr{cfg["LR"]}_L{cfg["LAYERS"]}' + ''.join(f'{SUF[k]}{cfg[k]}' for k in SUF if cfg[k] != RDEF[ds][k])


def cfgtxt(cfg):
    return ', '.join(f'{k} {cfg[k]}' for k in KN)


def selfile(chain, name):
    return os.path.join(OUT, f'selected_{name}.json' if chain is None else f'selected_{chain}_{name}.json')


def readsel(chain, name, knob, vals):
    p = selfile(chain, name)
    if not os.path.exists(p):
        return None
    s = json.load(open(p))
    if s.get('knob') != knob or s.get('chain') != chain:
        stop(f'{p}: knob {s.get("knob")} chain {s.get("chain")}, want {knob} {chain}')
    for ds in DSS:
        if str(s['selected'].get(ds)) not in vals:
            stop(f'{p}: {ds} selects {s["selected"].get(ds)}, not in {vals}')
    return {ds: str(s['selected'][ds]) for ds in DSS}


def walk():
    """-> (entries, final). entries: every stage reachable with the selections on disk, in order, as dicts
    {name, chain (None for shared), knob, vals, base {ds: configuration before the stage}, selected {ds: value} or None};
    the first stage of a chain without a selection is included (it is the next one to run), later ones are not.
    final: {chain: {ds: configuration}} once every stage of both chains is selected, else None."""
    E = []
    base = {ds: dict(START[ds]) for ds in DSS}
    for name, knob, vals in SHARED:
        s = readsel(None, name, knob, vals)
        E.append(dict(name=name, chain=None, knob=knob, vals=vals, base={ds: dict(base[ds]) for ds in DSS}, selected=s))
        if s is None:
            return E, None
        for ds in DSS:
            base[ds][knob] = s[ds]
    cb = {c: {ds: dict(base[ds]) for ds in DSS} for c in CHAINS}
    open_ = {c: True for c in CHAINS}
    for k in range(len(CHAIN_NAMES)):
        for c in CHAINS:
            if not open_[c]:
                continue
            name, knob, vals = CHAINS[c][k]
            s = readsel(c, name, knob, vals)
            E.append(dict(name=name, chain=c, knob=knob, vals=vals, base={ds: dict(cb[c][ds]) for ds in DSS}, selected=s))
            if s is None:
                open_[c] = False
                continue
            for ds in DSS:
                cb[c][ds][knob] = s[ds]
    return E, (cb if all(open_.values()) else None)


def requested():
    """-> (sel, fin): sets of (ds, configuration key) the reachable stages ask for at H96 / H720 (sel) and at H192 / H336 (fin)."""
    E, F = walk()
    sel = {(ds, key(dict(e['base'][ds], **{e['knob']: v}))) for e in E for ds in DSS for v in e['vals']}
    fin = set() if F is None else {(ds, key(F[c][ds])) for c in CHAINS for ds in DSS}
    return sel, fin
