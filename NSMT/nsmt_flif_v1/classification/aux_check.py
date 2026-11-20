"""MODI 2026-11-20 (aux-target memory check, user decision 11-20 "4-1 검증(작음) -> 범위 1 -> 4-2 (가) + (나)"): does a reconstruction
aux target that needs the past actually need the Neocortex's memory? Pre-registered in docs/MODI_PROJECT_LOG.md (2026-11-20 entry).

Runs (72, seed 42, no relative position bias): the all-SpikeDE base with the raw-patch aux in three Neocortex variants --
memory = nsmt4q_pop_<lif|flif>_msn_sde_auxf (normal), no memory = ..._sde_nomem_auxf (every token through the Neocortex alone, 1 step),
shuffled = ..._sde_nshuf_auxf (the Neocortex reads the tokens in a shuffled order, outputs put back: memory and firing as usual, but the
history is not the ordered past; evaluation uses the fixed permutation of token_order.TokenShuffle); aux target of the Neocortex output of token i = token i - s, s in {0 (the token itself), 2, 4 (past patches), -2 (the next
non-overlapping patch)}; ETTh2 H96 / ETTm2 H96 (alpha 0.8 / 0.5 -> r = 0.107 / 0.067) and MIT-BIH with the 80 / 20 validation split
(VAL=1, loss 0.9 CE + 0.1 MSE). Suites flif-v1-auxck-<ett|mb>-<s0|s2|s4|sm2>.

Measure (validation split, the tested weights: ETT best-validation checkpoint.pth, MIT-BIH last epoch best+model.pt, eval mode):
  E = mean squared error of the trained weak_decoder's reconstruction against its shifted target over all valid elements
  Z = mean squared deviation of that target from its per-(token position, element) mean over the split
  e = E / Z = the share of the target's variance left unexplained (0 = perfect, 1 = no better than the position-wise average)
Rule (per task, per neuron; s in {2, 4, -2}; control c in {no memory, shuffled}):
  g_c(s) = e(c, s) - e(memory, s);   d_c(s) = g_c(s) - g_c(0)   (the own-token gap calibrates what the control costs anyway)
  ETT: d and e are the means of the two cells, and d must be > 0 in both cells
  a control is valid when e(c, 0) < 0.90 (ETT: in both cells), i.e. it can at least reconstruct its own token
  deciding control = no memory if valid, else shuffled if valid, else "no verdict"
  "needs memory"  d(s) >= 0.10 and e(memory, s) <= 0.90 against the deciding control (both controls are always reported)
  a target passes on a task when it "needs memory" for pop LIF and pop f-LIF; best = the largest mean d of the passing targets.
  The main aux test is NOT started from this check (user decision).
Report only: data-level linear baselines (ridge, fitted on the training split, scored on the validation split; no trained model):
  e_lin(cur) predicts the target from token i alone, e_lin(hist) from tokens i, i-1, ..., i-N+1 (zero-padded); forecast / class
  metrics, Neocortex firing rate, energy.
Checks (fail-closed): 72 RESULT lines, no SKIP, unique keys, one sha, pe=0, auxs field / _auxs<s> folder suffix / config aux_shift
agree, _nomem / neo_memoryless and _nshuf / neo_shuffle agree, ssa_pe 0, keep_ratio 0, aux on; the weights load strictly; ETT validation MSE reproduces the
logged best-epoch val_mse (|diff| <= 2e-6), MIT-BIH validation macro-F1 reproduces the logged last-epoch val_f1 (<= 2e-6) and E is
within 1 % of the logged batch-averaged val_mse (the loader order is random); every value finite, Z > 0; the baseline tokens equal the
model's own aux targets.
usage (classification folder):
  python aux_check.py --measure      -> results/flif-v1-auxck/rows.jsonl (refuses an existing file)
  python aux_check.py --baselines    -> results/flif-v1-auxck/baselines.json
  python aux_check.py --summarize    -> results/flif-v1-auxck.{json,md}
"""
import argparse
import json
import math
import os
import sys
from types import SimpleNamespace as NS

import torch

sys.path.insert(0, os.getcwd())
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', 'forecasting'))
from spikingjelly.clock_driven import functional

RES = os.path.join(HERE, 'results')
FC = os.path.join(HERE, '..', 'forecasting', 'results')
OUTD = os.path.join(RES, 'flif-v1-auxck')
TARGETS = (('s0', 0), ('s2', 2), ('s4', 4), ('sm2', -2))
NEURONS = ('lif', 'flif')
VARIANTS = ('mem', 'nomem', 'shuf')
VSUF = {'mem': '', 'nomem': '_nomem', 'shuf': '_nshuf'}
VNAME = {'mem': 'memory', 'nomem': 'no memory', 'shuf': 'shuffled'}
CELLS = (('ETTh2', 96), ('ETTm2', 96))
SEED = 42
TOL = 2e-6
D_MIN, E_MAX, E_CTRL = 0.10, 0.90, 0.90
NAMES = {'lif': 'pop LIF', 'flif': 'pop f-LIF'}
TNAME = {0: 'token itself (s = 0)', 2: 'past, 2 tokens back (s = 2)', 4: 'past, 4 tokens back (s = 4)', -2: 'next patch (s = -2)'}


def cond(n, v):
    return f'nsmt4q_pop_{n}_msn_sde{VSUF[v]}_auxf'


def suite_path(task, tn):
    return os.path.join(FC if task == 'ett' else RES, f'flif-v1-auxck-{task}-{tn}', 'raw.txt')


def finite(*v):
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)


def rows():
    """[(task, ds, pl, n, v, s, RESULT dict)] in a fixed order; every check of the docstring on the RESULT lines."""
    out, shas = [], set()
    for task in ('ett', 'mb'):
        for tn, s in TARGETS:
            p = suite_path(task, tn)
            lines = open(p).read().splitlines()
            if any(l.startswith('SKIP') for l in lines):
                raise RuntimeError(f'{p}: SKIP lines')
            got = {}
            for ln in lines:
                if not ln.startswith('RESULT'):
                    continue
                d = dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t)
                k = (d['cond'], d.get('ds'), int(d['pl']) if 'pl' in d else None, int(d['seed']))
                if k in got:
                    raise RuntimeError(f'{p}: duplicate {k}')
                got[k] = d
            keys = [(cond(n, v), ds, pl, SEED) for n in NEURONS for v in VARIANTS
                    for (ds, pl) in (CELLS if task == 'ett' else ((None, None),))]
            if sorted(got, key=str) != sorted(keys, key=str):
                raise RuntimeError(f'{p}: rows {sorted(got, key=str)} != expected {sorted(keys, key=str)}')
            for k in keys:
                d = got[k]
                n = 'flif' if '_flif_' in k[0] else 'lif'
                v = 'nomem' if '_nomem_' in k[0] else 'shuf' if '_nshuf_' in k[0] else 'mem'
                base = os.path.basename(d['dir'])
                if d.get('pe') != '0' or '_nope' not in base:
                    raise RuntimeError(f'{p} {k}: not bias-off (pe={d.get("pe")}, {base})')
                if (d.get('auxs') if s else None) != (str(s) if s else None) or ('auxs' in d) != (s != 0):
                    raise RuntimeError(f'{p} {k}: auxs field {d.get("auxs")} for s={s}')
                if base.endswith(f'_auxs{s}') != (s != 0) or ('_auxs' in base) != (s != 0):
                    raise RuntimeError(f'{p} {k}: folder {base} for s={s}')
                if ('_nomem' in base) != (v == 'nomem') or ('_nshuf' in base) != (v == 'shuf'):
                    raise RuntimeError(f'{p} {k}: folder {base} for variant {v}')
                if task == 'mb' and ('_mbval' not in base or d.get('val') != '1'):
                    raise RuntimeError(f'{p} {k}: MIT-BIH without the validation split')
                shas.add(d['sha'])
                out.append((task, k[1], k[2], n, v, s, d))
    if len(out) != 72 or len(shas) != 1:
        raise RuntimeError(f'{len(out)} rows, shas {shas}')
    return out


def build(task, d, s, v):
    from load_model import LOAD_MODEL
    cfg = torch.load(os.path.join(d, 'model_state', 'config.pt'), map_location='cpu')
    a = NS(**dict(cfg, device='cuda'))
    bad = []
    if int(getattr(a, 'aux_shift', 0)) != s: bad.append(f'aux_shift {getattr(a, "aux_shift", None)}')
    if bool(getattr(a, 'neo_memoryless', False)) != (v == 'nomem'): bad.append('neo_memoryless')
    if bool(getattr(a, 'neo_shuffle', False)) != (v == 'shuf'): bad.append('neo_shuffle')
    if int(a.ssa_pe) != 0: bad.append(f'ssa_pe {a.ssa_pe}')
    if float(a.keep_ratio) != 0.: bad.append(f'keep_ratio {a.keep_ratio}')
    if task == 'ett' and not float(getattr(a, 'aux_alpha', 0.)) > 0: bad.append('aux_alpha')
    if task == 'mb' and (float(a.alpha) != 0.1 or not getattr(a, 'mitbih_val', False)): bad.append(f'alpha {a.alpha} / mitbih_val')
    if bad:
        raise RuntimeError(f'{d}: config {bad}')
    torch.manual_seed(0)
    body = LOAD_MODEL[a.model](a, True)
    if task == 'ett':
        from forecast_wrap import Forecaster
        m = Forecaster(body, a.embed_dim, a.pred_len, bias=a.bias)
        m.load_state_dict(torch.load(os.path.join(d, 'model_state', 'checkpoint.pth'), map_location='cpu'))   # strict
    else:
        m = body
        m.load_state_dict(torch.load(os.path.join(d, 'model_state', 'best+model.pt'), map_location='cpu'))     # strict
    return m.cuda().eval(), body, a


class Acc:
    """Squared error and the per-(position, element) moments of the target (float64)."""
    def __init__(self):
        self.sse = 0.; self.n = 0; self.s1 = None; self.s2 = None; self.cnt = 0

    def add(self, rec, org):                       # [N', B, 1, P]
        r = rec.double(); o = org.double()
        self.sse += ((r - o) ** 2).sum().item(); self.n += o.numel()
        o = o[:, :, 0].transpose(0, 1)             # [B, N', P]
        if self.s1 is None:
            self.s1 = torch.zeros_like(o[0]); self.s2 = torch.zeros_like(o[0])
        self.s1 += o.sum(0); self.s2 += (o ** 2).sum(0); self.cnt += o.shape[0]

    def result(self):
        mu = self.s1 / self.cnt
        Z = (self.s2 / self.cnt - mu ** 2).mean().item()
        E = self.sse / self.n
        if not finite(E, Z) or not Z > 0:
            raise RuntimeError(f'E {E} Z {Z}')
        return E, Z


def ett_split(a, flag, shuffle):
    from torch.utils.data import DataLoader
    from train_forecast import data_args
    from data_provider.data_factory import data_provider
    ds = data_provider(data_args(a, a), flag)[0]
    return DataLoader(ds, batch_size=a.batch_size, shuffle=shuffle, num_workers=2, drop_last=False)


def mb_split(a, which):
    from dataloader import create_loader
    (tr, va), _ = create_loader(train=True, batch_size=a.batch_size, num_classes=a.num_classes, data_root=a.root_path + '/5class_mitbih_train_7_187.csv',
                                train_val_ratio=[4, 1], data_window_size=187, sampling='None', num_workers=4)
    return tr if which == 'train' else va


@torch.no_grad()
def measure_ett(m, body, a, d):
    acc = Acc(); se = 0.; cnt = 0
    m.aux = True
    functional.reset_net(m)
    for x, y, _, _ in ett_split(a, 'val', False):
        x = x.float().cuda(); y = y.float().cuda()[:, -a.pred_len:, :]
        out = m(x)
        rec, org = body.aux_pair(body.weak_decoder(m._x_neo), body.org_x)
        m._x_neo = None
        acc.add(rec, org)
        se += ((out - y).double() ** 2).sum().item(); cnt += out.numel()
        functional.reset_net(m)
    E, Z = acc.result()
    val_mse = se / cnt
    be = int(open(os.path.join(d, 'log', 'best_epoch.txt')).read().split()[0])
    import csv
    logged = [r for r in csv.DictReader(open(os.path.join(d, 'log', 'best_log_0.csv'))) if int(r['epoch']) == be]
    if len(logged) != 1 or not abs(val_mse - float(logged[0]['val_mse'])) <= TOL:
        raise RuntimeError(f'{d}: validation MSE {val_mse:.7f} does not reproduce the logged best epoch {be}: {logged}')
    return dict(E=E, Z=Z, e=E / Z, val_mse=val_mse)


@torch.no_grad()
def measure_mb(m, a, d):
    from torcheval.metrics import MulticlassF1Score
    acc = Acc(); bm = []
    f1 = MulticlassF1Score(num_classes=a.num_classes, average='macro').to('cuda')
    m.train_mode = 'training'                      # as train.val_one_epoch: eval mode, the training-mode tuple
    functional.reset_net(m)
    for batch in mb_split(a, 'val'):
        x = batch[0].float().cuda(); lab = batch[1].long().squeeze(-1).cuda()
        z, _, _, org_x, rec_x = m(x)
        functional.reset_net(m)
        acc.add(rec_x, org_x); bm.append(((org_x - rec_x) ** 2).mean().item())
        f1.update(z.argmax(-1), lab)
    E, Z = acc.result()
    import csv
    last = list(csv.DictReader(open(os.path.join(d, 'log', 'best_log_0.csv'))))[-1]
    vf1 = f1.compute().item()
    if not abs(vf1 - float(last['val_f1'])) <= TOL:
        raise RuntimeError(f'{d}: validation macro-F1 {vf1:.7f} != logged {last["val_f1"]}')
    if not abs(sum(bm) / len(bm) / float(last['val_mse']) - 1) <= 0.01:
        raise RuntimeError(f'{d}: aux MSE {sum(bm) / len(bm):.6f} far from the logged val_mse {last["val_mse"]}')
    return dict(E=E, Z=Z, e=E / Z, val_f1=vf1)


def do_measure():
    os.makedirs(OUTD, exist_ok=True)
    path = os.path.join(OUTD, 'rows.jsonl')
    if os.path.exists(path):
        raise SystemExit(f'{path} exists')
    rs = rows()
    with open(path, 'w') as fo:
        for task, ds, pl, n, v, s, d in rs:
            m, body, a = build(task, d['dir'], s, v)
            r = measure_ett(m, body, a, d['dir']) if task == 'ett' else measure_mb(m, a, d['dir'])
            row = dict(task=task, ds=ds, pl=pl, neuron=n, variant=v, s=s, dir=d['dir'],
                       test=float(d['mse']) if task == 'ett' else float(d['f1']),
                       rho_neo=float(d['rho_neo']) if 'rho_neo' in d else None,
                       energy=float(d['syn_energy_uJ']) if task == 'ett' else float(d['total_direct_uJ']), **r)
            if not finite(*[x for x in r.values()]):
                raise RuntimeError(f'non-finite {row}')
            print(json.dumps(row), flush=True)
            fo.write(json.dumps(row) + '\n')
            del m, body
            torch.cuda.empty_cache()


def tokens_ett(a, x, body):
    """Window-normalised rows -> the aux tokens [N, B', 1, P] (Forecaster.forward + myModel._tokens_mc)."""
    B, L, C = x.shape
    mu = x.mean(1, keepdim=True); z = x - mu
    sd = torch.sqrt(torch.var(z, dim=1, keepdim=True, unbiased=False) + 1e-5)
    z = (z / sd).permute(0, 2, 1).reshape(B * C, 1, L)
    body._tokens_mc(z)
    return body.org_x


def feats(tok, s, hist):
    """tok [N, B, 1, P] -> (X [rows, F], Y [rows, P]) over the valid positions of target s."""
    N, B, _, P = tok.shape
    t = tok[:, :, 0].double()                                          # [N, B, P]
    lo, hi = max(0, s), N + min(0, s)                                  # output positions i with a target token i - s
    if hist:
        pad = torch.cat([torch.zeros(N - 1, B, P, dtype=t.dtype, device=t.device), t], 0)   # [2N - 1, B, P]
        X = torch.stack([pad[N - 1 + i - torch.arange(N, device=t.device)] for i in range(lo, hi)], 0)   # [n, N, B, P]
        X = X.permute(0, 2, 1, 3).reshape((hi - lo) * B, N * P)
    else:
        X = t[lo:hi].reshape((hi - lo) * B, P)
    Y = t[lo - s:hi - s].reshape((hi - lo) * B, P)
    return torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)], 1), Y


@torch.no_grad()
def do_baselines():
    from load_model import LOAD_MODEL
    os.makedirs(OUTD, exist_ok=True)
    rs = rows()
    out = []
    for task, cell in [('ett', c) for c in CELLS] + [('mb', (None, None))]:
        d = next(r[6] for r in rs if r[0] == task and (r[1], r[2]) == cell and r[3] == 'lif' and r[4] == 'mem' and r[5] == 0)
        m, body, a = build(task, d['dir'], 0, 'mem')
        body.train_mode = 'testing'

        def batches(flag):
            if task == 'ett':
                for x, _, _, _ in ett_split(a, flag, False):
                    yield tokens_ett(a, x.float().cuda(), body)
            else:
                for b in mb_split(a, flag):
                    body._tokens_mc(b[0].float().cuda().unsqueeze(1) if b[0].dim() == 2 else b[0].float().cuda())
                    yield body.org_x

        # the tokens are the model's own aux targets: one forward of the trained model on the first validation batch
        if task == 'ett':
            x0 = next(iter(ett_split(a, 'val', False)))[0].float().cuda()
            t_mine = tokens_ett(a, x0, body).clone()
            m.aux = True; functional.reset_net(m); m(x0); functional.reset_net(m); m._x_neo = None
        else:
            b0 = next(iter(mb_split(a, 'val')))[0].float().cuda()
            body._tokens_mc(b0.unsqueeze(1) if b0.dim() == 2 else b0); t_mine = body.org_x.clone()
            m.train_mode = 'training'; functional.reset_net(m); m(b0); functional.reset_net(m)
        if not torch.equal(t_mine, body.org_x):
            raise RuntimeError(f'{task} {cell}: baseline tokens differ from the model aux target')
        for _, s in TARGETS:
            for hist in (False, True):
                G = H = None
                for tok in batches('train'):
                    X, Y = feats(tok, s, hist)
                    G = X.T @ X if G is None else G + X.T @ X
                    H = X.T @ Y if H is None else H + X.T @ Y
                lam = 1e-4 * torch.diagonal(G).mean()
                W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype, device=G.device), H)
                acc = Acc()
                for tok in batches('val'):
                    X, Y = feats(tok, s, hist)
                    pr = (X @ W).float()
                    nn_ = Y.shape[0] // tok.shape[1]
                    acc.add(pr.reshape(nn_, tok.shape[1], 1, -1), Y.float().reshape(nn_, tok.shape[1], 1, -1))
                E, Z = acc.result()
                row = dict(task=task, ds=cell[0], pl=cell[1], s=s, inputs='history' if hist else 'current', e=E / Z, E=E, Z=Z)
                print(json.dumps(row), flush=True)
                out.append(row)
        del m, body
        torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(OUTD, 'baselines.json'), 'w'), indent=1)


def do_summarize():
    R = [json.loads(l) for l in open(os.path.join(OUTD, 'rows.jsonl'))]
    BL = json.load(open(os.path.join(OUTD, 'baselines.json')))
    if len(R) != 72 or len({(r['task'], r['ds'], r['neuron'], r['variant'], r['s']) for r in R}) != 72:
        raise RuntimeError(f'{len(R)} measured rows')
    if len(BL) != 3 * 4 * 2:
        raise RuntimeError(f'{len(BL)} baseline rows')
    dirs = {r[6]['dir'] for r in rows()}
    if {r['dir'] for r in R} != dirs:
        raise RuntimeError('measured rows do not match the RESULT lines')
    for r in R + BL:
        if not finite(r['e'], r['E'], r['Z']):
            raise RuntimeError(f'non-finite {r}')

    def e(task, ds, n, v, s):
        g = [r['e'] for r in R if r['task'] == task and r['ds'] == ds and r['neuron'] == n and r['variant'] == v and r['s'] == s]
        if len(g) != 1:
            raise RuntimeError((task, ds, n, v, s, g))
        return g[0]

    cells = {'ett': [c[0] for c in CELLS], 'mb': [None]}
    verdict = {}
    for task in ('ett', 'mb'):
        for n in NEURONS:
            ctl = {c: max(e(task, ds, n, c, 0) for ds in cells[task]) for c in ('nomem', 'shuf')}
            valid = {c: ctl[c] < E_CTRL for c in ctl}
            decide = 'nomem' if valid['nomem'] else 'shuf' if valid['shuf'] else None
            for _, s in TARGETS[1:]:
                em = sum(e(task, ds, n, 'mem', s) for ds in cells[task]) / len(cells[task])
                per = {}
                for c in ('nomem', 'shuf'):
                    ds_ = [(e(task, ds, n, c, s) - e(task, ds, n, 'mem', s)) - (e(task, ds, n, c, 0) - e(task, ds, n, 'mem', 0))
                           for ds in cells[task]]
                    dm = sum(ds_) / len(ds_)
                    v = ('no verdict' if not valid[c] else 'needs memory' if (dm >= D_MIN and em <= E_MAX and all(x > 0 for x in ds_))
                         else 'does not need memory')
                    per[c] = dict(d=dm, d_cells=ds_, ctrl=ctl[c], verdict=v)
                verdict[(task, n, s)] = dict(e_mem=em, per=per, decide=decide,
                                             verdict=per[decide]['verdict'] if decide else 'no verdict',
                                             d=per[decide]['d'] if decide else float('nan'))
    passed = {task: [s for _, s in TARGETS[1:] if all(verdict[(task, n, s)]['verdict'] == 'needs memory' for n in NEURONS)]
              for task in ('ett', 'mb')}
    best = {task: (max(passed[task], key=lambda s: sum(verdict[(task, n, s)]['d'] for n in NEURONS)) if passed[task] else None)
            for task in ('ett', 'mb')}
    json.dump(dict(rows=R, baselines=BL, verdict={f'{k[0]} {k[1]} {k[2]}': v for k, v in verdict.items()}, passed=passed, best=best),
              open(os.path.join(RES, 'flif-v1-auxck.json'), 'w'), indent=1)

    L = ['# Aux-target memory check (pre-registered, seed 42, no relative position bias) — summary', '',
         'See the docstring of aux_check.py. e = share of the aux target\'s variance left unexplained on the validation split '
         '(0 = perfect, 1 = no better than the position-wise average). memory = normal Neocortex, no memory = every token alone, '
         'shuffled = the Neocortex reads the tokens in a shuffled order.', '',
         '## e per run', '',
         '| task | neuron | target | e memory | e no memory | e shuffled |', '|---|---|---|---|---|---|']
    for task in ('ett', 'mb'):
        for ds in cells[task]:
            for n in NEURONS:
                for _, s in TARGETS:
                    L.append(f'| {ds or "MIT-BIH"} | {NAMES[n]} | {TNAME[s]} | ' + ' | '.join(f'{e(task, ds, n, v, s):.4f}' for v in VARIANTS) + ' |')
    L += ['', '## verdict (d = g(s) − g(0); needs memory: d ≥ 0.10 and e memory ≤ 0.90; ETT: mean of the two cells, d > 0 in both; '
          'deciding control = no memory if it can reconstruct its own token (e < 0.90), else shuffled)', '',
          '| task | neuron | target | e memory | no memory: d (per cell) | no memory: own-token e | shuffled: d (per cell) | shuffled: own-token e | deciding | verdict |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for task in ('ett', 'mb'):
        for n in NEURONS:
            for _, s in TARGETS[1:]:
                v = verdict[(task, n, s)]
                cols = []
                for c in ('nomem', 'shuf'):
                    q = v['per'][c]
                    cols += [f"{q['d']:+.4f} (" + ' / '.join(f'{x:+.4f}' for x in q['d_cells']) + f") {q['verdict']}", f"{q['ctrl']:.4f}"]
                L.append(f'| {"ETT" if task == "ett" else "MIT-BIH"} | {NAMES[n]} | {TNAME[s]} | {v["e_mem"]:.4f} | ' + ' | '.join(cols)
                         + f' | {VNAME[v["decide"]] if v["decide"] else "—"} | {v["verdict"]} |')
    L += ['', '| task | targets that pass (both neurons) | best |', '|---|---|---|']
    for task in ('ett', 'mb'):
        L.append(f'| {"ETT" if task == "ett" else "MIT-BIH"} | {", ".join(TNAME[s] for s in passed[task]) or "none"} | '
                 f'{TNAME[best[task]] if best[task] is not None else "—"} |')
    L += ['', '## report only: linear baselines on the data (ridge, train -> validation; no trained model)', '',
          '| task | target | e_lin from token i alone | e_lin from tokens i, i-1, ... (history) |', '|---|---|---|---|']
    for task in ('ett', 'mb'):
        for ds in cells[task]:
            for _, s in TARGETS:
                g = {b['inputs']: b['e'] for b in BL if b['task'] == task and b['ds'] == ds and b['s'] == s}
                L.append(f'| {ds or "MIT-BIH"} | {TNAME[s]} | {g["current"]:.4f} | {g["history"]:.4f} |')
    L += ['', '## report only: task metric, Neocortex firing, energy (seed 42; not judged)', '',
          '| task | neuron | variant | target | validation | test | rho_neo | energy µJ |', '|---|---|---|---|---|---|---|---|']
    for r in R:
        val = r.get('val_mse', r.get('val_f1'))
        L.append(f'| {r["ds"] or "MIT-BIH"} | {NAMES[r["neuron"]]} | {VNAME[r["variant"]]} | s = {r["s"]} | '
                 f'{val:.4f} | {r["test"]:.4f} | {r["rho_neo"] if r["rho_neo"] is not None else "—"} | {r["energy"]:.2f} |')
    open(os.path.join(RES, 'flif-v1-auxck.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('all checks passed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--measure', action='store_true')
    g.add_argument('--baselines', action='store_true')
    g.add_argument('--summarize', action='store_true')
    o = ap.parse_args()
    do_measure() if o.measure else do_baselines() if o.baselines else do_summarize()
