"""Token-order shuffle evaluation (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG order-eval pre-registration; user "(b) 순서 섞기 먼저";
no training). Best-validation checkpoints, full test split, every RESULT row of the chosen arm (4 cells x 3 seeds). Per row of the batch
(one series of one channel) a random permutation of the N tokens is drawn (generator seeded by permutation seed and batch index):
  none   no shuffle; its test MSE must equal the logged one (relative 1e-5), else the script stops (exit 2)
  mixer  the block's mixer input is shuffled along the token axis and its output m is put back in the original order before the
         residual -- only the neo-mixer sees the shuffled order (embedding / MLP / head are per token or position-aligned)
  head   only the forecast head's input tokens are shuffled (body unchanged)
  full   the patch tokens are shuffled before the embedding (mixer and head both see the shuffled order)
Wrapper check (fail closed): on the first run every variant with the identity permutation must give exactly the 'none' MSE.
Each shuffle variant is run with permutation seeds 0, 1, 2 and averaged. Output: one line per run and variant with MSE / MAE and the
change in % against 'none'; --out writes a JSON for the summary.
usage (from cinc_mixer_v1/): CUDA_VISIBLE_DEVICES=<g> python -B results/check-order-20261127/order_eval.py --arm <base|pmux> --out <json>
"""
import argparse
import json
import math
import os
import sys

import torch

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, C)
os.chdir(C)
sys.path.append(os.path.join(C, '..', 'forecasting'))
from spikingjelly.clock_driven import functional                 # noqa: E402
from load_model import LOAD_MODEL                               # noqa: E402
from forecast_wrap import Forecaster                            # noqa: E402
from train_forecast import data_args                            # noqa: E402
from data_provider.data_factory import data_provider            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--arm', choices=['base', 'pmux'], required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
REF = {'ETTh1': 'spkmx_lif_muxlin_iand_lr0.00025_L1_e32_mlp4_wd0.1', 'ETTm1': 'spkmx_lif_muxlin_iand_lr0.0005_L2_pt16_e32_mlp0.5'}
if a.arm == 'base':
    src, pick = 'flif-v1-ett-spkmux-retune', lambda r: r['ds'] in REF and r['tag'] == REF[r['ds']] and r['pl'] in ('96', '720') and r['blocks'] == '1'
else:
    src, pick = 'flif-v1-ett-sxa-s4', lambda r: r['cond'] == 'spkmx_lif_msxdpca_iand'
rows = [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in open(os.path.join(C, 'results', src, 'raw.txt')) if l.startswith('RESULT')]
rows = [r for r in rows if pick(r)]
if len(rows) != 12:
    sys.exit(f'STOP: {len(rows)} rows (want 12)')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
state = {'pseed': 0, 'batch': 0, 'identity': False}


def perm_for(Bp, N, device):
    if state['identity']:
        p = torch.arange(N, device=device).unsqueeze(0).expand(Bp, N).contiguous()
        return p, p
    g = torch.Generator().manual_seed(100003 * state['pseed'] + state['batch'])
    p = torch.argsort(torch.rand(Bp, N, generator=g), dim=1)
    return p.to(device), torch.argsort(p, dim=1).to(device)


def take(x, idx, dim):                                            # per-row permutation along `dim` (row axis = 1)
    shape = [1] * x.dim()
    shape[1], shape[dim] = idx.shape[0], idx.shape[1]
    full = idx.view(*shape).expand(*[x.shape[i] if i not in (1, dim) else (idx.shape[0] if i == 1 else idx.shape[1]) for i in range(x.dim())])
    return torch.gather(x, dim, full)


def evaluate(model, dl, H, variant):
    inner, blk = model.inner, model.inner.blocks[0]
    o_mix, o_tok, o_feat = blk.mix, inner.tokens, model.features
    if variant == 'mixer':
        def mix(x):                                               # x [T, B', N, D]
            p, ip = perm_for(x.shape[1], x.shape[2], x.device)
            return take(o_mix(take(x, p, 2)), ip, 2)
        blk.mix = mix
    elif variant == 'full':
        def tokens(x):
            t = o_tok(x)                                          # [T, B', N, P]
            p, _ = perm_for(t.shape[1], t.shape[2], t.device)
            return take(t, p, 2)
        inner.tokens = tokens
    elif variant == 'head':
        def features(z):
            f = o_feat(z)                                         # [B', N, D]
            p, _ = perm_for(f.shape[0], f.shape[1], f.device)
            return torch.gather(f, 1, p.unsqueeze(-1).expand_as(f))
        model.features = features
    se = ae = cnt = 0.
    try:
        with torch.no_grad():
            for bi, (x, y, _, _) in enumerate(dl):
                state['batch'] = bi
                x = x.float().to(dev); y = y.float().to(dev)[:, -H:, :]
                d = (model(x) - y).double()
                se += float((d ** 2).sum()); ae += float(d.abs().sum()); cnt += d.numel()
                functional.reset_net(model)
    finally:
        for nm in ('mix',):
            blk.__dict__.pop(nm, None)
        inner.__dict__.pop('tokens', None)
        model.__dict__.pop('features', None)
    return se / cnt, ae / cnt


res = []
for r in sorted(rows, key=lambda r: (r['ds'], int(r['pl']), r['seed'])):
    cfg = torch.load(os.path.join(r['dir'], 'model_state', 'config.pt'), map_location='cpu')
    args = argparse.Namespace(**cfg); args.device = dev
    model = Forecaster(LOAD_MODEL[args.model](args, True), args.embed_dim, args.pred_len, bias=args.bias)
    model.load_state_dict(torch.load(os.path.join(r['dir'], 'model_state', 'checkpoint.pth'), map_location='cpu'))
    model = model.to(dev).eval(); model.train_mode = 'testing'
    _, dl = data_provider(data_args(args, args), 'test')
    mse0, mae0 = evaluate(model, dl, args.pred_len, 'none')
    if not (math.isfinite(mse0) and abs(mse0 - float(r['mse'])) <= 1e-5 * float(r['mse'])):
        sys.exit(f'STOP: {r["ds"]} H{r["pl"]} seed {r["seed"]}: recomputed MSE {mse0} != logged {r["mse"]}')
    if not res:                                                   # wrapper check on the first run
        state['identity'] = True
        for var in ('mixer', 'head', 'full'):
            mi, _ = evaluate(model, dl, args.pred_len, var)
            if mi != mse0:
                sys.exit(f'STOP: identity permutation through {var} gives {mi} != {mse0}')
        state['identity'] = False
        print('identity-permutation check passed (mixer, head, full give the unshuffled MSE exactly)', flush=True)
    out = dict(ds=r['ds'], pl=r['pl'], seed=r['seed'], mse=mse0, mae=mae0)
    for var in ('mixer', 'head', 'full'):
        ms, ma = [], []
        for ps in (0, 1, 2):
            state['pseed'] = ps
            m1, m2 = evaluate(model, dl, args.pred_len, var)
            ms.append(m1); ma.append(m2)
        out[var] = dict(mse=sum(ms) / 3, mae=sum(ma) / 3, d_mse_pct=100 * (sum(ms) / 3 - mse0) / mse0, d_mae_pct=100 * (sum(ma) / 3 - mae0) / mae0,
                        mse_by_seed=ms)
    res.append(out)
    print(f'{a.arm} {r["ds"]} H{r["pl"]} seed {r["seed"]}: MSE {mse0:.6f} (logged {float(r["mse"]):.6f}) | ' +
          ' | '.join(f'{v} {out[v]["d_mse_pct"]:+.2f} % MSE, {out[v]["d_mae_pct"]:+.2f} % MAE' for v in ('mixer', 'head', 'full')), flush=True)
    del model
json.dump(dict(arm=a.arm, source=src, runs=res), open(a.out, 'w'), indent=1)
print('done', len(res))
