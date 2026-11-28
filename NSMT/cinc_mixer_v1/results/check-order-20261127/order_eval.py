"""Token-order shuffle evaluation (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG order-eval pre-registration; user "(b) 순서 섞기 먼저";
no training). Best-validation checkpoints, full test split, every RESULT row of the chosen arm (4 cells x 3 seeds). Per row of the batch
(one series of one channel) a random permutation of the N tokens is drawn (generator seeded by permutation seed and batch index):
  none   no shuffle; its test MSE must equal the logged one (relative 1e-5), else the script stops (exit 2)
  mixer  the block's mixer input is shuffled along the token axis and its output m is put back in the original order before the
         residual -- only the neo-mixer sees the shuffled order (embedding / MLP / head are per token or position-aligned)
  head   only the forecast head's input tokens are shuffled (body unchanged)
  full   the patch tokens are shuffled before the embedding (mixer and head both see the shuffled order)
Wrapper check (fail closed): on the first run every variant with the identity permutation must give exactly the 'none' MSE.
2026-11-27 extension (user "attention만 있는 구조(Spikformer)에서도 attention 입력에서 token의 순서를 섞어서"): arms 'sfde' (Spikformer-style
SF-de, SSA with relative position bias, patch 16, ETTh2 / ETTm2 x H96 / H720 x 3 seeds, flif-v1-ett-sfde-pt16) and 'base2' (the
first-chain muxlin IAND on the same cells, patch 16, flif-v1-ett-spkmux-ettune). For the Spikformer-style model the 'mixer' variant
shuffles the SSA input and puts the SSA output back in order (the token-mixing module only), and 'full' shuffles the patch tokens.
2026-11-27 arm 'sfde1': SF-de trained on ETTh1 / ETTm1 (suite flif-v1-ett-sfde-h1m1; user "SF-de에 대해서 ETTh1, ETTm1의 순서 섞기").
2026-11-28 arm 'sfde1n': the same SF-de with --ssa_pe 0 (suite flif-v1-ett-sfde-nope; user "SF-de의 위치 정보 끄기").
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
ap.add_argument('--arm', choices=['base', 'pmux', 'sfde', 'base2', 'sfde1', 'sfde1n'], required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
REF = {'ETTh1': 'spkmx_lif_muxlin_iand_lr0.00025_L1_e32_mlp4_wd0.1', 'ETTm1': 'spkmx_lif_muxlin_iand_lr0.0005_L2_pt16_e32_mlp0.5'}
if a.arm == 'base':
    src, pick = 'flif-v1-ett-spkmux-retune', lambda r: r['ds'] in REF and r['tag'] == REF[r['ds']] and r['pl'] in ('96', '720') and r['blocks'] == '1'
elif a.arm == 'pmux':
    src, pick = 'flif-v1-ett-sxa-s4', lambda r: r['cond'] == 'spkmx_lif_msxdpca_iand'
elif a.arm == 'sfde1':
    src, pick = 'flif-v1-ett-sfde-h1m1', lambda r: r['cond'] == 'spk_ssa' and r['pl'] in ('96', '720')
elif a.arm == 'sfde1n':                                           # 2026-11-28: SF-de with --ssa_pe 0 (user "SF-de의 위치 정보 끄기")
    src, pick = 'flif-v1-ett-sfde-nope', lambda r: r['cond'] == 'spk_ssa' and r['tag'].endswith('_pe0') and r['pl'] in ('96', '720')
elif a.arm == 'sfde':
    src, pick = 'flif-v1-ett-sfde-pt16', lambda r: r['cond'] == 'spk_ssa' and r['pl'] in ('96', '720')
else:
    T2 = {'ETTh2': 'spkmx_lif_muxlin_iand_lr0.0005_L1_pt16_wd0.1', 'ETTm2': 'spkmx_lif_muxlin_iand_lr0.0001_L1_pt16_mlp2_wd0.1'}
    src, pick = 'flif-v1-ett-spkmux-ettune', lambda r: r['ds'] in T2 and r['tag'] == T2[r['ds']] and r['pl'] in ('96', '720')
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
    inner = model.inner
    ssa = model.kind == 'ssa'                                     # Spikformer-style: the token mixer is the block's SSA
    blk = inner.data_block[0] if ssa else inner.blocks[0]
    mod = blk.ssa if ssa else blk
    o_mix = mod.forward if ssa else blk.mix
    o_tok = inner._tokens_mc if ssa else inner.tokens
    o_feat = model.features
    if variant == 'mixer':
        def mix(x, *args, **kw):                                  # x [T, B', N, D]; output put back in the original order
            p, ip = perm_for(x.shape[1], x.shape[2], x.device)
            return take(o_mix(take(x, p, 2), *args, **kw), ip, 2)
        if ssa:
            mod.forward = mix
        else:
            blk.mix = mix
    elif variant == 'full':
        def tokens(x):
            t = o_tok(x)                                          # [T, B', N, P] (Spikformer-style: (patched, neo_in))
            pt = t[0] if ssa else t
            p, _ = perm_for(pt.shape[1], pt.shape[2], pt.device)
            return (take(pt, p, 2), t[1]) if ssa else take(pt, p, 2)
        if ssa:
            inner._tokens_mc = tokens
        else:
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
        mod.__dict__.pop('forward', None) if ssa else None
        inner.__dict__.pop('tokens', None)
        inner.__dict__.pop('_tokens_mc', None)
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
