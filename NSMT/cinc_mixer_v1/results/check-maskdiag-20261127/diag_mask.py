"""Mask diagnostic on trained checkpoints (2026-11-27, exp/f-lif-pop-v3; user question "더하는 것으로 변경하면? spike형태여서 웬만한 칸이 0일
확률이 높아"; no training). For the attention arms of stages 1 / 2 / 3a (best-validation weights, test split, first 20 batches) records:
  zero_frac   share of (b', t, n, n') entries of the spike map A_s = Q K^T that are 0 (no channel fires in both tokens)
  mean_As     mean of A_s;   ge2  share of entries >= 2
  q / v rate  firing rate of Q and V (the forward / reverse spikes, or their projections)
  mask        mean, overall std, mean within-row std (spread over the keys n' of one query n), min / max of the mask M
  on_nonzero  share of the mask total that falls on entries with A_s > 0 (the part a multiplicative mask can act on)
  s_d rate    MUX arms: share of (token, channel) positions where the attention output is used
usage (from cinc_mixer_v1/): CUDA_VISIBLE_DEVICES=<g> python -B results/check-maskdiag-20261127/diag_mask.py
"""
import argparse
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
from sxa import AttnMath                                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--batches', type=int, default=20)
a = ap.parse_args()
ARMS = [('flif-v1-ett-sxa-s1', 'spkmx_lif_sxdpc_add', 'sxdpc ADD (s1)'), ('flif-v1-ett-sxa-s1', 'spkmx_lif_sxppc_add', 'sxppc ADD (s1)'),
        ('flif-v1-ett-sxa-s2', 'spkmx_lif_sxdot_add', 'sxdot ADD (s2, OT)'), ('flif-v1-ett-sxa-s2', 'spkmx_lif_msxdpc_add', 'msxdpc ADD (s2)'),
        ('flif-v1-ett-sxa-s3', 'spkmx_lif_msxdpc_iand', 'msxdpc IAND (s3a)')]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device {dev}, first {a.batches} test batches per run, seeds 42 / 7 / 2026 (independent weights)')
print('arm | cell | zero_frac | mean_As | ge2 | q rate | v rate | mask mean | mask std | row std | min / max | on_nonzero | s_d rate')
for suite, cond, label in ARMS:
    runs = []
    for l in open(os.path.join(C, 'results', suite, 'raw.txt')):
        if l.startswith('RESULT'):
            r = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
            if r['cond'] == cond and not r['tag'].endswith('_ws'):
                runs.append(r)
    for ds, pl in (('ETTh1', '96'), ('ETTh1', '720'), ('ETTm1', '96'), ('ETTm1', '720')):
        acc = dict(z=0., n=0., As=0., ge2=0., q=0., qn=0., v=0., vn=0., mm=0., ms=0., m2=0., mn=0., rs=0., rn=0., on=0., tot=0., sd=0., sdn=0.,
                   mn_=1e9, mx_=-1e9)
        for r in [r for r in runs if r['ds'] == ds and r['pl'] == pl]:
            run = r['dir']
            cfg = torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')
            args = argparse.Namespace(**cfg); args.device = dev
            model = Forecaster(LOAD_MODEL[args.model](args, True), args.embed_dim, args.pred_len, bias=args.bias)
            model.load_state_dict(torch.load(os.path.join(run, 'model_state', 'checkpoint.pth'), map_location='cpu'))
            model = model.to(dev).eval(); model.train_mode = 'testing'
            blk = model.inner.blocks[0]

            def hk(m, i, o):
                q, k, v = i[0], i[1], i[2]
                As = torch.einsum('nbtd,mbtd->btnm', q, k)
                acc['z'] += float((As == 0).sum()); acc['n'] += As.numel(); acc['As'] += float(As.sum()); acc['ge2'] += float((As >= 2).sum())
                acc['q'] += float(q.count_nonzero()); acc['qn'] += q.numel(); acc['v'] += float(v.count_nonzero()); acc['vn'] += v.numel()
                M = m.last_mask                                                    # [B', N, N]
                Mb = M.unsqueeze(1).expand_as(As)
                acc['mm'] += float(M.sum()); acc['m2'] += float((M * M).sum()); acc['mn'] += M.numel()
                acc['rs'] += float(M.std(-1, unbiased=False).sum()); acc['rn'] += M.shape[0] * M.shape[1]
                acc['on'] += float((Mb * (As > 0)).sum()); acc['tot'] += float(Mb.sum())
                acc['mn_'] = min(acc['mn_'], float(M.min())); acc['mx_'] = max(acc['mx_'], float(M.max()))
            hs = [blk.xattn.core.register_forward_hook(hk)]
            if hasattr(blk, 'sd_neuron') and cond.startswith('spkmx_lif_msx'):
                hs.append(blk.sd_neuron.register_forward_hook(lambda m, i, o: acc.update(sd=acc['sd'] + float(o.sum()), sdn=acc['sdn'] + o.numel())))
            _, dl = data_provider(data_args(args, args), 'test')
            with torch.no_grad():
                for bi, batch in enumerate(dl):
                    if bi >= a.batches:
                        break
                    model(batch[0].float().to(dev)); functional.reset_net(model)
            for h in hs:
                h.remove()
            del model
        mean = acc['mm'] / acc['mn']
        std = (acc['m2'] / acc['mn'] - mean ** 2) ** .5
        sd = f'{acc["sd"] / acc["sdn"]:.3f}' if acc['sdn'] else '—'
        print(f'{label} | {ds} H{pl} | {acc["z"] / acc["n"]:.3f} | {acc["As"] / acc["n"]:.3f} | {acc["ge2"] / acc["n"]:.3f} | '
              f'{acc["q"] / acc["qn"]:.3f} | {acc["v"] / acc["vn"]:.3f} | {mean:.3f} | {std:.3f} | {acc["rs"] / acc["rn"]:.3f} | '
              f'{acc["mn_"]:.3f} / {acc["mx_"]:.3f} | {acc["on"] / acc["tot"]:.3f} | {sd}', flush=True)
