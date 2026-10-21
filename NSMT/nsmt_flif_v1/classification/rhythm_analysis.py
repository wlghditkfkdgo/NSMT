"""Stage 9 analysis (report only; MODI_PROJECT_LOG 2026-10-19): how far does information travel along the token axis?

For one trained run (best-validation weights, model_state/checkpoint.pth) and up to --n evaluation units of the test set:
replace input token j of the token-mixing network (NSMT neocortex / Neo-mixer first block) by token j of another sample and
measure the share of that network's output spikes that change at every token t. Averaged by lag d = t - j (d > 0: later
tokens; d < 0: earlier tokens, non-zero only for bidirectional mixers). Also the network's mean firing rate.
Not run for spk_ssa (no token-mixing network on the aligned axis) and mix_none (none).
usage: python rhythm_analysis.py --run_dir <dir> --name <cond>_seed<s> --out_dir <dir>
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spikingjelly.clock_driven import functional          # noqa: E402
from load_model import LOAD_MODEL                         # noqa: E402
import rhythm_data as rd                                   # noqa: E402


@torch.no_grad()
def net_out(m, x, j=None, perm=None):
    """token-mixing network output [N, B, D] for input x [B, C, L]; optionally token j replaced by sample perm's token j."""
    if hasattr(m, 'forward_parts'):                        # Neo-mixer / stream model (BiX: mix() = cross-attention output)
        tok = m.tokens(x)
        if j is not None:
            tok = tok.clone(); tok[j] = tok[j][perm]
        o = m.blocks[0].mix(m.encoding(tok))
    else:                                                  # NSMT neocortex
        _, neo = m._tokens_mc(x)
        if j is not None:
            neo = neo.clone(); neo[j] = neo[j][perm]
        _, o = m.neo_forward(neo)
    functional.reset_net(m)
    return o.flatten(2)                                     # [N, B, K*D] (K = 1 except the stream model, MODI 2026-10-21)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True); ap.add_argument('--name', required=True); ap.add_argument('--out_dir', required=True)
    ap.add_argument('--n', type=int, default=512); ap.add_argument('--bs', type=int, default=64)
    a = ap.parse_args()
    cfg = torch.load(os.path.join(a.run_dir, 'model_state', 'config.pt'), map_location='cpu')
    args = argparse.Namespace(**cfg)
    m = LOAD_MODEL[args.model](args, True)
    m.load_state_dict(torch.load(os.path.join(a.run_dir, 'model_state', 'checkpoint.pth'), map_location='cpu'))
    m = m.cuda().eval(); m.train_mode = 'testing'
    if getattr(m, 'hippo_ssa', False) or getattr(m, 'mixer_kind', '') == 'none':
        print('no token-mixing network on the aligned axis; skipped'); return
    data = rd.build(args, int(args.seed))
    ds = data['test']; n = min(a.n, len(ds))
    xs = torch.stack([ds[i][0] for i in range(n)]).float()
    N = m.num_patches
    ch = np.zeros((N, N)); rate = []
    for b0 in range(0, n, a.bs):
        x = xs[b0:b0 + a.bs].cuda(); B = x.shape[0]
        perm = torch.roll(torch.arange(B), 1).cuda()
        o0 = net_out(m, x); rate.append(o0.float().mean().item())
        for j in range(N):
            o1 = net_out(m, x, j, perm)
            ch[j] += (o1 != o0).float().mean(dim=(1, 2)).cpu().numpy() * B
    ch /= n
    rows = []
    for d in range(-(N - 1), N):
        v = [ch[j, j + d] for j in range(N) if 0 <= j + d < N]
        rows.append(dict(lag=d, change=float(np.mean(v)), n_pairs=len(v)))
    out = os.path.join(a.out_dir, a.name); os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, 'reach_lag.csv'), index=False, float_format='%.6f')
    meta = dict(name=a.name, run_dir=a.run_dir, n_units=n, tokens=N, rate=float(np.mean(rate)),
                change_d0=float(ch.diagonal().mean()), change_backward_max=float(max((ch[j, t] for j in range(N) for t in range(j)), default=0.)))
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
