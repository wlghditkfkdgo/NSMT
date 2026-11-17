"""No-training probe (2026-11-17, user question 1): in trained LIF rep SpkMixer checkpoints (tuned, IAND), how different are the 4 repeat
sequences that the mixer receives, and what happens to the test MSE if the trained mixer + fusion is run ONCE on the repeat-pooled input
(mean or max over the 4 repeats) and its output is broadcast back to the 4 repeats for the residual and the MLP (no retraining, eval BN).
usage: python pool_probe.py <run dir> [<run dir> ...]   (run dir = the seed folder with model_state/)"""
import argparse, os, sys, warnings
warnings.filterwarnings('ignore')
C = '/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1'
sys.path.insert(0, C); os.chdir(C)
import torch
from spikingjelly.clock_driven import functional
from forecast_wrap import Forecaster
from load_model import LOAD_MODEL
from data_provider.data_factory import data_provider
from train_forecast import data_args


@torch.no_grad()
def probe(run, dev='cuda'):
    cfg = torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')
    args = argparse.Namespace(**cfg); args.device = dev
    model = Forecaster(LOAD_MODEL[args.model](args, True), args.embed_dim, args.pred_len, bias=args.bias)
    model.load_state_dict(torch.load(os.path.join(run, 'model_state', 'checkpoint.pth'), map_location='cpu'))
    model = model.to(dev).eval(); model.train_mode = 'testing'
    m = model.inner; blk = m.blocks[0]
    assert len(m.blocks) == 1 and m.T_h == 4 and m.connect_f == 'IAND'
    _, dl = data_provider(data_args(args, args), 'test')
    se = {k: 0. for k in ('rep', 'mean', 'max')}; cnt = 0
    st = dict(e_rate=0., e_or=0., e_diff=0., m_rate=0., m_diff=0., n=0)
    functional.reset_net(model)
    for x, y, _, _ in dl:
        x = x.float().to(dev); y = y.float().to(dev)[:, -args.pred_len:, :]
        B, L, Cc = x.shape
        mu = x.mean(1, keepdim=True); xc = x - mu
        sd = torch.sqrt(torch.var(xc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        z = (xc / sd).permute(0, 2, 1).reshape(B * Cc, 1, L)
        e = m.encoding(m.tokens(z))                                   # [4, B', N, D] 0/1
        outs = {}
        mr = blk.mix(e)                                               # per repeat (the trained model)
        outs['rep'] = mr
        outs['mean'] = blk.mix(e.mean(0, keepdim=True)).expand_as(e)  # once on the pooled input, broadcast to the 4 repeats
        outs['max'] = blk.mix(e.amax(0, keepdim=True)).expand_as(e)
        for k, mm in outs.items():
            h = blk.connect(e, mm); h = blk.connect(h, blk.fc2(blk.fc1(h)))
            yy = model.head(h.mean(0).reshape(B * Cc, -1)).reshape(B, Cc, -1).permute(0, 2, 1) * sd + mu
            se[k] += float(((yy - y) ** 2).sum())
        cnt += y.numel()
        n = e[0].numel()
        st['e_rate'] += float(e.mean()) * n; st['e_or'] += float(e.amax(0).mean()) * n
        st['e_diff'] += float((e.amax(0) != e.amin(0)).float().mean()) * n
        st['m_rate'] += float(mr.mean()) * n; st['m_diff'] += float((mr.amax(0) != mr.amin(0)).float().mean()) * n
        st['n'] += n
    n = st.pop('n')
    return {k: v / cnt for k, v in se.items()}, {k: v / n for k, v in st.items()}, args


for run in sys.argv[1:]:
    mse, st, a = probe(run)
    print(f"{a.data} H{a.pred_len} seed {a.seed}: test MSE rep {mse['rep']:.6f} | mixer once on mean-pooled {mse['mean']:.6f} ({100 * (mse['mean'] / mse['rep'] - 1):+.2f} %) "
          f"| on max-pooled {mse['max']:.6f} ({100 * (mse['max'] / mse['rep'] - 1):+.2f} %)")
    print(f"   embedding rate {st['e_rate']:.4f}, OR over repeats {st['e_or']:.4f}, positions where the 4 repeats differ {st['e_diff']:.4f}; "
          f"mixer output rate {st['m_rate']:.4f}, positions where the 4 repeats differ {st['m_diff']:.4f}", flush=True)
