# usage (classification folder): python check_head_h2h6.py
# MODI 2026-11-04: checks of the H2 (--head_pool attn, MIT-BIH) and H6 (--fc_head h6, ETT) heads on fresh query models of the real
# run configurations (config.pt of the MIT-BIH / ETTh2 query runs). Fail-closed: any failed check raises.
#   c55b  H2: q = 0 at init, the rest of the model initialised exactly as the mean head (same state_dict apart from head_score);
#         train-mode logits = mean head (|diff| <= 1e-6, softmax 1/N vs mean); q gets a non-zero gradient; with a random q the
#         pooled vector equals an explicit softmax-weighted sum and the weights are non-uniform and sum to 1
#   c55c  H6 (body = the trained ETTh2 seed-42 checkpoint, so the hippocampus fires; a fresh body emits no spikes and would make
#         the comparisons empty): per-step proj then T_h mean = proj of the T_h mean (<= 1e-6); body_off output = shortcut alone (exact); full output =
#         shortcut + head(proj(...)) recomputed by hand (<= 1e-5); gradients reach proj, head, shortcut, Neocortex and hippocampus;
#         parameter shapes D x d, N*d x H, L x H and no N*D x H layer
import glob, os, sys
from types import SimpleNamespace as NS
import torch
sys.path.insert(0, os.getcwd())
from spikingjelly.clock_driven import functional
from load_model import LOAD_MODEL
from forecast_wrap import Forecaster


def need(ok, msg):
    if not ok:
        raise RuntimeError('FAIL ' + msg)
    print('ok', msg)


dev = 'cuda'
mb = glob.glob('log/flif-v1-mitbih-query/MITBIH/*/*+nsmt4q_pop_lif_lr0.01_v0+seed42+model+*/seed42_*')
assert len(mb) == 1
cfg = torch.load(os.path.join(mb[0], 'model_state', 'config.pt'), map_location='cpu')
x = torch.randn(8, 1, 187, generator=torch.Generator().manual_seed(5)).to(dev)
y = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2]).to(dev)
for nn_ in ('pop_lif', 'pop_flif'):
    ov = {} if nn_ == 'pop_lif' else dict(neo_neuron='pop_flif', pop_soma_init=1.0)
    torch.manual_seed(0); m0 = LOAD_MODEL['myModel'](NS(**dict(cfg, device=dev, **ov)), True).to(dev)
    torch.manual_seed(0); m1 = LOAD_MODEL['myModel'](NS(**dict(cfg, device=dev, head_pool='attn', **ov)), True).to(dev)
    s0, s1 = m0.state_dict(), m1.state_dict()
    need(set(s1) - set(s0) == {'head_score.weight'} and all(torch.equal(s0[k], s1[k]) for k in s0),
         f'c55b {nn_}: body initialised exactly as the mean head, only head_score.weight added')
    need(float(m1.head_score.weight.abs().max()) == 0., f'c55b {nn_}: q = 0 at init')
    outs = []
    for m in (m0, m1):
        m.train(); m.train_mode = 'training'; functional.reset_net(m)
        o = m(x); lg = o[0] if isinstance(o, tuple) else o
        torch.nn.functional.cross_entropy(lg, y).backward(); outs.append(lg.detach()); functional.reset_net(m)
    d = float((outs[0] - outs[1]).abs().max())
    need(d <= 1e-6, f'c55b {nn_}: train logits at init = mean head (max |diff| {d:.2e})')
    g = float(m1.head_score.weight.grad.abs().max())
    need(g > 0, f'c55b {nn_}: q gets a gradient (max |grad| {g:.2e})')
    with torch.no_grad():
        m1.head_score.weight.normal_(0, 1., generator=torch.Generator(device=dev).manual_seed(3))
        xs = (torch.rand(4, 3, m1.num_patches, cfg['embed_dim'], device=dev, generator=torch.Generator(device=dev).manual_seed(4)) < 0.2).float()
        q = m1.head_score.weight[0]
        s = torch.einsum('tbnd,d->tbn', xs, q) / cfg['embed_dim'] ** 0.5
        a = torch.exp(s - s.max(2, keepdim=True).values); a = a / a.sum(2, keepdim=True)
        ref = torch.einsum('tbn,tbnd->tbd', a, xs)
        got = m1._pool(xs)
        dd = float((ref - got).abs().max())
        need(dd <= 1e-5 and float((a.sum(2) - 1).abs().max()) <= 1e-6 and float(a.std(2).min()) > 0,
             f'c55b {nn_}: random q pooling = explicit softmax-weighted sum (max |diff| {dd:.2e}), weights sum to 1, non-uniform')

for pl in (96, 720):
    run = glob.glob(f'../forecasting/log/flif-v1-ett-query/ETTh2/p{pl}/*/*+nsmt4q_pop_lif+seed42+*/seed42_*')
    assert len(run) == 1
    cfg = torch.load(os.path.join(run[0], 'model_state', 'config.pt'), map_location='cpu')
    a = NS(**dict(cfg, device=dev))
    torch.manual_seed(0)
    f = Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, pl, bias=a.bias, head_kind='h6', head_dim=8, seq_len=96).to(dev)
    sd0 = torch.load(os.path.join(run[0], 'model_state', 'checkpoint.pth'), map_location='cpu')
    f.inner.load_state_dict({k[6:]: v for k, v in sd0.items() if k.startswith('inner.')})
    N, D = f.num_patches, a.embed_dim
    shapes = {n: tuple(p.shape) for n, p in f.named_parameters() if not n.startswith('inner.')}
    need(shapes == {'proj.weight': (8, D), 'head.weight': (pl, 8 * N), 'shortcut.weight': (pl, 96)},
         f'c55c H{pl}: head parameters {shapes} (no N*D x H layer)')
    xe = torch.randn(4, 96, 7, generator=torch.Generator().manual_seed(6)).to(dev)
    ye = torch.randn(4, pl, 7, generator=torch.Generator().manual_seed(7)).to(dev)
    f.eval(); f.train_mode = 'testing'
    with torch.no_grad():
        means = xe.mean(1, keepdim=True); xc = xe - means
        sd = torch.sqrt(torch.var(xc, dim=1, keepdim=True, unbiased=False) + 1e-5); z = (xc / sd).permute(0, 2, 1).reshape(28, 1, 96)
        functional.reset_net(f); st = f.step_features(z); functional.reset_net(f)
        need(float(st.mean()) > 0, f'c55c H{pl}: hippocampus spikes present (rate {float(st.mean()):.4f})')
        p_step = f.proj(st).mean(0); p_mean = f.proj(st.mean(0))
        dp = float((p_step - p_mean).abs().max() / p_mean.abs().max())
        need(dp <= 1e-6, f'c55c H{pl}: per-step proj then mean = proj of the mean (relative {dp:.2e})')
        lin = f.shortcut(z[:, 0]).reshape(4, 7, pl).permute(0, 2, 1) * sd + means
        full = (f.shortcut(z[:, 0]) + f.head(p_mean.reshape(28, -1))).reshape(4, 7, pl).permute(0, 2, 1) * sd + means
        f.body_off = True; functional.reset_net(f); o_off = f(xe); f.body_off = False
        functional.reset_net(f); o_on = f(xe); functional.reset_net(f)
        need(torch.equal(o_off, lin), f'c55c H{pl}: body_off output = shortcut alone (exact)')
        dd = float((o_on - full).abs().max() / full.abs().max())
        need(dd <= 1e-5, f'c55c H{pl}: full output = shortcut + head(proj) by hand (relative {dd:.2e})')
    f.train(); f.train_mode = 'training'; functional.reset_net(f)
    torch.nn.functional.l1_loss(f(xe), ye).backward(); functional.reset_net(f)
    gn = {k: sum(float(p.grad.abs().sum()) for n, p in f.named_parameters() if p.grad is not None and n.startswith(k))
          for k in ('proj.', 'head.', 'shortcut.', 'inner.time_block', 'inner.encoding_neo', 'inner.data_block', 'inner.encoding.')}
    need(all(v > 0 for v in gn.values()), f'c55c H{pl}: gradients reach ' + ', '.join(f'{k} {v:.2e}' for k, v in gn.items()))
print('all checks passed')
