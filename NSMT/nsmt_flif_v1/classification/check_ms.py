"""MODI 2026-11-08: fail-closed checks of the membrane shortcut (--res_mode ms / --neo_res ms) on [query pop LIF NSMT T=4] and the
query pop f-LIF (docs/MODI_PROJECT_LOG.md 2026-11-08). c57a (default paths bit-identical) is fw_identity_ms.py.
  c57b no new parameters: for every variant the state_dict has the base's keys and shapes, and the same seed gives the same init
  c57c hippocampus wiring (hooks, fresh model in training mode and the base checkpoint loaded into the variant in eval mode):
       input current of the proj neuron == U0 + BN_proj output, of the fc2 neuron == that + BN_fc2 output, with U0 = the input current
       of the hippocampus embedding neuron; the blocks' output == the fc2 neuron's spikes; every Linear in the blocks reads 0/1
  c57d Neocortex wiring: input current of Neocortex layer i's neuron == U0 + sum_{j <= i} BN_j output, U0 = the input current of the
       Neocortex embedding neuron; spikes 0/1; the query Linear reads the last Neocortex layer's spikes (copied to every T_h step)
  c57e zero stream: forward_ms(x, 0) gives exactly the plain forward's spikes (SpikLinearLayer and NeoLayer)
  c57f gradients: the same parameters receive a gradient as in the base model, all finite; the variant's output differs from the base's
  any failed comparison or empty capture raises (exit code 1).
usage (classification folder): CUDA_VISIBLE_DEVICES=0 python check_ms.py
"""
import glob
import os
import sys
from types import SimpleNamespace as NS

import torch
import torch.nn as nn

sys.path.insert(0, os.getcwd())
from spikingjelly.clock_driven import functional                        # noqa: E402
from load_model import LOAD_MODEL                                        # noqa: E402
from forecast_wrap import Forecaster                                     # noqa: E402

VARIANTS = {'msh': dict(res_mode='ms'), 'msn': dict(neo_res='ms'), 'mshn': dict(res_mode='ms', neo_res='ms')}
TOL = 1e-6
FAIL = []


def ok(cond, what):
    print(('pass ' if cond else 'FAIL ') + what)
    if not cond:
        FAIL.append(what)


def binary(t):
    return t.numel() > 0 and bool(((t == 0) | (t == 1)).all())


def one(pattern):
    g = glob.glob(pattern)
    if len(g) != 1:
        raise RuntimeError(f'{pattern}: {g}')
    return g[0]


MB = one('log/flif-v1-mitbih-query/MITBIH/*/*+nsmt4q_pop_lif_lr0.01_v0+seed42+model+*/seed42_*')
EM = one('../forecasting/log/flif-v1-ett-query-m2/ETTm2/p96/*/*+nsmt4q_pop_lif+seed42+*/seed42_*')
EH = one('../forecasting/log/flif-v1-ett-query/ETTh2/p96/*/*+nsmt4q_pop_lif+seed42+*/seed42_*')


def build(run, task, ov, seed=0):
    cfg = torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')
    a = NS(**dict(cfg, device='cuda', **ov))
    torch.manual_seed(seed)
    m = LOAD_MODEL[a.model](a, True)
    if task == 'ett':
        m = Forecaster(m, a.embed_dim, a.pred_len, bias=a.bias)
    return a, m.cuda()


def inner(m):
    return m.inner if isinstance(m, Forecaster) else m


class Cap:
    def __init__(self):
        self.d, self.h = {}, []

    def on(self, mod, name, what='both'):
        def fn(_m, i, o):
            self.d[name] = (i[0].detach().clone(), (o[0] if isinstance(o, tuple) else o).detach().clone())
        self.h.append(mod.register_forward_hook(fn))

    def off(self):
        for h in self.h:
            h.remove()


def wiring(m, x, label, res_ms, neo_ms):
    """c57c / c57d on one forward of m (hooks); x = the model input."""
    im = inner(m); cap = Cap()
    cap.on(im.encoding.emb_lif, 'h_emb')
    blk = im.data_block[0]
    cap.on(blk.ssa.proj_linear.bn, 'h_pbn'); cap.on(blk.ssa.proj_linear.lif, 'h_plif')
    cap.on(blk.mlp.fc2.bn, 'h_fbn'); cap.on(blk.mlp.fc2.lif, 'h_flif'); cap.on(blk.ssa.q_linear, 'q_lin')
    lin = [(n, mm) for n, mm in im.data_block.named_modules() if isinstance(mm, nn.Linear)]
    for n, mm in lin:
        cap.on(mm, 'lin:' + n)
    cap.on(im.encoding_neo.emb_lif, 'n_emb')
    L = im.time_block.num_layers
    for i in range(L):
        cap.on(im.time_block.Mem[i].bn, f'n_bn{i}'); cap.on(im.time_block.Mem[i].lif, f'n_lif{i}')
    functional.reset_net(m)
    if isinstance(m, Forecaster):
        out = m.step_features(x)
    else:
        m(x)
        out = None
    cap.off()
    d = cap.d
    need = ['h_emb', 'h_pbn', 'h_plif', 'h_fbn', 'h_flif', 'q_lin', 'n_emb'] + [f'n_bn{i}' for i in range(L)] + [f'n_lif{i}' for i in range(L)]
    ok(all(k in d for k in need), f'{label}: every hook captured ({len(d)} tensors)')
    if not all(k in d for k in need):
        return
    T, B, N, D = d['h_plif'][0].shape
    U0 = d['h_emb'][0].transpose(-1, -2)                                   # [T B D N] -> [T B N D]
    za = d['h_pbn'][1].transpose(-1, -2).reshape(T, B, N, D)
    zm = d['h_fbn'][1].transpose(-1, -2).reshape(T, B, N, D)
    I1, S1 = d['h_plif']; I2, S2 = d['h_flif']
    if res_ms:
        e1 = float((I1 - (U0 + za)).abs().max()); e2 = float((I2 - (I1 + zm)).abs().max())
        ok(e1 <= TOL and e2 <= TOL, f'{label} c57c: proj neuron input == U0 + BN_proj ({e1:.2e}), fc2 neuron input == + BN_fc2 ({e2:.2e})')
    else:
        e1 = float((I1 - za).abs().max()); e2 = float((I2 - zm).abs().max())
        ok(e1 <= TOL and e2 <= TOL, f'{label} c57c (IAND hippocampus): proj / fc2 neuron inputs == their own BN outputs ({e1:.2e}, {e2:.2e})')
    if out is not None and res_ms:                                          # forecaster: the head reads step_features
        ok(torch.equal(out, S2), f'{label} c57c: step_features == fc2 spikes')
    ok(all(binary(d['lin:' + n][0]) for n, _ in lin), f'{label} c57c: all {len(lin)} hippocampus Linear inputs are 0/1')
    Nn = d['n_emb'][0].shape[0]
    u = d['n_emb'][0].transpose(-1, -2)                                    # [N B D 1] -> [N B 1 D]
    errs = []
    for i in range(L):
        Ii, Si = d[f'n_lif{i}']
        z = d[f'n_bn{i}'][1].reshape(Ii.shape)
        u = (u + z) if neo_ms else z
        errs.append(float((Ii - u).abs().max()))
        if neo_ms:
            u = Ii                                                         # continue from the captured stream
    ok(max(errs) <= TOL, f'{label} c57d: Neocortex layer neuron inputs == ' + ('U0 + sum BN_j' if neo_ms else 'own BN output') +
       f' ({", ".join(f"{e:.2e}" for e in errs)})')
    o = d[f'n_lif{L - 1}'][1]
    ok(binary(o) and binary(d['n_emb'][1]), f'{label} c57d: Neocortex spikes are 0/1')
    q_in = d['q_lin'][0]                                                   # [T_h*B, N, D]
    kv = o.transpose(0, 2).expand(T, -1, -1, -1).reshape(T * B, Nn, D)
    ok(torch.equal(q_in, kv), f'{label} c57d: the query Linear reads the last Neocortex spikes')


def main():
    # c57b
    for run, task, nm in ((MB, 'mb', 'MIT-BIH'), (EM, 'ett', 'ETTm2 H96'), (EH, 'ett', 'ETTh2 H96')):
        for neu, nov in (('pop LIF', {}), ('pop f-LIF', dict(neo_neuron='pop_flif', pop_soma_init=1.0))):
            _, b = build(run, task, nov)
            sb = b.state_dict()
            for v, ov in VARIANTS.items():
                _, m = build(run, task, dict(nov, **ov))
                s = m.state_dict()
                same = s.keys() == sb.keys() and all(s[k].shape == sb[k].shape and torch.equal(s[k], sb[k]) for k in sb)
                ok(same, f'{nm} {neu} {v} c57b: same {len(sb)} state_dict entries and init as the base')
    # c57c / c57d: fresh models in training mode, and the base checkpoint in eval mode
    xm = torch.randn(16, 1, 187, generator=torch.Generator().manual_seed(11)).cuda()
    ze = torch.randn(28, 1, 96, generator=torch.Generator().manual_seed(12)).cuda()
    for v, ov in dict(base={}, **VARIANTS).items():
        rm, nm_ = ov.get('res_mode') == 'ms', ov.get('neo_res') == 'ms'
        for run, task, nm, x, ck in ((MB, 'mb', 'MIT-BIH', xm, 'best+model.pt'), (EM, 'ett', 'ETTm2 H96', ze, 'checkpoint.pth'),
                                     (EH, 'ett', 'ETTh2 H96', ze, 'checkpoint.pth')):
            _, m = build(run, task, ov)
            m.train(); m.train_mode = 'training'
            with torch.no_grad():
                wiring(m, x, f'{nm} {v} train', rm, nm_)
            m.load_state_dict(torch.load(os.path.join(run, 'model_state', ck), map_location='cpu'))
            m.eval(); m.train_mode = 'testing'
            with torch.no_grad():
                wiring(m, x, f'{nm} {v} eval (base checkpoint)', rm, nm_)
    # c57e
    from layers import SpikLinearLayer
    torch.manual_seed(1)
    sl = SpikLinearLayer(64, 64).cuda().eval()
    xs = (torch.rand(4, 3, 24, 64, generator=torch.Generator().manual_seed(2)) < 0.2).float().cuda()
    with torch.no_grad():
        functional.reset_net(sl); a = sl(xs)
        functional.reset_net(sl); u, b2 = sl.forward_ms(xs, torch.zeros(4, 3, 24, 64, device='cuda'))
    ok(torch.equal(a, b2), 'c57e: SpikLinearLayer.forward_ms(x, 0) spikes == forward(x)')
    _, m = build(EM, 'ett', dict(neo_res='ms'))
    m.eval()
    nl = m.inner.time_block.Mem[1]
    xn = (torch.rand(24, 5, 1, 64, generator=torch.Generator().manual_seed(3)) < 0.1).float().cuda()
    with torch.no_grad():
        functional.reset_net(nl); a = nl(xn)
        functional.reset_net(nl); u, b2 = nl.forward_ms(xn, torch.zeros_like(xn))
    ok(torch.equal(a, b2), 'c57e: NeoLayer.forward_ms(x, 0) spikes == forward(x) (pop LIF)')
    # c57f
    for run, task, nm, x in ((MB, 'mb', 'MIT-BIH', xm), (EM, 'ett', 'ETTm2 H96', torch.randn(4, 96, 7, generator=torch.Generator().manual_seed(6)).cuda())):
        outs, grads = {}, {}
        for v, ov in dict(base={}, **VARIANTS).items():
            a_, m = build(run, task, ov)
            m.train(); m.train_mode = 'training'; functional.reset_net(m)
            if task == 'mb':
                o = m(x); o = o[0] if isinstance(o, tuple) else o
                loss = torch.nn.functional.cross_entropy(o, torch.arange(16, device='cuda') % 5)
            else:
                o = m(x); loss = torch.nn.functional.l1_loss(o, torch.zeros_like(o))
            loss.backward()
            outs[v] = o.detach()
            grads[v] = {n: p.grad for n, p in m.named_parameters() if p.grad is not None}
        for v in VARIANTS:
            ok(grads[v].keys() == grads['base'].keys() and all(bool(torch.isfinite(g).all()) for g in grads[v].values()),
               f'{nm} {v} c57f: gradient on the same {len(grads["base"])} parameters as the base, all finite')
    # c57f (option active): a fresh Neocortex emits no spikes (first run of this check failed closed on msn for that reason, as c55c on
    # 11-04), so the outputs are compared with the trained base weights loaded into every variant, in eval mode
    for run, task, nm, x, ck in ((MB, 'mb', 'MIT-BIH', xm, 'best+model.pt'), (EM, 'ett', 'ETTm2 H96', ze, 'checkpoint.pth'),
                                 (EH, 'ett', 'ETTh2 H96', ze, 'checkpoint.pth')):
        outs, neo = {}, {}
        for v, ov in dict(base={}, **VARIANTS).items():
            _, m = build(run, task, ov)
            m.load_state_dict(torch.load(os.path.join(run, 'model_state', ck), map_location='cpu'))
            m.eval(); m.train_mode = 'testing'
            im = inner(m); cap = Cap(); cap.on(im.time_block.Mem[im.time_block.num_layers - 1].lif, 'neo')
            with torch.no_grad():
                functional.reset_net(m)
                if task == 'mb':
                    o = m(x); o = o[0] if isinstance(o, tuple) else o
                else:
                    o = m.step_features(x)
            cap.off()
            outs[v] = o.detach(); neo[v] = cap.d['neo'][1]
        for v in VARIANTS:
            ok(not torch.equal(outs[v], outs['base']),
               f'{nm} {v} c57f: output differs from the base with the same trained weights (last Neocortex layer spikes: base '
               f'{int(neo["base"].sum())}, {v} {int(neo[v].sum())}; Neocortex spikes differ: {not torch.equal(neo[v], neo["base"])})')
    print('\nALL PASS' if not FAIL else f'\n{len(FAIL)} FAILED: {FAIL}')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
