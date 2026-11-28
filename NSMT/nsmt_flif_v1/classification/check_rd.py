"""MODI 2026-11-27 (approved plan 1-snuggly-galaxy.md, step 2.3): checks for --neo_readout memd (the Neocortex membrane, BN only, used
directly as the hippocampus queries; no q_linear, no query LIF) and for the accumulation aux on ETT (Forecaster.aux_past_loss,
train_forecast --past_aux_w) and MIT-BIH (train.py takes outputs 4 / 5). The past_pair refactor is covered by fw_identity_rd.py
(the mem / acc models of MIT-BIH and Sleep reproduce bit for bit). Fail closed.
usage (classification folder): python check_rd.py
"""
import os
import sys
from types import SimpleNamespace as NS

import torch
from spikingjelly.clock_driven import functional

from config import Config, parse_arguments
from forecast_wrap import Forecaster
from load_model import LOAD_MODEL
import train_forecast as TF

HERE = os.path.dirname(os.path.abspath(__file__))
FAILS = []
SLEEP = ("--data SLEEPEDF --seed 42 --log_dir /tmp/claude-1001/check_rd_cfg --gating attn -s -nd 0 -emb 384 -nh 8 --mlp_ratios 1 --layers 1 "
         "--time_layers 2 --keep_ratio 0.15 --alpha 0 --tau 2.0 --no-bias --patch_size 1 --token_step 1 --head_pool last --ssa_pe 0 --fold 0 "
         "--ctx_epochs 16 --model myModel --neo_where both --hippo_T 4 --hippo_xattn --neo_grad 1.0 --neo_res ms --neuron_all lif")
NRN = {'lif': '--neo_neuron lif', 'flif': '--neo_neuron flif --neo_alpha 0.5', 'pop_lif': '--neo_neuron pop_lif --pop_soma_init 0.5'}


def ok(name, cond, info=''):
    print(f'{"PASS" if cond else "FAIL"} {name} {info}', flush=True)
    if not cond:
        FAILS.append(name)


def sleep_model(extra):
    sys.argv = ['x'] + (SLEEP + ' ' + extra).split()
    a = Config(); a.set_args(parse_arguments()); a.num_classes, a.num_channels, a.seq_len = 5, 60, 16
    torch.manual_seed(0)
    return LOAD_MODEL['myModel'](a, True).cuda()


def r1_cfg(path, cond, **kw):
    for ln in open(path):
        if ln.startswith('RESULT') and f' cond={cond} ' in ln and ' seed=42 ' in ln and all(f' {k}={v} ' in ln for k, v in kw.items()):
            d = dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t)['dir']
            return torch.load(os.path.join(d, 'model_state', 'config.pt'), map_location='cpu')
    raise RuntimeError((path, cond))


def ssas(m):
    return [blk.ssa for blk in m.data_block]


def main():
    torch.manual_seed(1)
    xs = torch.randn(8, 60, 16).cuda(); ys = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2]).cuda()
    # a. memd structure (Sleep size, all three default neurons)
    for n, flag in NRN.items():
        mm, md = sleep_model(f'{flag} --hippo_xattn_mode query --neo_readout mem'), sleep_model(f'{flag} --hippo_xattn_mode query --neo_readout memd')
        ok(f'a1 {n}: memd removes q_linear and the query LIF, q_direct on', all(s.q_linear is None and s.q_lif is None and s.q_direct for s in ssas(md))
           and all(s.q_linear is not None for s in ssas(mm)))
        sm, sd = mm.state_dict(), md.state_dict()
        gone = sorted(set(sm) - set(sd))
        ok(f'a2 {n}: only q_linear / q_lif entries are gone, every other initial value equals the mem model',
           set(sd) <= set(sm) and bool(gone) and all('.ssa.q_linear' in k or '.ssa.q_lif' in k for k in gone)
           and all(torch.equal(sm[k], sd[k]) for k in sd), f'{len(gone)} entries gone')
        cap = {}
        hk = ssas(md)[0].q_bn.register_forward_hook(lambda mod, i, o: cap.update(i=i[0].detach(), o=o.detach()))
        md.train(); md.train_mode = 'training'; functional.reset_net(md)
        with torch.no_grad():
            _, neo_in = md._tokens_mc(xs)
            _, xn = md.neo_forward(neo_in)
            functional.reset_net(md)
        cap.clear()
        o = md(xs); hk.remove()
        mem = xn[:, :, 0].permute(1, 2, 0)                                   # [B, D, N]
        ok(f'a3 {n}: the q_bn input is the Neocortex membrane readout (real-valued)', 'i' in cap and cap['i'].shape == mem.shape
           and torch.allclose(cap['i'], mem, atol=1e-6) and ((mem != 0) & (mem != 1)).float().mean().item() > 0.5)
        loss = torch.nn.functional.cross_entropy(o[0], ys); loss.backward(); functional.reset_net(md)
        g_last = md.time_block.Mem[-1].w_in.weight.grad
        g_bn = ssas(md)[0].q_bn.weight.grad
        ok(f'a4 {n}: training step finite; gradient reaches q_bn and the last Neocortex layer', bool(torch.isfinite(loss))
           and g_bn is not None and g_bn.abs().sum().item() > 0 and g_last is not None and g_last.abs().sum().item() > 0
           and all(bool(torch.isfinite(p.grad).all()) for p in md.parameters() if p.grad is not None))
        md.eval(); md.train_mode = 'testing'; functional.reset_net(md)
        with torch.no_grad():
            oe = md(xs)
        oe = oe[0] if isinstance(oe, tuple) else oe
        ok(f'a5 {n}: eval output finite', bool(torch.isfinite(oe).all()))
    refused = 0
    for extra in ('--neo_neuron lif --neo_readout memd', '--neo_neuron lif --hippo_xattn_mode query --neo_readout memd --attn_score xnor_log'):
        try:
            sleep_model(extra)
        except (NotImplementedError, ValueError):
            refused += 1
    ok('a6 memd refused outside the query form with the dot score', refused == 2)
    # b. ETT: Forecaster aux
    ETT = os.path.join(HERE, '..', 'forecasting', 'results', 'flif-v1-ett-r1', 'raw.txt')
    cfg = r1_cfg(ETT, 'nsmt4q_pop_lif_msn_sde', ds='ETTh2', pl='96')
    xe = torch.randn(4, 96, 7, generator=torch.Generator().manual_seed(5)).cuda()
    ye = torch.randn(4, 96, 7, generator=torch.Generator().manual_seed(6)).cuda()
    for n, ov in (('lif', dict(neo_neuron='lif', pop_soma_init=None)), ('pop_lif', {})):
        for ro in ('spike', 'memd'):
            a = NS(**dict(cfg, device='cuda', aux_past='rand', neo_readout=ro, **ov))
            torch.manual_seed(0); f = Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, a.pred_len, bias=a.bias).cuda()
            f.aux_past = True
            m = f.inner
            f.train(); f.train_mode = 'training'; functional.reset_net(f)
            out = f(xe)
            x_neo, neo_in = f._past
            loss = f.aux_past_loss()
            t, d, src = m._aux_past_last
            N, Bp = neo_in.shape[0], neo_in.shape[1]
            x = xe - xe.mean(1, keepdim=True); x = x / torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            z = x.permute(0, 2, 1).reshape(Bp, 96)
            P, S = m.patch_size, m.stride
            tok_ok = all(torch.allclose(neo_in[j, :, 0], z[:, j * S:j * S + P], atol=1e-6) for j in range(N) if j * S + P <= 96)
            torch.manual_seed(3); tgt, rec = m.past_pair(x_neo, neo_in); t2, d2, src2 = m._aux_past_last
            tg_ok = all(torch.equal(tgt[i, b], neo_in[src2[i, b], b, 0]) for i in range(N - 1) for b in range(0, Bp, 5))
            ok(f'b1 {n} {ro}: tokens = normalised input patches, target = token t - d, d in 1..t, finite loss',
               tok_ok and tg_ok and bool(((d >= 1) & (d <= t)).all()) and bool(torch.isfinite(loss)), f'N {N}, patch {P}, step {S}')
            if ro == 'memd':
                ok(f'b2 {n} memd: the aux reads the membrane', ((x_neo != 0) & (x_neo != 1)).float().mean().item() > 0.5)
            calls = []
            h = m.aux_past_dec.register_forward_hook(lambda *_: calls.append(1))
            f.eval(); f.train_mode = 'testing'; functional.reset_net(f)
            with torch.no_grad():
                f(xe)
            h.remove()
            ok(f'b3 {n} {ro}: no decoder call and nothing kept at test time', not calls and f._past is None)
            try:
                f.aux_past_loss(); raised = False
            except RuntimeError:
                raised = True
            ok(f'b4 {n} {ro}: aux_past_loss without a training forward stops', raised)
    # b5. loss mixing of train_forecast.train_epoch: (1 - w) L1 + w aux, the aux from aux_fn
    w, c = 0.1, torch.tensor(2.5, device='cuda', requires_grad=True)
    f.train()
    dl = [(xe, torch.cat([torch.zeros(4, 48, 7).cuda(), ye], 1), None, None)]
    opt = torch.optim.SGD(f.parameters(), lr=0.)
    with torch.no_grad():
        f.eval(); f.train_mode = 'testing'; functional.reset_net(f)
    f.train()
    res = TF.train_epoch(f, dl, opt, 96, 'cuda', r=w, aux_fn=lambda: c * 1.)
    ok('b5 train_epoch: loss = (1 - w) task + w aux (aux from aux_fn)', abs(res['loss'] - ((1 - w) * res['task'] + w * 2.5)) < 1e-5 and res['rec'] == 2.5,
       f"loss {res['loss']:.6f} task {res['task']:.6f}")
    # c. MIT-BIH: the model output carries (target, reconstruction) at positions 4 / 5 (train.py: mse of those, weight --alpha)
    MB = os.path.join(HERE, 'results', 'flif-v1-mitbih-r1', 'raw.txt')
    cfg = r1_cfg(MB, 'nsmt4q_pop_lif_msn_sde')
    xm = torch.randn(8, 1, 187, generator=torch.Generator().manual_seed(5)).cuda()
    for n, ov in (('lif', dict(neo_neuron='lif', pop_soma_init=None)), ('flif', dict(neo_neuron='flif', neo_alpha=0.5, pop_soma_init=None))):
        for ro in ('spike', 'memd'):
            a = NS(**dict(cfg, device='cuda', aux_past='rand', alpha=0.1, neo_readout=ro, **ov))
            torch.manual_seed(0); m = LOAD_MODEL[a.model](a, True).cuda()
            m.train(); m.train_mode = 'training'; functional.reset_net(m)
            o = m(xm)
            t, d, src = m._aux_past_last
            raw = m._neo_in[:, :, 0]
            okt = all(torch.equal(o[3][i, b], raw[src[i, b], b]) for i in range(o[3].shape[0]) for b in range(8))
            ok(f'c1 MIT-BIH {n} {ro}: outputs 4 / 5 = (raw token t - d, reconstruction), finite', okt and bool(torch.isfinite(((o[3] - o[4]) ** 2).mean())),
               f'tokens {raw.shape[0]}')
            functional.reset_net(m)
    print('ALL PASS' if not FAILS else f'FAILED: {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
