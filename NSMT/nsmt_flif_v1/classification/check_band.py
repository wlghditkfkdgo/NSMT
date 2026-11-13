"""MODI 2026-11-13: fail-closed checks of the frequency-band input of the population Neocortex (--neo_band freq / same / rev on the
Neocortex-MS query population, docs/MODI_PROJECT_LOG.md 2026-11-13). c58a (default paths and the msn baseline path bit-identical) is
fw_identity_ms.py + fw_identity_band.py.
  c58b band_split: the 4 bands add up to the signal, each band's DCT coefficients outside its range are 0, the mean is in band 0,
       edges [0, n/8, n/4, n/2, n] (ceil)
  c58c neuron: per-member currents with 4 equal slots give exactly the spikes of the shared current (pop LIF and pop f-LIF); with 4
       different currents the pop LIF spikes equal the written equations v_k <- v_k + (I_k - v_k) / tau_k, a = sum_k w_k v_k,
       V <- V + (a - V) / tau_s, s = [V >= theta], V <- V - theta s (member k really integrates I_k with tau_k)
  c58d wiring (hooks; trained msn weights transplanted, eval mode, and fresh model, training mode): member time constants are
       (4, 8, 16, 32); the input of embedding Linear k is the expected signal (freq: band 3 - k, i.e. lowest band -> tau 32;
       rev: band k; same: the unsplit tokens); the embedding neuron's member currents == BN(Linear_k(.)) (eval: running statistics);
       Neocortex layer i's member currents == embedding member currents + sum_{j <= i} BN_j output (the same for every member);
       spikes 0/1; the query Linear reads the last Neocortex layer's spikes
  c58e 'same' with the msn checkpoint transplanted (the 4 Linears = the msn embedding Linear, BN copied per member) gives the msn
       checkpoint's output
  c58f the three variants have the same state_dict keys, shapes and (same seed) the same init; against msn only the Neocortex embedding
       entries differ; parameter counts
  c58g gradients (transplanted weights, training mode): finite everywhere, non-zero on all 4 embedding Linears
  c58h the routing has an effect with the same weights: the last Neocortex layer's spikes and the query neuron's spikes differ between
       every pair of variants, and the outputs of freq and rev differ from same's. Compared with the embedding BN on batch statistics.
       History: run 1 compared the outputs with the msn running statistics (fitted to the unsplit signal); the band currents are then
       too small, the ETTh2 Neocortex emits no spikes and freq == rev (failed closed). Run 2 (batch statistics) still found the ETTh2
       pop LIF outputs of freq and rev equal: their Neocortex (23 positions) and query spikes (374) differ, but with these transplanted
       weights the attention neuron emits no spike in either, so nothing reaches the output. The pairwise criterion was therefore moved
       to the Neocortex and query spikes; the freq-vs-rev output difference is printed, not required.
  any failed comparison or empty capture raises (exit code 1).
usage (classification folder): CUDA_VISIBLE_DEVICES=0 python check_band.py
"""
import glob
import math
import os
import sys
from types import SimpleNamespace as NS

import torch

sys.path.insert(0, os.getcwd())
from spikingjelly.clock_driven import functional                        # noqa: E402
from load_model import LOAD_MODEL                                        # noqa: E402
from forecast_wrap import Forecaster                                     # noqa: E402
from model import band_split, dct_matrix, dct_time                       # noqa: E402
from layers import build_neo_neuron                                      # noqa: E402

BANDS = ('freq', 'same', 'rev')
TAUS = (4., 8., 16., 32.)
TOL = 1e-6
FAIL = []


def ok(cond, what):
    print(('pass ' if cond else 'FAIL ') + what, flush=True)
    if not cond:
        FAIL.append(what)


def binary(t):
    return t.numel() > 0 and bool(((t == 0) | (t == 1)).all())


def one(pattern):
    g = glob.glob(pattern)
    if len(g) != 1:
        raise RuntimeError(f'{pattern}: {g}')
    return g[0]


def runs(n):
    return (('MIT-BIH', 'mb', one(f'log/flif-v1-mitbih-ms/MITBIH/*/*+nsmt4q_{n}_msn_lr0.01_v0+seed42+model+*/seed42_*'), 'best+model.pt'),
            ('ETTm2 H96', 'ett', one(f'../forecasting/log/flif-v1-ett-ms/ETTm2/p96/*/*+nsmt4q_{n}_msn+seed42+*/seed42_*'), 'checkpoint.pth'),
            ('ETTh2 H96', 'ett', one(f'../forecasting/log/flif-v1-ett-ms/ETTh2/p96/*/*+nsmt4q_{n}_msn+seed42+*/seed42_*'), 'checkpoint.pth'))


def build(run, task, ov, seed=0):
    cfg = torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')
    a = NS(**dict(cfg, device='cuda', **ov))
    assert a.neo_res == 'ms' and a.res_mode == 'iand'
    torch.manual_seed(seed)
    m = LOAD_MODEL[a.model](a, True)
    if task == 'ett':
        m = Forecaster(m, a.embed_dim, a.pred_len, bias=a.bias)
    return a, m.cuda()


def inner(m):
    return m.inner if isinstance(m, Forecaster) else m


def transplant(m, sd, K=4):
    """msn state_dict -> band model: the msn embedding Linear into all K Linears, the msn embedding BN per (unit, member) channel
    (channel d * K + k <- d); everything else by name. Raises on any key left over."""
    pre = ('inner.' if isinstance(m, Forecaster) else '') + 'encoding_neo.'
    new, used = {}, set()
    for k, v in m.state_dict().items():
        if k.startswith(pre + 'emb_linear.'):
            src = pre + 'emb_linear.weight'
            new[k] = sd[src].clone(); used.add(src)
        elif k.startswith(pre + 'emb_bn.'):
            new[k] = sd[k].clone() if v.ndim == 0 else sd[k].repeat_interleave(K); used.add(k)
        else:
            new[k] = sd[k].clone(); used.add(k)
    left = set(sd) - used
    if left:
        raise RuntimeError(f'transplant: unused checkpoint keys {sorted(left)}')
    m.load_state_dict(new)


class Cap:
    def __init__(self):
        self.d, self.h = {}, []

    def on(self, mod, name):
        def fn(_m, i, o):
            self.d[name] = (i[0].detach().clone(), (o[0] if isinstance(o, tuple) else o).detach().clone())
        self.h.append(mod.register_forward_hook(fn))

    def off(self):
        for h in self.h:
            h.remove()


def run_model(m, x):
    functional.reset_net(m)
    if isinstance(m, Forecaster):
        return m.step_features(x)
    o = m(x)
    return o[0] if isinstance(o, tuple) else o


def expected_slots(im, x, band):
    """Independent recomputation of the K signals the embedding Linears must read: [N, B, 1, P] per member k."""
    xp = im.padding_patch_layer(x)
    B, C, _ = xp.shape

    def patch(z):
        p = z.unfold(dimension=-1, size=im.patch_size, step=im.stride)
        return p.permute(0, 2, 1, 3).reshape(B, p.shape[2], C * im.patch_size).transpose(0, 1).unsqueeze(2)
    if band == 'same':
        return [patch(xp)] * 4
    b = band_split(xp, 4, time_dim=2)
    return [patch(b[3 - k]) for k in range(4)] if band == 'freq' else [patch(b[k]) for k in range(4)]


def wiring(m, x, label, band, eval_bn):
    im = inner(m); cap = Cap(); enc = im.encoding_neo
    for k in range(4):
        cap.on(enc.emb_linear[k], f'lin{k}')
    cap.on(enc.emb_bn, 'ebn'); cap.on(enc.emb_lif.neuron, 'eneu'); cap.on(enc.emb_lif, 'eneo')
    L = im.time_block.num_layers
    for i in range(L):
        cap.on(im.time_block.Mem[i].bn, f'bn{i}'); cap.on(im.time_block.Mem[i].lif.neuron, f'neu{i}'); cap.on(im.time_block.Mem[i].lif, f'neo{i}')
    cap.on(im.data_block[0].ssa.q_linear, 'q')
    with torch.no_grad():
        run_model(m, x)
    cap.off(); d = cap.d
    need = [f'lin{k}' for k in range(4)] + ['ebn', 'eneu', 'eneo', 'q'] + [f'{p}{i}' for i in range(L) for p in ('bn', 'neu', 'neo')]
    ok(all(k in d for k in need), f'{label}: every hook captured ({len(d)} tensors)')
    if not all(k in d for k in need):
        return
    taus = enc.emb_lif.neuron.tau.detach().cpu().tolist()
    ok(taus == list(TAUS) and all(im.time_block.Mem[i].lif.neuron.tau.detach().cpu().tolist() == list(TAUS) for i in range(L)),
       f'{label} c58d: member time constants {taus}')
    exp = expected_slots(im, x, band)
    e = max(float((d[f'lin{k}'][0] - exp[k]).abs().max()) for k in range(4))
    names = {'freq': 'band 3 - k (lowest band -> tau 32)', 'rev': 'band k (lowest band -> tau 4)', 'same': 'the unsplit tokens'}[band]
    ok(e == 0., f'{label} c58d: embedding Linear k reads {names} (max diff {e:.1e})')
    if band != 'same':
        e2 = float((sum(exp) - expected_slots(im, x, 'same')[0]).abs().max())
        ok(e2 <= 1e-5, f'{label} c58b: the 4 member inputs add up to the unsplit tokens ({e2:.1e})')
    N, B, N1, D = d['lin0'][1].shape
    cur = d['eneu'][0].reshape(N, B, N1, D, 4)                              # member currents the population neuron received
    pre = torch.stack([d[f'lin{k}'][1] for k in range(4)], dim=-1)         # [N, B, 1, D, K]
    ok(torch.equal(d['ebn'][0].reshape(N, B, N1, D, 4), pre), f'{label} c58d: BN channel d * 4 + k holds Linear k, unit d')
    ok(torch.equal(d['ebn'][1].reshape(N, B, N1, D, 4), cur), f'{label} c58d: the embedding neuron receives the BN output as member currents')
    if eval_bn:
        bn = enc.emb_bn
        man = (pre.reshape(-1, D * 4) - bn.running_mean) / torch.sqrt(bn.running_var + bn.eps) * bn.weight + bn.bias
        e = float((man.reshape(cur.shape) - cur).abs().max())
        ok(e <= 1e-5, f'{label} c58d: member current k == BN(Linear_k(signal k)) from the running statistics ({e:.1e})')
    u, errs = cur, []
    for i in range(L):
        Ii = d[f'neu{i}'][0].reshape(N, B, N1, D, 4)
        u = u + d[f'bn{i}'][1].reshape(N, B, N1, D).unsqueeze(-1)
        errs.append(float((Ii - u).abs().max()))
        u = Ii
    ok(max(errs) == 0., f'{label} c58d: layer member currents == embedding member currents + sum BN_j ({", ".join(f"{v:.1e}" for v in errs)})')
    o = d[f'neo{L - 1}'][1]
    ok(binary(o) and binary(d['eneo'][1]) and all(binary(d[f'neo{i}'][1]) for i in range(L)), f'{label} c58d: Neocortex spikes are 0/1')
    q_in = d['q'][0]
    T = q_in.shape[0] // B
    ok(torch.equal(q_in, o.transpose(0, 2).expand(T, -1, -1, -1).reshape(T * B, N, D)), f'{label} c58d: the query Linear reads the last Neocortex spikes')


def main():
    # c58b
    for n in (100, 203):
        x = torch.randn(64, 1, n, generator=torch.Generator().manual_seed(1)).cuda()
        b = band_split(x, 4, 2)
        C = dct_matrix(n, device='cuda', dtype=torch.float64)
        edges = [0] + [int(math.ceil(n / 2 ** (3 - i))) for i in range(3)] + [n]
        leak = 0.
        for i in range(4):
            Xc = dct_time(b[i].double(), C, 2).abs(); msk = torch.ones(n, dtype=torch.bool, device='cuda'); msk[edges[i]:edges[i + 1]] = False
            leak = max(leak, float(Xc[..., msk].max()))
        es, em = float((b.sum(0) - x).abs().max()), float((b[0].mean(-1) - x.mean(-1)).abs().max())
        ok(es <= 5e-6 and leak <= 5e-6 and em <= 1e-6, f'c58b n={n}: edges {edges}; bands add up ({es:.1e}), out-of-band coefficients ({leak:.1e}), mean in band 0 ({em:.1e})')
    # c58c
    g = torch.Generator().manual_seed(2)
    for kind, soma_init in (('pop_lif', 0.5), ('pop_flif', 1.0)):
        nn_ = build_neo_neuron(kind, 16, 24, 'ND', tau=2., alpha=.5, pop_soma_init=soma_init).cuda()
        cur = (torch.randn(24, 5, 1, 16, generator=g) * 2.).cuda()
        with torch.no_grad():
            a = nn_(cur); b = nn_(cur.unsqueeze(-1).expand(-1, -1, -1, -1, 4).contiguous())
        ok(torch.equal(a, b) and float(a.sum()) > 0, f'c58c {kind}: 4 equal member currents == the shared current (spikes {int(a.sum())})')
    nn_ = build_neo_neuron('pop_lif', 16, 24, 'ND', tau=2., pop_soma_init=0.5).cuda()
    with torch.no_grad():
        nn_.neuron.soma.weight.copy_(torch.rand(16, 4, generator=g) + 0.2)
        cur = (torch.randn(24, 5, 1, 16, 4, generator=g) * 3.).cuda()
        s = nn_(cur)
        n_ = nn_.neuron; tau = n_.tau.double(); w = n_.soma.weight.double(); ts, th = n_.soma.tau_s, n_.soma.threshold
        v = torch.zeros(5, 16, 4, dtype=torch.float64, device='cuda'); V = torch.zeros(5, 16, dtype=torch.float64, device='cuda')
        man, margin = [], []
        for t in range(24):
            v = v + (cur[t, :, 0].double() - v) / tau
            V = V + ((w * v).sum(-1) - V) / ts
            sp = (V >= th).double(); margin.append((V - th).abs()); man.append(sp); V = V - th * sp
        man, margin = torch.stack(man), torch.stack(margin)
        diff = (man != s[:, :, 0].double())
    ok(int((diff & (margin > 1e-4)).sum()) == 0 and int(diff.sum()) <= 2 and float(s.sum()) > 0,
       f'c58c pop LIF: member k integrates its own current with tau_k (spikes {int(s.sum())} of {s.numel()}, mismatches {int(diff.sum())}, '
       f'away from the threshold {int((diff & (margin > 1e-4)).sum())})')
    # c58d / c58e / c58g / c58h
    xm = torch.randn(16, 1, 187, generator=torch.Generator().manual_seed(11)).cuda()
    ze = torch.randn(28, 1, 96, generator=torch.Generator().manual_seed(12)).cuda()
    for n, nlab in (('pop_lif', 'pop LIF'), ('pop_flif', 'pop f-LIF')):
        for nm, task, run, ck in runs(n):
            x = xm if task == 'mb' else ze
            sd = torch.load(os.path.join(run, 'model_state', ck), map_location='cpu')
            _, base = build(run, task, {})
            base.load_state_dict(sd); base.eval(); base.train_mode = 'testing'
            cb = Cap(); ib = inner(base); cb.on(ib.time_block.Mem[ib.time_block.num_layers - 1].lif, 'neo')
            with torch.no_grad():
                ob = run_model(base, x)
            cb.off()
            outs, neo, states, outs_b, neo_b, npar, q_b = {}, {}, {}, {}, {}, {}, {}
            for band in BANDS:
                lab = f'{nm} {nlab} {band}'
                _, m = build(run, task, dict(neo_band=band))
                states[band] = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                npar[band] = sum(q.numel() for q in m.parameters())
                m.train(); m.train_mode = 'training'
                wiring(m, x, lab + ' train (fresh)', band, False)
                transplant(m, sd)
                m.eval(); m.train_mode = 'testing'
                wiring(m, x, lab + ' eval (msn weights)', band, True)
                im = inner(m); cap = Cap(); cap.on(im.time_block.Mem[im.time_block.num_layers - 1].lif, 'neo')
                with torch.no_grad():
                    outs[band] = run_model(m, x).detach()
                cap.off(); neo[band] = cap.d['neo'][1]
                im.encoding_neo.emb_bn.train()                          # c58h: batch statistics in the embedding BN only
                mom = im.encoding_neo.emb_bn.momentum; im.encoding_neo.emb_bn.momentum = 0.   # running statistics untouched
                cap = Cap(); cap.on(im.time_block.Mem[im.time_block.num_layers - 1].lif, 'neo'); cap.on(im.data_block[0].ssa.q_lif, 'q')
                with torch.no_grad():
                    outs_b[band] = run_model(m, x).detach()
                cap.off(); neo_b[band] = cap.d['neo'][1]; q_b[band] = cap.d['q'][1]
                im.encoding_neo.emb_bn.momentum = mom
                # c58g
                m.train(); m.train_mode = 'training'; functional.reset_net(m)
                if task == 'mb':
                    o = m(x); o = o[0] if isinstance(o, tuple) else o
                    loss = torch.nn.functional.cross_entropy(o, torch.arange(16, device='cuda') % 5)
                else:
                    o = m(torch.randn(4, 96, 7, generator=torch.Generator().manual_seed(6)).cuda()); loss = torch.nn.functional.l1_loss(o, torch.zeros_like(o))
                loss.backward()
                gr = {k: p.grad for k, p in m.named_parameters() if p.grad is not None}
                el = [k for k in gr if 'encoding_neo.emb_linear.' in k]
                ok(len(el) == 4 and all(bool(torch.isfinite(v).all()) for v in gr.values()) and all(float(gr[k].abs().sum()) > 0 for k in el),
                   f'{lab} c58g: gradients finite on {len(gr)} parameters, non-zero on the 4 embedding Linears '
                   f'({", ".join(f"{float(gr[k].abs().mean()):.1e}" for k in sorted(el))})')
            e = float((outs['same'] - ob).abs().max())
            ok(e <= 1e-5 and torch.equal(neo['same'], cb.d['neo'][1]) and float(cb.d['neo'][1].sum()) > 0,
               f'{nm} {nlab} c58e: same + transplanted msn weights == the msn checkpoint (output diff {e:.1e}, last Neocortex spikes equal, '
               f'{int(cb.d["neo"][1].sum())})')
            pairs = (('freq', 'same'), ('freq', 'rev'), ('rev', 'same'))
            dn = {f'{a}-{b}': int((neo_b[a] != neo_b[b]).sum()) for a, b in pairs}
            dq = {f'{a}-{b}': int((q_b[a] != q_b[b]).sum()) for a, b in pairs}
            do = {f'{a}-{b}': int((outs_b[a] != outs_b[b]).sum()) for a, b in pairs}
            ok(min(dn.values()) > 0 and min(dq.values()) > 0 and do['freq-same'] > 0 and do['rev-same'] > 0 and min(float(neo_b[b].sum()) for b in BANDS) > 0,
               f'{nm} {nlab} c58h: same weights, embedding BN on batch statistics -- Neocortex spikes differ {dn}, query spikes differ {dq}, '
               f'outputs differ {do} (freq-rev printed only); last Neocortex spikes same {int(neo_b["same"].sum())}, freq {int(neo_b["freq"].sum())}, '
               f'rev {int(neo_b["rev"].sum())} (with the msn running statistics: {int(neo["same"].sum())}, {int(neo["freq"].sum())}, {int(neo["rev"].sum())})')
            # c58f
            sf = states['freq']
            same3 = all(states[b].keys() == sf.keys() and all(torch.equal(states[b][k], sf[k]) for k in sf) for b in ('same', 'rev'))
            sb = {k: v for k, v in base.state_dict().items()}
            only = sorted(set(sf) ^ set(sb))
            shapes = all(sf[k].shape == sb[k].shape for k in set(sf) & set(sb) if 'encoding_neo.emb_bn' not in k)
            pb, pf = sum(q.numel() for q in base.parameters()), npar['freq']
            ok(same3 and shapes and all('encoding_neo.emb_linear' in k for k in only) and len(set(npar.values())) == 1,
               f'{nm} {nlab} c58f: freq / same / rev share {len(sf)} state_dict entries and init; against msn only encoding_neo.emb_linear '
               f'/ emb_bn differ; parameters {pb} -> {pf} (+{pf - pb})')
    print('\nALL PASS' if not FAIL else f'\n{len(FAIL)} FAILED: {FAIL}')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
