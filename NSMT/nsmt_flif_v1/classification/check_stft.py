"""MODI 2026-11-13: fail-closed checks of the STFT input of the population Neocortex and of the soma-less (firing-member) population
(--neo_band stft, --neo_pop fire; docs/MODI_PROJECT_LOG.md 2026-11-13/14, user request 11-13). c59a (every earlier path bit-identical)
is fw_identity_ms.py + fw_identity_band.py + fw_identity_bandvar.py.
  c59b stft_bands == torch.stft (periodic Hann, n_fft = patch, hop = stride, no centring, orthonormal): frames = tokens, bins grouped by
       band_edges, real parts then imaginary parts (bin 0 / Nyquist imaginary parts left out); the band sizes add up to the patch size
  c59c neuron, no soma: with per-member currents the member spikes are exactly those of K separate single neurons (kind 'lif' /
       'flif', the same tau) fed the member's current, and the unit output is their OR; 0/1
  c59d wiring (hooks; fresh model, training mode; 2 neurons x 3 sizes x 3 variants): member time constants; which band each embedding
       Linear reads (stft: lowest band -> largest tau, with equal taus band i -> member i; fire_same: the unsplit tokens), recomputed
       with torch.stft; member currents == the embedding BN output; layer member currents == embedding member currents + sum_j BN_j;
       soma variants: neuron output [.., D]; fire variants: neuron output [.., D * K] and the layer output == OR over the members;
       spikes 0/1; the query Linear reads the last Neocortex layer's (OR-ed) spikes
  c59e parameters: stft variants keep the base's embedding weight count (band sizes add up to the patch size); counts printed
  c59f gradients (training mode): finite everywhere, non-zero on all 4 embedding Linears
  c59g the Neocortex emits spikes and every band matters: zeroing one band's input changes the Neocortex spikes (fresh model, embedding
       BN gain raised so that the fresh Neocortex fires; a fresh soma population is otherwise silent, as c55c / c57f / c58h found)
  any failed comparison or empty capture raises (exit code 1).
usage (classification folder): CUDA_VISIBLE_DEVICES=0 python check_stft.py
"""
import glob
import os
import sys
from types import SimpleNamespace as NS

import torch

sys.path.insert(0, os.getcwd())
from spikingjelly.clock_driven import functional                        # noqa: E402
from load_model import LOAD_MODEL                                        # noqa: E402
from forecast_wrap import Forecaster                                     # noqa: E402
import model as model_mod                                                # noqa: E402
from model import band_edges, stft_band_dims, stft_bands, stft_member_order   # noqa: E402
from layers import build_neo_neuron                                      # noqa: E402

VARIANTS = {'stft': dict(neo_band='stft'), 'fire_stft': dict(neo_band='stft', neo_pop='fire', pop_taus='2,2,2,2'),
            'fire_same': dict(neo_band='same', neo_pop='fire', pop_taus='2,2,2,2')}
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
    return (('MIT-BIH', 'mb', one(f'log/flif-v1-mitbih-ms/MITBIH/*/*+nsmt4q_{n}_msn_lr0.01_v0+seed42+model+*/seed42_*')),
            ('ETTm2 H96', 'ett', one(f'../forecasting/log/flif-v1-ett-ms/ETTm2/p96/*/*+nsmt4q_{n}_msn+seed42+*/seed42_*')),
            ('ETTh2 H96', 'ett', one(f'../forecasting/log/flif-v1-ett-ms/ETTh2/p96/*/*+nsmt4q_{n}_msn+seed42+*/seed42_*')))


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


def ref_bands(im, x):
    """torch.stft on the padded signal (1 channel), grouped as band_edges: list of 4 tensors [N, B, 1, dim_i]."""
    xp = im.padding_patch_layer(x)
    P, S = im.patch_size, im.stride
    X = torch.stft(xp[:, 0], n_fft=P, hop_length=S, win_length=P, window=torch.hann_window(P, periodic=True, device=xp.device),
                   center=False, normalized=True, onesided=True, return_complex=True)                    # [B, P/2+1, N]
    e = band_edges(P // 2 + 1)
    out = []
    for i in range(4):
        ks = list(range(e[i], e[i + 1])); im_ = [k for k in ks if k not in (0, P // 2)]
        out.append(torch.cat([X[:, ks].real, X[:, im_].imag], dim=1).permute(2, 0, 1).unsqueeze(2))
    return out


def tokens(im, x):
    xp = im.padding_patch_layer(x)
    p = xp.unfold(dimension=-1, size=im.patch_size, step=im.stride)
    return p.permute(0, 2, 1, 3).reshape(xp.shape[0], p.shape[2], -1).transpose(0, 1).unsqueeze(2)


def wiring(m, x, label, v, taus):
    im = inner(m); cap = Cap(); enc = im.encoding_neo
    fire = v.startswith('fire')
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
    got = enc.emb_lif.neuron.tau.detach().cpu().tolist()
    ok(got == list(taus) and all(im.time_block.Mem[i].lif.neuron.tau.detach().cpu().tolist() == list(taus) for i in range(L))
       and enc.emb_lif.neuron.has_soma == (not fire) and all(im.time_block.Mem[i].lif.neuron.has_soma == (not fire) for i in range(L)),
       f'{label} c59d: member time constants {got}, soma {not fire}')
    if 'stft' in v:
        rb = ref_bands(im, x)
        order = [3, 2, 1, 0] if list(taus) == [4., 8., 16., 32.] else [0, 1, 2, 3]            # member that reads band i
        ok(stft_member_order(taus) == order, f'{label} c59d: band i -> member {order}')
        exp = [None] * 4
        for i, mth in enumerate(order):
            exp[mth] = rb[i]
        names = 'STFT band (lowest band -> member %d)' % order[0]
    else:
        exp = [tokens(im, x)] * 4
        names = 'the unsplit tokens'
    shp = all(d[f'lin{k}'][0].shape == exp[k].shape for k in range(4))
    e = max(float((d[f'lin{k}'][0] - exp[k]).abs().max()) for k in range(4)) if shp else float('inf')
    ok(shp and e <= 1e-6, f'{label} c59d: embedding Linear k reads {names}, sizes {[exp[k].shape[-1] for k in range(4)]} (max diff {e:.1e})')
    N, B, N1, D = d['lin0'][1].shape
    cur = d['eneu'][0].reshape(N, B, N1, D, 4)
    pre = torch.stack([d[f'lin{k}'][1] for k in range(4)], dim=-1)
    ok(torch.equal(d['ebn'][0].reshape(N, B, N1, D, 4), pre) and torch.equal(d['ebn'][1].reshape(N, B, N1, D, 4), cur),
       f'{label} c59d: BN channel d * 4 + k holds Linear k of unit d, and its output is the member current')
    u, errs = cur, []
    for i in range(L):
        Ii = d[f'neu{i}'][0].reshape(N, B, N1, D, 4)
        u = u + d[f'bn{i}'][1].reshape(N, B, N1, D).unsqueeze(-1)
        errs.append(float((Ii - u).abs().max()))
        u = Ii
    ok(max(errs) == 0., f'{label} c59d: layer member currents == embedding member currents + sum BN_j ({", ".join(f"{z:.1e}" for z in errs)})')
    outs = [('eneu', 'eneo')] + [(f'neu{i}', f'neo{i}') for i in range(L)]
    if fire:
        good = all(d[a][1].shape[-1] == D * 4 and torch.equal(d[b][1], d[a][1].reshape(N, B, N1, D, 4).amax(-1)) for a, b in outs)
        ok(good, f'{label} c59d: every Neocortex neuron emits D * 4 member spike trains and the layer output is their OR')
    else:
        ok(all(d[a][1].shape[-1] == D for a, b in outs), f'{label} c59d: every Neocortex neuron emits D soma spike trains')
    o = d[f'neo{L - 1}'][1]
    ok(all(binary(d[a][1]) and binary(d[b][1]) for a, b in outs), f'{label} c59d: Neocortex spikes are 0/1')
    q_in = d['q'][0]
    T = q_in.shape[0] // B
    ok(torch.equal(q_in, o.transpose(0, 2).expand(T, -1, -1, -1).reshape(T * B, N, D)), f'{label} c59d: the query Linear reads the last Neocortex spikes')


def neo_spikes(m, x, zero_band=None):
    """Last Neocortex layer's output spikes; zero_band = index of the member whose embedding input is replaced by zeros."""
    im = inner(m); cap = Cap(); cap.on(im.time_block.Mem[im.time_block.num_layers - 1].lif, 'neo')
    h = None
    if zero_band is not None:
        h = im.encoding_neo.emb_linear[zero_band].register_forward_pre_hook(lambda _m, i: (torch.zeros_like(i[0]),))
    with torch.no_grad():
        run_model(m, x)
    cap.off()
    if h is not None:
        h.remove()
    return cap.d['neo'][1]


def main():
    assert os.path.dirname(os.path.abspath(model_mod.__file__)) == os.getcwd()
    # c59b
    for P, n in ((8, 100), (32, 203)):
        x = torch.randn(16, 1, n, generator=torch.Generator().manual_seed(1)).cuda()
        b = stft_bands(x, P, P // 2)
        X = torch.stft(x[:, 0], n_fft=P, hop_length=P // 2, win_length=P, window=torch.hann_window(P, periodic=True, device='cuda'),
                       center=False, normalized=True, onesided=True, return_complex=True)
        e = band_edges(P // 2 + 1); err = 0.
        for i in range(4):
            ks = list(range(e[i], e[i + 1])); im_ = [k for k in ks if k not in (0, P // 2)]
            want = torch.cat([X[:, ks].real, X[:, im_].imag], dim=1).permute(2, 0, 1).unsqueeze(2)
            err = max(err, float((b[i] - want).abs().max()) if b[i].shape == want.shape else float('inf'))
        dims = stft_band_dims(P)
        fr = x.unfold(-1, P, P // 2)
        ok(err <= 1e-6 and sum(dims) == P and [z.shape[-1] for z in b] == dims and b[0].shape[0] == fr.shape[2],
           f'c59b patch {P}: stft_bands == torch.stft (max diff {err:.1e}); {P // 2 + 1} bins, edges {e}, band sizes {dims} (sum {sum(dims)}), '
           f'{b[0].shape[0]} frames = tokens')
    # c59c
    g = torch.Generator().manual_seed(2)
    for kind, single in (('pop_lif', 'lif'), ('pop_flif', 'flif')):
        pop = build_neo_neuron(kind, 16, 24, 'ND', tau=2., alpha=.5, pop_taus=(2., 2., 2., 2.), pop_soma='none').cuda()
        one_ = build_neo_neuron(single, 16, 24, 'ND', tau=2., alpha=.5).cuda()
        cur = (torch.randn(24, 5, 1, 16, 4, generator=g) * 2.).cuda()
        with torch.no_grad():
            cap = Cap(); cap.on(pop.neuron, 'n'); out = pop(cur); cap.off()
            mem = cap.d['n'][1].reshape(24, 5, 16, 4)
            sep = torch.stack([one_(cur[..., k]) for k in range(4)], dim=-1)[:, :, 0]         # [24, 5, 16, 4]
        ok(torch.equal(mem, sep) and torch.equal(out[:, :, 0], sep.amax(-1)) and binary(out) and float(mem.sum()) > 0
           and len({int(mem[..., k].sum()) for k in range(4)}) > 1,
           f'c59c {kind} without soma: member spikes == 4 separate {single} neurons (tau 2) on their own currents, output == their OR '
           f'(member spikes {[int(mem[..., k].sum()) for k in range(4)]}, OR {int(out.sum())})')
    # c59d-g
    xm = torch.randn(16, 1, 187, generator=torch.Generator().manual_seed(11)).cuda()
    ze = torch.randn(28, 1, 96, generator=torch.Generator().manual_seed(12)).cuda()
    for n, nlab, nov in (('pop_lif', 'pop LIF', {}), ('pop_flif', 'pop f-LIF', {})):
        for nm, task, run in runs(n):
            x = xm if task == 'mb' else ze
            _, base = build(run, task, {})
            pb = sum(q.numel() for q in base.parameters())
            eb = sum(q.numel() for q in inner(base).encoding_neo.emb_linear.parameters())
            for v, ov in VARIANTS.items():
                lab = f'{nm} {nlab} {v}'
                taus = (2., 2., 2., 2.) if v.startswith('fire') else (4., 8., 16., 32.)
                a, m = build(run, task, ov)
                ok(a.neo_neuron == n, f'{lab}: neuron {a.neo_neuron}')
                m.train(); m.train_mode = 'training'
                wiring(m, x, lab + ' train (fresh)', v, taus)
                im = inner(m)
                pv = sum(q.numel() for q in m.parameters()); ev = sum(q.numel() for q in im.encoding_neo.emb_linear.parameters())
                ok((ev == eb) if 'stft' in v else (ev == 4 * eb),
                   f'{lab} c59e: embedding Linear weights {ev} (base {eb}); parameters {pv} (base {pb}, {pv - pb:+d})')
                # c59g: raise the embedding BN gain so that the fresh Neocortex fires, then zero one member's input at a time
                with torch.no_grad():
                    im.encoding_neo.emb_bn.weight.mul_(4.)
                s0 = neo_spikes(m, x)
                ch = [int((neo_spikes(m, x, zero_band=k) != s0).sum()) for k in range(4)]
                ok(float(s0.sum()) > 0 and min(ch) > 0,
                   f'{lab} c59g: Neocortex spikes {int(s0.sum())} of {s0.numel()} (embedding BN gain x4); zeroing member k\'s input changes {ch} positions')
                with torch.no_grad():
                    im.encoding_neo.emb_bn.weight.div_(4.)
                # c59f
                functional.reset_net(m)
                if task == 'mb':
                    o = m(x); o = o[0] if isinstance(o, tuple) else o
                    loss = torch.nn.functional.cross_entropy(o, torch.arange(16, device='cuda') % 5)
                else:
                    o = m(torch.randn(4, 96, 7, generator=torch.Generator().manual_seed(6)).cuda()); loss = torch.nn.functional.l1_loss(o, torch.zeros_like(o))
                loss.backward()
                gr = {k: p.grad for k, p in m.named_parameters() if p.grad is not None}
                el = [k for k in gr if 'encoding_neo.emb_linear.' in k]
                ok(len(el) == 4 and all(bool(torch.isfinite(z).all()) for z in gr.values()) and all(float(gr[k].abs().sum()) > 0 for k in el),
                   f'{lab} c59f: gradients finite on {len(gr)} parameters, non-zero on the 4 embedding Linears '
                   f'({", ".join(f"{float(gr[k].abs().mean()):.1e}" for k in sorted(el))})')
    print('\nALL PASS' if not FAIL else f'\n{len(FAIL)} FAILED: {FAIL}')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
