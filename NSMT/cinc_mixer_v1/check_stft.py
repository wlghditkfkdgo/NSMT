"""Checks of the STFT band test (2026-11-16, PROJECT_LOG, exp/f-lif-pop-v3). Fail closed: any failed item exits 1.

  --snapshot OUT --code DIR   tensors of the EARLIER paths (no new option used), written from the code in DIR; run once on the
                              export of the commit before the edit and once on the working tree (c0 compares the two files)
  --compare A B               c0: every entry of the two snapshots is bitwise equal
  (no option)                 c1 STFT = torch.stft, band sizes;  c2 A wiring;  c3 B wiring and the per-member neuron;
                              c4 spiking-axis layouts;  c5 gradients;  c6 gain tables load and match the reference rates
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OK, BAD = [], []


def chk(name, cond, info=''):
    (OK if cond else BAD).append(name)
    print(('pass  ' if cond else 'FAIL  ') + name + (f'  [{info}]' if info else ''), flush=True)


def snapshot(out, code):
    sys.path.insert(0, code)
    from config import set_random_seed
    from spk_mixer import SpkMixer
    from neo_mixer import NeoMixer
    res = {}
    cfgs = {
        'ett_lif': dict(seq_len=96, data_patch_size=8, embed_dim=64, mlp_ratios=0.5, mixer_layers=1, connect_f='IAND', mixer='lif', mixer_comb='muxlin'),
        'mit_lif': dict(seq_len=187, data_patch_size=32, embed_dim=384, mlp_ratios=1.0, mixer_layers=2, connect_f='ADD', mixer='lif', mixer_comb='muxlin'),
        'mit_flif': dict(seq_len=187, data_patch_size=32, embed_dim=384, mlp_ratios=1.0, mixer_layers=2, connect_f='OR', mixer='flif', mixer_comb='sdlin'),
        'mit_ms': dict(seq_len=187, data_patch_size=32, embed_dim=384, mlp_ratios=1.0, mixer_layers=2, connect_f='MS', mixer='lif', mixer_comb='muxlin'),
        'cinc_pop_lif': dict(seq_len=1500, data_patch_size=64, embed_dim=384, mlp_ratios=1.0, mixer_layers=2, connect_f='IAND', mixer='pop_lif',
                             mixer_comb='sdlin', pop_soma_init=0.5, num_classes=4),
        'cinc_pop_flif': dict(seq_len=1500, data_patch_size=64, embed_dim=384, mlp_ratios=1.0, mixer_layers=2, connect_f='OR', mixer='pop_flif',
                              mixer_comb='sdor', pop_soma_init=1.0, num_classes=4),
        'cinc_popg_lif': dict(seq_len=1500, data_patch_size=64, embed_dim=384, mlp_ratios=1.0, mixer_layers=2, connect_f='IAND', mixer='popg_lif',
                              mixer_comb='or', num_classes=4),
    }
    for key, kw in cfgs.items():
        set_random_seed(42)
        kw = dict(dict(num_classes=5, depths=1, T_h=4, bias=False, in_channels=1), **kw)
        m = SpkMixer(**kw).cuda()
        res[f'{key}/rng'] = torch.get_rng_state().clone()
        g = torch.Generator().manual_seed(1)
        x = torch.randn(6, 1, kw['seq_len'], generator=g).cuda()
        m.train(); m.train_mode = 'training'
        z = m(x); (z ** 2).sum().backward()
        res[f'{key}/train_out'] = z.detach().cpu()
        for n, p in m.named_parameters():
            res[f'{key}/w/{n}'] = p.detach().cpu().clone()
            res[f'{key}/g/{n}'] = None if p.grad is None else p.grad.detach().cpu().clone()
        m.eval(); m.train_mode = 'testing'
        with torch.no_grad():
            z, h = m(x)
        res[f'{key}/eval_out'] = z.cpu(); res[f'{key}/eval_h'] = h.cpu()
        del m
    set_random_seed(42)                                                     # the Neo-mixer shares layers.NeoNeuron / neurons.py
    m = NeoMixer(seq_len=187, data_patch_size=32, embed_dim=384, num_classes=5, depths=1, mlp_ratios=1.0, mixer='pop_lif', mixer_bidir=True,
                 mixer_layers=2, pop_soma_init=0.5).cuda()
    x = torch.randn(6, 1, 187, generator=torch.Generator().manual_seed(1)).cuda()
    m.train(); m.train_mode = 'training'
    z = m(x); z = z[0] if isinstance(z, tuple) else z
    (z ** 2).sum().backward()
    res['neomixer/train_out'] = z.detach().cpu()
    for n, p in m.named_parameters():
        res[f'neomixer/g/{n}'] = None if p.grad is None else p.grad.detach().cpu().clone()
    torch.save(res, out)
    print(f'{len(res)} entries -> {out}')


def compare(a, b):
    A, B = torch.load(a), torch.load(b)
    chk('c0 same entries', set(A) == set(B) and len(A) > 0, f'{len(A)} / {len(B)}')
    diff = [k for k in A if k in B and not ((A[k] is None and B[k] is None) or (A[k] is not None and B[k] is not None and torch.equal(A[k], B[k])))]
    chk('c0 earlier paths bitwise equal (weights, random state, training output, gradients, eval output)', not diff, f'{len(A)} entries, {len(diff)} differ {diff[:3]}')


def main():
    sys.path.insert(0, HERE)
    import stft_common as sc
    from layers import NeoNeuron, build_neo_neuron
    from spk_band import stft_values, stft_band_dims, band_edges, member_order
    from spk_mixer import apply_gain_table
    dev = 'cuda'
    # ---- c1 STFT -------------------------------------------------------------------------------------------------------
    chk('c1 band sizes', stft_band_dims(8) == [1, 2, 2, 3] and stft_band_dims(32) == [5, 4, 8, 15], f'{stft_band_dims(8)} {stft_band_dims(32)}')
    chk('c1 band edges', band_edges(5) == [0, 1, 2, 3, 5] and band_edges(17) == [0, 3, 5, 9, 17])
    for task in ('ETTh2', 'MITBIH'):
        m = sc.build(task, 'lif', 'group').to(dev)
        x = sc.batch(task, 8).to(dev)
        P, st = m.patch_size, m.stride
        tok = m.tokens(x)                                                   # [T_h, B, N, P]
        bands = stft_values(tok)
        xp = m.padding_patch_layer(x)[:, 0]
        ref = torch.stft(xp, n_fft=P, hop_length=st, win_length=P, window=torch.hann_window(P, periodic=True, device=dev), center=False,
                         normalized=True, onesided=True, return_complex=True).transpose(1, 2)        # [B, N, bins]
        e = band_edges(P // 2 + 1)
        worst = 0.
        for i in range(4):
            ks = list(range(e[i], e[i + 1])); im = [k for k in ks if k not in (0, P // 2)]
            r = torch.cat([ref[..., ks].real, ref[..., im].imag], -1)
            worst = max(worst, float((bands[i][0] - r).abs().max()))
            chk(f'c1 {task} band {i}: every repeat holds the same values', all(torch.equal(bands[i][0], bands[i][t]) for t in range(tok.shape[0])))
        chk(f'c1 {task} stft_values == torch.stft', worst < 1e-5 and ref.shape[1] == m.num_patches, f'max diff {worst:.1e}, frames {ref.shape[1]} = tokens {m.num_patches}')
        full = torch.cat(bands, -1)                                          # the transform keeps the frame: energy of the windowed frame
        win = torch.hann_window(P, periodic=True, device=dev)
        en = ((tok * win) ** 2).sum(-1)
        rec = (full ** 2).sum(-1) * 2 - (bands[0][..., 0] ** 2) - (torch.cat(bands, -1)[..., sum(stft_band_dims(P)[:3]) + len(range(e[3], e[4])) - 1] ** 2)
        chk(f'c1 {task} Parseval (windowed frame energy)', float((en - rec).abs().max()) < 1e-3 * float(en.abs().max() + 1e-6), f'{float((en - rec).abs().max()):.1e}')
    # ---- c2 A: channel groups ------------------------------------------------------------------------------------------
    for task in ('ETTh2', 'ETTm2', 'MITBIH'):
        for taxis in ('rep', 'token'):
            base = sc.build(task, 'lif', 'none', taxis, seed=42); rng0 = torch.get_rng_state().clone()
            a = sc.build(task, 'lif', 'group', taxis, seed=42); rng1 = torch.get_rng_state().clone()
            sb, sa = base.state_dict(), a.state_dict()
            shared = [k for k in sa if k in sb and 'emb_linear' not in k]
            chk(f'c2 {task} {taxis}: A shares every other weight and the random state with the band-less model',
                torch.equal(rng0, rng1) and all(torch.equal(sa[k], sb[k]) for k in shared) and set(sb) - set(sa) == {'encoding.emb_linear.weight'},
                f'{len(shared)} shared tensors')
            P, D = a.patch_size, a.encoding.embed_dim
            dims = stft_band_dims(P)
            chk(f'c2 {task} {taxis}: A embedding weights = sum(dim_k) * D / 4', sum(l.weight.numel() for l in a.encoding.emb_linear) == P * D // 4
                and [tuple(l.weight.shape) for l in a.encoding.emb_linear] == [(D // 4, d) for d in dims])
            a = a.to(dev).train(); x = sc.batch(task, 8).to(dev)
            tok = stft_values(a.tokens(x))
            if taxis == 'token':
                tok = [t.permute(2, 1, 0, 3).contiguous() for t in tok]
            a.eval()                                                        # running statistics: one band cannot reach other groups through BN
            c0 = a.encoding.current(tok)
            okg = True
            for j in range(4):
                t2 = [t.clone() for t in tok]; t2[j] = t2[j] + 1.
                d = (a.encoding.current(t2) - c0).abs().reshape(-1, 4, D // 4).amax((0, 2))
                okg &= bool(d[j] > 0) and all(float(d[i]) == 0. for i in range(4) if i != j)
            chk(f'c2 {task} {taxis}: band j reaches only channel group j', okg)
            caps = []
            hs = [l.register_forward_hook(lambda md, i, o: caps.append(i[0].detach())) for l in a.encoding.emb_linear]
            a.train(); a.forward_parts(x)
            for h in hs: h.remove()
            chk(f'c2 {task} {taxis}: Linear k reads band k', len(caps) == 4 and all(torch.equal(caps[k], tok[k]) for k in range(4)))
    # ---- c3 B: one band per member -------------------------------------------------------------------------------------
    for kind in ('pop_lif',):
        n = build_neo_neuron(kind, 16, 24, 'ND', tau=2., pop_taus=(4., 8., 16., 32.), pop_soma_init=0.5).to(dev)
        g = torch.Generator().manual_seed(3)
        cur = (torch.randn(24, 5, 7, 16, 4, generator=g) * 3.).to(dev)
        s = n(cur)                                                           # [T, B, N', D]
        taus = torch.tensor([4., 8., 16., 32.], device=dev); w = n.neuron.soma.weight
        v = torch.zeros(5, 7, 16, 4, device=dev); V = torch.zeros(5, 7, 16, device=dev); out = []
        for t in range(24):                                                  # written from the definition: leaky members, one soma
            v = v + (cur[t] - v) / taus
            a_t = (w * v).sum(-1)
            V = V + (a_t - V) / 2.
            sp = (V >= 1.).float(); out.append(sp); V = V - sp
        out = torch.stack(out)
        chk(f'c3 {kind}: per-member currents == members and soma written from the definition', float((s - out).abs().sum()) <= 2 and float(s.mean()) > 0.01,
            f'{int((s - out).abs().sum())} of {s.numel()} differ, rate {float(s.mean()):.3f}')
        same = cur[..., :1].expand(-1, -1, -1, -1, 4).contiguous()
        chk(f'c3 {kind}: equal currents to all members == the ordinary call', torch.equal(n(same), n(cur[..., 0].contiguous())))
        c2 = cur.clone(); c2[..., 1] = c2[..., 1] + 5.
        chk(f'c3 {kind}: one member\'s current changes the spikes', not torch.equal(n(c2), s))
    chk('c3 member order: lowest band -> slowest member', member_order((4., 8., 16., 32.)) == [3, 2, 1, 0] and member_order((2., 2., 2., 2.)) == [0, 1, 2, 3])
    for task in ('ETTh2', 'ETTm2', 'MITBIH'):
        for taxis in ('rep', 'token'):
            base = sc.build(task, 'pop_lif', 'none', taxis, seed=42); rng0 = torch.get_rng_state().clone()
            b = sc.build(task, 'pop_lif', 'member', taxis, seed=42); rng1 = torch.get_rng_state().clone()
            sb, sa = base.state_dict(), b.state_dict()
            shared = [k for k in sa if k in sb and 'emb_linear' not in k and 'emb_bn' not in k]
            chk(f'c3 {task} {taxis}: B shares every other weight and the random state with the band-less population model',
                torch.equal(rng0, rng1) and all(torch.equal(sa[k], sb[k]) for k in shared), f'{len(shared)} shared tensors')
            P, D = b.patch_size, b.encoding.embed_dim
            chk(f'c3 {task} {taxis}: B embedding weights = P * D (as the band-less embedding)', sum(l.weight.numel() for l in b.encoding.emb_linear) == P * D
                and b.encoding.emb_bn.weight.numel() == 4 * D)
            chk(f'c3 {task} {taxis}: member taus 4 / 8 / 16 / 32, soma tau 2', b.encoding.emb_lif.neuron.tau.tolist() == [4., 8., 16., 32.]
                and b.encoding.emb_lif.neuron.soma.tau_s == 2. and b.encoding.emb_lif.neuron.has_soma)
            b = b.to(dev).eval(); x = sc.batch(task, 8).to(dev)
            tok = stft_values(b.tokens(x))
            if taxis == 'token':
                tok = [t.permute(2, 1, 0, 3).contiguous() for t in tok]
            c0 = b.encoding.current(tok)
            okm = True
            for i in range(4):
                t2 = [t.clone() for t in tok]; t2[i] = t2[i] + 1.
                d = (b.encoding.current(t2) - c0).abs().amax((0, 1, 2, 3))   # per member
                okm &= bool(d[3 - i] > 0) and all(float(d[k]) == 0. for k in range(4) if k != 3 - i)
            chk(f'c3 {task} {taxis}: band i reaches only member 3 - i (band 0 -> tau 32)', okm)
            seen = []
            h = b.encoding.emb_lif.neuron.register_forward_hook(lambda md, i, o: seen.append((tuple(i[0].shape), tuple(o.shape))))
            b.train(); b.forward_parts(x); h.remove()
            T = b.num_patches if taxis == 'token' else 4
            chk(f'c3 {task} {taxis}: the embedding neuron receives [T, rows, D, 4] and returns [T, rows, D]', seen[0][0][0] == T and seen[0][0][2:] == (D, 4)
                and seen[0][1][2:] == (D,), str(seen[0]))
    # ---- c4 spiking-axis layouts ---------------------------------------------------------------------------------------
    for task in ('ETTh2', 'MITBIH'):
        for mixer, band in (('lif', 'none'), ('lif', 'group'), ('pop_lif', 'none'), ('pop_lif', 'member')):
            x = sc.batch(task, 8).to(dev)
            for taxis in ('rep', 'token'):
                gt = sc.table_path(task, mixer, band, taxis) if mixer == 'pop_lif' else None
                m = sc.build(task, mixer, band, taxis, seed=42, tau_calib=mixer == 'pop_lif', gain_table=gt).to(dev).train()
                N = m.num_patches
                steps = {}
                hs = [mod.register_forward_hook(lambda md, i, o, n=name: steps.__setitem__(n, i[0].shape[0])) for name, mod in m.named_modules()
                      if isinstance(mod, NeoNeuron)]
                z, e, m0, f, h = m.forward_parts(x)
                for hk in hs: hk.remove()
                mix = {k: v for k, v in steps.items() if 'mixer' in k or 'sd_neuron' in k or 'fuse' in k}
                oth = {k: v for k, v in steps.items() if k not in mix}
                want = N if taxis == 'token' else 4
                chk(f'c4 {task} {mixer} {band} {taxis}: mixer neurons step over {N} tokens, embedding / MLP neurons over {want}',
                    all(v == N for v in mix.values()) and all(v == want for v in oth.values()) and len(oth) == 3, f'{len(mix)} + {len(oth)} neuron layers')
                T = 1 if taxis == 'token' else 4
                binary = bool(((e == 0) | (e == 1)).all()) and (m.connect_f == 'ADD' or bool(((h == 0) | (h == 1)).all()))
                chk(f'c4 {task} {mixer} {band} {taxis}: outputs [T_h {T}, rows, N, D], spikes 0 / 1, finite logits',
                    tuple(e.shape) == (T, x.shape[0], N, m.encoding.embed_dim) and tuple(h.shape) == tuple(e.shape) and tuple(m0.shape) == tuple(e.shape)
                    and binary and bool(torch.isfinite(z).all()) and float(e.mean()) > 0, f'embedding rate {float(e.mean()):.4f}, output rate {float(h.mean()):.4f}')
                # memory across tokens: the embedding output at token n depends on earlier tokens only in the token layout, never on later ones
                with torch.no_grad():
                    for bn in [mm for mm in m.modules() if isinstance(mm, torch.nn.BatchNorm1d)]:
                        bn.momentum = 1.                                     # running statistics of a fresh model do not fire it (11-08 check m2;
                    m.forward_parts(x)                                       # 40 passes at momentum 0.1 still leave 1.5 % of the initial variance 1,
                    m.eval()                                                 # which is larger than the real one): set them to this batch's statistics
                    cap = {}
                    hk = m.encoding.emb_lif.register_forward_hook(lambda md, i, o: cap.update(i=i[0].detach(), o=o.detach()))
                    e0 = m.forward_parts(x)[1]
                    hk.remove()
                    # the same currents, one call per step (no state carried between steps): differs from the single call only if the
                    # neuron's time axis carries state; in the token layout that axis is the token axis
                    sep = torch.cat([m.encoding.emb_lif(cap['i'][t:t + 1].contiguous()) for t in range(cap['i'].shape[0])])
                    carried = int((sep != cap['o']).sum())
                    g = torch.Generator().manual_seed(5)
                    L = x.shape[-1]; x1 = x.clone(); x1[..., :m.stride] = 10. * torch.randn(x.shape[0], 1, m.stride, generator=g).to(dev)   # only token 0 changes
                    x2 = x.clone(); x2[..., L - 1:] += 10.                             # only the last tokens change
                    e1, e2 = m.forward_parts(x1)[1], m.forward_parts(x2)[1]
                later = bool((e1[:, :, 1:] != e0[:, :, 1:]).any()); first = bool((e1[:, :, 0] != e0[:, :, 0]).any())
                early = bool((e2[:, :, :N - 2] != e0[:, :, :N - 2]).any())
                if taxis == 'token':
                    chk(f'c4 {task} {mixer} {band} token: the embedding neuron carries state from token to token (one call over N steps != N one-step calls); '
                        'earlier tokens never change with the last one', carried > 0 and cap['i'].shape[0] == N and not early,
                        f'{carried} spikes differ, later tokens changed by token 0: {later}, early {early}')
                else:
                    chk(f'c4 {task} {mixer} {band} rep: embedding spikes of a token depend on that token only (token 0 changed: itself yes, later no; last changed: earlier no); '
                        'state is carried over the 4 repeats', first and not later and not early and carried > 0 and cap['i'].shape[0] == 4,
                        f'first {first} later {later} early {early}, {carried} spikes differ from one-step calls')
    # ---- c5 gradients --------------------------------------------------------------------------------------------------
    for task in ('ETTh2', 'MITBIH'):
        x = sc.batch(task, 16).to(dev)
        for mixer, band in (('lif', 'group'), ('pop_lif', 'member')):
            for taxis in ('rep', 'token'):
                gt = sc.table_path(task, mixer, band, taxis) if mixer == 'pop_lif' else None
                m = sc.build(task, mixer, band, taxis, seed=42, tau_calib=mixer == 'pop_lif', gain_table=gt).to(dev).train()
                z = m.forward_parts(x)[0]; (z ** 2).sum().backward()
                gs = [l.weight.grad for l in m.encoding.emb_linear]
                fin = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters())
                chk(f'c5 {task} {mixer} {band} {taxis}: finite gradients, all 4 band Linears receive one', fin and all(g is not None and float(g.abs().sum()) > 0 for g in gs),
                    ' '.join(f'{float(g.abs().mean()):.1e}' for g in gs))
    # ---- c6 gain tables ------------------------------------------------------------------------------------------------
    for task in sc.TASKS:
        x = sc.batch(task, 32).to(dev)
        for taxis in ('rep', 'token'):
            ref = sc.build(task, 'lif', 'none', taxis, seed=0).to(dev).train()
            rr = {}
            hs = [mod.register_forward_hook(lambda md, i, o, n=name: rr.__setitem__(n, float(o.float().mean()))) for name, mod in ref.named_modules() if isinstance(mod, NeoNeuron)]
            with torch.no_grad():
                ref.forward_parts(x)
            for hk in hs: hk.remove()
            for band in ('none', 'member'):
                m = sc.build(task, 'pop_lif', band, taxis, seed=0, tau_calib=True, gain_table=sc.table_path(task, 'pop_lif', band, taxis)).to(dev).train()
                pr = {}
                hs = [mod.register_forward_hook(lambda md, i, o, n=name: pr.__setitem__(n, float(o.float().mean()))) for name, mod in m.named_modules() if isinstance(mod, NeoNeuron)]
                with torch.no_grad():
                    m.forward_parts(x)
                for hk in hs: hk.remove()
                worst = max(abs(pr[k] - rr[k]) / rr[k] for k in rr)
                chk(f'c6 {task} pop_lif {band} {taxis}: table loads, every layer fires at the LIF rate', m.tau_calibrated and set(pr) == set(rr) and worst < 0.10,
                    f'{len(pr)} layers, worst relative difference {worst:.3f}')
    try:
        sc.build('ETTh2', 'pop_lif', 'member', 'rep', tau_calib=True)
        chk('c6 a population model with band / taxis and no table is refused', False)
    except ValueError:
        chk('c6 a population model with band / taxis and no table is refused', True)
    for kw, name in ((dict(mixer='pop_lif', band='group'), 'group with a population'), (dict(mixer='lif', band='member'), 'member with LIF')):
        try:
            sc.build('ETTh2', kw['mixer'], kw['band']); chk(f'c6 refused: {name}', False)
        except ValueError:
            chk(f'c6 refused: {name}', True)
    try:
        sc.build('ETTh2', 'lif', 'group', connect_f='MS'); chk('c6 refused: band with the membrane shortcut', False)
    except NotImplementedError:
        chk('c6 refused: band with the membrane shortcut', True)
    print(f'\n{len(OK)} passed, {len(BAD)} failed')
    if BAD:
        raise SystemExit(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot'); ap.add_argument('--code', default=HERE); ap.add_argument('--compare', nargs=2)
    a = ap.parse_args()
    if a.snapshot:
        snapshot(a.snapshot, os.path.abspath(a.code))
    elif a.compare:
        compare(*a.compare)
        print(f'\n{len(OK)} passed, {len(BAD)} failed')
        if BAD:
            raise SystemExit(1)
    else:
        main()
