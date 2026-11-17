"""Checks of the mixer-variant test (2026-11-17, PROJECT_LOG, exp/f-lif-pop-v3; user request 11-17). Fail closed: any failed item exits 1.
  --compare A B   c0: snapshots of the earlier paths (snap_mx3.py; before / after the edit) are bitwise equal
  (no option)     c1 reset of the stack LIF (traces, forward == charge path, gradients);  c2 repeat pooling (= the stack and fusion run once
                  on the pooled sequence, copied to the repeats; neuron steps);  c3 shortcut inside each direction (= the hand-written
                  stream; same parameters and initial weights as the plain 2-layer stack);  c4 every variant: same initial weights and
                  random state as its control at the same seed, gradients reach every parameter;  c5 refused combinations;  c6 flags,
                  loader and folder suffixes
"""
import argparse
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OK, BAD = [], []


def chk(name, cond, info=''):
    (OK if cond else BAD).append(name)
    print(('pass  ' if cond else 'FAIL  ') + name + (f'  [{info}]' if info else ''), flush=True)


def compare(a, b):
    A, B = torch.load(a), torch.load(b)
    chk('c0 same entries', set(A) == set(B) and len(A) > 0, f'{len(A)} / {len(B)}')
    diff = [k for k in A if k in B and not ((A[k] is None and B[k] is None) or (A[k] is not None and B[k] is not None and torch.equal(A[k], B[k])))]
    chk('c0 earlier paths bitwise equal (weights, random state, outputs, gradients, neuron spikes / charge)', not diff, f'{len(A)} entries, {len(diff)} differ {diff[:3]}')


ETT = dict(seq_len=96, data_patch_size=8, embed_dim=64, mlp_ratios=0.5, mixer_layers=1, connect_f='IAND', mixer='lif', mixer_comb='muxlin', num_classes=5,
           depths=1, T_h=4, bias=False, in_channels=1)
MIT = dict(seq_len=187, data_patch_size=32, embed_dim=384, mlp_ratios=1.0, mixer_layers=2, connect_f='ADD', mixer='lif', mixer_comb='muxlin', num_classes=5,
           depths=1, T_h=4, bias=False, in_channels=1)


def build(base, seed=42, **kw):
    from config import set_random_seed
    from spk_mixer import SpkMixer
    set_random_seed(seed)
    m = SpkMixer(**dict(base, **kw)).cuda()
    return m, torch.get_rng_state().clone()


def spikes_in(m, x):
    """embedding spikes [T_h, B, N, D] of model m for input x (train mode)."""
    return m.encoding(m.tokens(x))


def main():
    from neurons import LIFNeuron
    from layers import spikes_and_charge
    dev = 'cuda'
    # ---- c1 reset ------------------------------------------------------------------------------------------------------
    for mode, after in (('spikede', 0.7), ('subtract', 0.2), ('zero', 0.0)):
        n = LIFNeuron(1, tau=2., threshold=1., max_length=8, **({} if mode == 'spikede' else dict(reset=mode)))
        x = torch.full((8, 1, 1), 1.6)
        u = torch.zeros(1, 1, 1); trace = []
        for t in range(8):
            F, s = n.rhs(u, x[t]); nxt = n._apply_reset(u + F, s); trace.append((float(s), float(nxt))); u = nxt
        s_fwd = n(x).flatten().tolist()
        chk(f'c1 reset {mode}: constant input 1.6, membrane after the first spike', abs(trace[1][1] - after) < 1e-6 and trace[1][0] == 1.,
            f'{trace[1][1]:.3f} (expected {after}), spikes {[int(v) for v in s_fwd]}')
        chk(f'c1 reset {mode}: forward spikes == hand trace', s_fwd == [v[0] for v in trace])
    g = torch.Generator().manual_seed(3)
    for mode in ('spikede', 'subtract', 'zero'):
        n = LIFNeuron(32, tau=2., max_length=24, **({} if mode == 'spikede' else dict(reset=mode)))
        x = (1.2 * torch.randn(24, 5, 32, generator=g)).requires_grad_(True)
        s = n(x); s2, h = spikes_and_charge(n, x)
        chk(f'c1 reset {mode}: charge path spikes == forward, spikes == (H - theta >= 0)', torch.equal(s, s2) and torch.equal(s2, (h >= 0).float()),
            f'rate {float(s.mean()):.3f}')
        (s.sum() + h.sum()).backward()
        chk(f'c1 reset {mode}: input gradient finite and nonzero', bool(torch.isfinite(x.grad).all()) and float(x.grad.abs().sum()) > 0)
    rates = {}
    for mode in ('spikede', 'subtract', 'zero'):
        n = LIFNeuron(64, tau=2., max_length=24, **({} if mode == 'spikede' else dict(reset=mode)))
        rates[mode] = float(n(1.2 * torch.randn(24, 8, 64, generator=torch.Generator().manual_seed(4))).mean())
    chk('c1 firing on the same input: spikede > subtract >= zero', rates['spikede'] > rates['subtract'] >= rates['zero'], str({k: round(v, 4) for k, v in rates.items()}))
    m, _ = build(ETT, mixer_reset='zero')
    modes = {nm: getattr(getattr(mod, 'neuron', None), 'reset_mode', None) for nm, mod in m.named_modules() if type(mod).__name__ == 'NeoNeuron'}
    stack = {k: v for k, v in modes.items() if '.mixer.' in k or '.mixer_rev.' in k}
    other = {k: v for k, v in modes.items() if k not in stack}
    chk('c1 mixer_reset reaches the directional-stack neurons only', len(stack) == 2 and all(v == 'zero' for v in stack.values()) and
        all(v == 'spikede' for v in other.values()), f'stack {sorted(stack)}, others {len(other)}')
    # ---- c2 pooling ----------------------------------------------------------------------------------------------------
    x_in = torch.randn(6, 1, 96, generator=torch.Generator().manual_seed(1)).cuda()
    for pool in ('max', 'mean'):
        m, _ = build(ETT, mix_pool=pool)
        m.train()
        with torch.no_grad():
            e = spikes_in(m, x_in)
            blk = m.blocks[0]
            out = blk.mix(e)
            blk.mix_pool = 'none'
            xp = e.amax(0, keepdim=True) if pool == 'max' else e.mean(0, keepdim=True)
            ref = blk.mix(xp).expand_as(e)
            blk.mix_pool = pool
        same_rep = all(torch.equal(out[0], out[k]) for k in range(1, 4))
        chk(f'c2 {pool}: mixer output == the plain block run once on the pooled input, copied to the 4 repeats', torch.equal(out, ref) and same_rep,
            f'rate {float(out.mean()):.4f}, input values {sorted(set(xp.unique().tolist()))[:6]}')
        steps = []
        hk = [mod.register_forward_hook(lambda md, i, o: steps.append(tuple(i[0].shape))) for nm, mod in m.named_modules()
              if type(mod).__name__ == 'LIFNeuron' and ('.mixer.' in nm or '.mixer_rev.' in nm)]
        with torch.no_grad():
            m(x_in)
        for h in hk: h.remove()
        N = m.num_patches
        chk(f'c2 {pool}: stack neurons step over the {N} tokens of one sequence per window (rows B x 1 x N', all(s[0] == N and s[1] == 6 for s in steps) and len(steps) == 2,
            f'{steps}')
    # ---- c3 shortcut inside each direction ------------------------------------------------------------------------------
    for base, tag in ((dict(ETT, mixer_layers=2), 'ETT 2 layers'), (MIT, 'MIT-BIH')):
        m, _ = build(base, mixer_ms=True)
        p, _ = build(base)
        sd_m, sd_p = m.state_dict(), p.state_dict()
        chk(f'c3 {tag}: same parameters and initial weights as the plain stack', set(sd_m) == set(sd_p) and all(torch.equal(sd_m[k], sd_p[k]) for k in sd_m))
        blk = m.blocks[0]
        with torch.no_grad():
            m.train()
            x = spikes_in(m, torch.randn(4, 1, base['seq_len'], generator=torch.Generator().manual_seed(5)).cuda()).permute(2, 1, 0, 3).contiguous()
            for st in (blk.mixer, blk.mixer_rev):
                s_ms, h_ms = blk.stack(st, x, True)
                L1, L2 = st.Mem[0], st.Mem[1]
                T, B, N, D = x.shape
                c1 = L1.bn(L1.w_in(x).reshape(-1, D)).reshape(T, B, N, D)
                s1 = L1.lif(c1)
                u2 = c1 + L2.bn(L2.w_in(s1).reshape(-1, D)).reshape(T, B, N, D)
                s2, h2 = L2.lif.forward_charge(u2)
                plain, _ = st.forward_charge(x)
            chk(f'c3 {tag}: stack with shortcut == hand-written stream (u2 = BN1 W1 x + BN2 W2 s1)', torch.equal(s_ms, s2) and torch.equal(h_ms, h2),
                f'rate {float(s_ms.mean()):.4f}, plain {float(plain.mean()):.4f}, differ {float((s_ms != plain).float().mean()):.4f}')
    # ---- c4 init / RNG / gradients ---------------------------------------------------------------------------------------
    cases = [('pool max', ETT, dict(mix_pool='max'), {}), ('pool mean', ETT, dict(mix_pool='mean'), {}),
             ('ms ETT', dict(ETT, mixer_layers=2), dict(mixer_ms=True), {}), ('reset subtract', ETT, dict(mixer_reset='subtract'), {}),
             ('reset zero', ETT, dict(mixer_reset='zero'), {}),
             ('MIT pool max', MIT, dict(mix_pool='max'), {}), ('MIT pool mean', MIT, dict(mix_pool='mean'), {}), ('MIT ms', MIT, dict(mixer_ms=True), {}),
             ('MIT reset subtract', MIT, dict(mixer_reset='subtract'), {}), ('MIT reset zero', MIT, dict(mixer_reset='zero'), {})]
    for name, base, kw, _ in cases:
        m, rng_m = build(base, **kw)
        p, rng_p = build(base)
        sd_m, sd_p = m.state_dict(), p.state_dict()
        chk(f'c4 {name}: initial weights and random state == control at seed 42', set(sd_m) == set(sd_p) and all(torch.equal(sd_m[k], sd_p[k]) for k in sd_m) and torch.equal(rng_m, rng_p))
        m.train(); m.train_mode = 'training'
        x = torch.randn(6, 1, base['seq_len'], generator=torch.Generator().manual_seed(1)).cuda()
        z = m(x); (z ** 2).sum().backward()
        none = [n for n, q in m.named_parameters() if q.grad is None or float(q.grad.abs().sum()) == 0.]
        zp = p(x)
        chk(f'c4 {name}: gradients reach every parameter; output differs from the control', not none and not torch.equal(z, zp), f'no grad: {none[:4]}')
    # ---- c5 refused -------------------------------------------------------------------------------------------------------
    bad = [('ms with 1 layer', ETT, dict(mixer_ms=True)), ('reset with pop_lif', dict(ETT, mixer='pop_lif', pop_soma_init=0.5), dict(mixer_reset='zero')),
           ('pool with token layout', ETT, dict(mix_pool='max', taxis='token')), ('pool with connect MS', MIT, dict(mix_pool='max', connect_f='MS')),
           ('pool with band', MIT, dict(mix_pool='max', band='group')), ('pop + pool without gain table', dict(ETT, mixer='pop_lif', pop_soma_init=0.5), dict(mix_pool='max')),
           ('bad pool', ETT, dict(mix_pool='avg')), ('bad reset', ETT, dict(mixer_reset='hard'))]
    for name, base, kw in bad:
        try:
            build(base, **kw); ok = False
        except (ValueError, NotImplementedError):
            ok = True
        chk(f'c5 refused: {name}', ok)
    # ---- c6 flags / loader / folder ------------------------------------------------------------------------------------------
    import config
    from load_model import load_spk_mixer, _mixer_variants
    ap_args = ['--model', 'spkMixer', '--mixer', 'lif', '--mixer_comb', 'muxlin', '--connect_f', 'IAND', '--spk_mixpool', 'max', '--mixer_reset', 'zero']
    src = open(os.path.join(HERE, 'config.py')).read()
    chk('c6 flags in config.py', all(f in src for f in ("'--spk_mixpool'", "'--mixer_ms'", "'--mixer_reset'", '_mp{', '"_mms"', '_mr{')))
    a = types.SimpleNamespace(spk_mixpool='mean', mixer_ms=True, mixer_reset='subtract')
    chk('c6 loader passes only non-default options', _mixer_variants(a) == dict(mix_pool='mean', mixer_ms=True, mixer_reset='subtract') and
        _mixer_variants(types.SimpleNamespace()) == {} and _mixer_variants(types.SimpleNamespace(spk_mixpool='none', mixer_ms=False, mixer_reset='spikede')) == {})
    print(f'\n{len(OK)} passed, {len(BAD)} failed')
    sys.exit(1 if BAD else 0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--compare', nargs=2)
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare); sys.exit(1 if BAD else 0)
    main()
