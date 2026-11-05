"""Internal state of the forward mixer neuron by token step (2026-11-05, PROJECT_LOG, exp/f-lif-pop-v3; user: explain the neuron warm-up in
more detail). No training, report only. It shows why a spike rate that needs many tokens to rise does so: what each member holds at each step.

Checkpoints: CinC fold 0, best-validation weights, val split, seeds 42 / 7 / 2026 (bottleneck_probe.cinc_runs): muxlin LIF, sdlin population LIF,
sdlin population f-LIF. The mixer has two NeoLayers per direction here (time_num_layers 2): both layers of the forward direction are read, layer 2
taking layer 1's spikes as input, as in MixerStack.forward.
Input: mixer tokens shuffled by a new permutation per batch (generator seed 0), so content is flat over positions and step = tokens seen.
The loop is the one in layers.spikes_and_charge (same rhs / integrate / emit calls). At every step t it records, pooled over batch, repeat, channel:
    I           input current (BN output)                                     mean, std
    u_k         member state after the step, one per time constant             mean, std
    w_k u_k     what member k adds to the soma drive (population only)         std
    a           soma drive (population only)                                  mean, std
    H           the value compared with theta (soma charge, or the look-ahead of a member that fires)     mean, std
    s           output spike rate, and share of H >= theta (must be equal)
Fail-closed checks: logged validation score reproduced (1e-6); the recomputed spikes are bitwise the forward mixer's spikes on the same input;
s equals the share of H >= theta at every step (population, LIF).
usage: python member_state_probe.py --cond muxlin|pop_lif|pop_flif -nd <gpu>       (one part per call)
       python member_state_probe.py --summarize                                    -> results/member-state-20261105/summary.md
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottleneck_probe as bp                                              # noqa: E402  (spikingjelly before timm, then load_model)
from load_model import LOAD_MODEL                                          # noqa: E402
from mux_probe import sha256                                               # noqa: E402

OUT = os.path.join(HERE, 'results', 'member-state-20261105')
bp.CINC.update({'pop_flif': ('flif-v1-rhythm-spk-cinc', 'spk_pop_flif_sdlin_or')})
NAMES = {'muxlin': 'muxlin LIF', 'pop_lif': 'sdlin population LIF', 'pop_flif': 'sdlin population f-LIF'}
STEPS = (0, 1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 58, 80, 116)


class Stat:
    def __init__(self, shape):
        self.n, self.s, self.q = torch.zeros(shape, dtype=torch.float64), torch.zeros(shape, dtype=torch.float64), torch.zeros(shape, dtype=torch.float64)

    def add(self, t, v, dims):                                               # v values at step t, reduced over dims (the rest = member axis)
        v = v.double()
        self.s[t] += v.sum(dims).cpu()
        self.q[t] += v.square().sum(dims).cpu()
        c = 1
        for d in dims:
            c *= v.shape[d]
        self.n[t] += c

    def result(self):
        if not bool((self.n > 0).all()):
            raise SystemExit('step without data')
        m = self.s / self.n
        return {'mean': m.tolist(), 'std': (self.q / self.n - m.square()).clamp(min=0.).sqrt().tolist()}


def walk(n, x, st):
    """layers.spikes_and_charge's loop on current x [T, B, D], recording the state. -> spikes [T, B, out_features]"""
    T, B, D = x.shape
    n._check_length(T)
    u = x.new_zeros(B, D, n.num_members)
    soma = n.initial_soma(x)
    history = n.new_history()
    out = []
    for t in range(T):
        F, s = n.rhs(u, x[t])
        nxt, _, _ = n.integrate(t, u, F, x[t], history, None)
        if n.has_soma:
            prev = soma
            o, soma, drive = n.emit(nxt, s, soma)
            h = prev + (drive - prev) / n.soma.tau_s
            st['w_u'].add(t, n.soma.weight.unsqueeze(0) * nxt, (0, 1))
            st['a'].add(t, drive, (0, 1))
            theta = n.soma.threshold
        else:
            o, soma, _ = n.emit(nxt, s, soma)
            h = (u + (x[t].unsqueeze(-1) - u) / n.tau).reshape(B, -1)
            theta = n.threshold
        st['I'].add(t, x[t], (0, 1))
        st['u'].add(t, nxt, (0, 1))
        st['H'].add(t, h, (0, 1))
        st['s'].add(t, o, (0, 1))
        st['H_ge'].add(t, (h >= theta).to(x.dtype), (0, 1))
        out.append(o)
        u = nxt
    return torch.stack(out)


def run_part(cond, dev):
    os.makedirs(os.path.join(OUT, 'parts'), exist_ok=True)
    path = os.path.join(OUT, 'parts', f'cinc_{cond}.json')
    if os.path.exists(path):
        raise SystemExit(f'{path} exists (not overwritten)')
    rec = {'script_sha256': sha256(__file__), 'cond': cond, 'device': dev, 'runs': []}
    for r in bp.cinc_runs(cond):
        t0 = time.time()
        d = r['dir']
        cfg = torch.load(os.path.join(d, 'model_state', 'config.pt'), map_location='cpu')
        args = argparse.Namespace(**cfg)
        args.device = dev
        model = LOAD_MODEL[args.model](args, True)
        ck = os.path.join(d, 'model_state', 'checkpoint.pth')
        model.load_state_dict(torch.load(ck, map_location='cpu'))
        model = model.to(dev).eval()
        model.train_mode = 'testing'
        if model.__class__.__name__ != 'SpkMixer' or len(model.blocks) != 1:
            raise SystemExit(f'{d}: not a one-block SpkMixer')
        blk, N = model.blocks[0], model.num_patches
        layers = list(blk.mixer.Mem)
        stats = []
        for layer in layers:
            K = layer.lif.neuron.num_members
            stats.append({'I': Stat(N), 'u': Stat((N, K)), 'w_u': Stat((N, K)), 'a': Stat(N), 'H': Stat(N), 's': Stat(N), 'H_ge': Stat(N)})
        gen = torch.Generator().manual_seed(0)
        orig_mix = blk.mix
        state = {'batches': 0, 'bitwise': True}

        def probe_mix(x):                                                   # x [T_h, B, N, D]
            p = torch.randperm(N, generator=gen).to(x.device)
            xm = x[:, :, p].permute(2, 1, 0, 3).contiguous()                 # [N, B, T_h, D]
            n_, b_, k_, d_ = xm.shape
            y = xm
            for layer, st in zip(layers, stats):
                I = layer.bn(layer.w_in(y).reshape(-1, d_)).reshape(n_, b_, k_, d_)
                y = walk(layer.lif.neuron, I.reshape(n_, b_ * k_, d_), st).reshape(n_, b_, k_, -1)
            state['bitwise'] = state['bitwise'] and bool(torch.equal(y, blk.mixer(xm)))
            state['batches'] += 1
            return orig_mix(x)

        blk.mix = probe_mix
        met = bp.eval_cinc(model, args, r['seed'], dev, ('val',))
        blk.mix = orig_mix
        got, want = met['val']['score'], float(cfg['best_val_score'])
        if abs(got - want) > 1e-6:
            raise SystemExit(f'{d}: val score {got:.6f} != logged {want:.6f}')
        if not state['bitwise']:
            raise SystemExit(f'{d}: recomputed spikes differ from the forward mixer')
        run = {'dir': d, 'seed': int(r['seed']), 'N': N, 'checkpoint_sha256': sha256(ck), 'score': got, 'batches': state['batches'], 'layers': []}
        for layer, st in zip(layers, stats):
            neuron = layer.lif.neuron
            one = {'taus': neuron.tau.tolist(), 'has_soma': bool(neuron.has_soma)}
            for k, v in st.items():
                if (k in ('w_u', 'a')) and not neuron.has_soma:
                    continue
                one[k] = v.result()
            gap = max(abs(a - b) for a, b in zip(one['s']['mean'], one['H_ge']['mean']))
            if gap > 1e-9:
                raise SystemExit(f'{d}: spike rate differs from the share of H >= theta by {gap:.2e}')
            if neuron.has_soma:
                w = neuron.soma.weight.detach().double().cpu()
                one['w_mean'], one['w_std'] = w.mean(0).tolist(), w.std(0).tolist()
            one['bn_gain_mean'] = float(layer.bn.weight.detach().double().abs().mean())
            run['layers'].append(one)
        run['seconds'] = time.time() - t0
        rec['runs'].append(run)
        print(f"[member] {cond} seed {r['seed']}: score {got:.6f} (logged {want:.6f}), batches {state['batches']} | {run['seconds']:.0f} s", flush=True)
        del model
    json.dump(rec, open(path, 'x'), indent=1)
    print(f'wrote {path}')


def half(v, N):
    lo, hi = N // 4, N - N // 4
    mid = float(np.mean(v[lo:hi]))
    return next((t for t in range(N) if v[t] >= .5 * mid), None), mid


def summarize():
    L = ['# Forward mixer neuron state by token step (member_state_probe.py)', '', 'CinC fold 0, val split, shuffled tokens, mean of 3 seeds. No training. Report only. '
         'Checks passed: logged score reproduced (1e-6), recomputed spikes bitwise the forward mixer, spike rate = share of H >= theta.', '']
    L += ['Leak-only reference: a member holds a fraction 1 - (1 - 1/tau)^(t + 1) of a constant input after step t, half of it after about '
          + ', '.join(f'{np.log(.5) / np.log(1 - 1 / t):.1f} tokens (tau {t})' for t in (2, 4, 8, 16, 32)) + '.', '']
    for p in sorted(glob.glob(os.path.join(OUT, 'parts', '*.json')), key=lambda s: ('muxlin', 'pop_lif', 'pop_flif').index(os.path.basename(s)[5:-5])):
        rec = json.load(open(p))
        R = rec['runs']
        if len(R) != 3:
            raise SystemExit(f'{p}: {len(R)} seeds')
        N = R[0]['N']
        for li in range(len(R[0]['layers'])):
            Rl = [r['layers'][li] for r in R]
            taus = Rl[0]['taus']
            avg = lambda key, stat: np.mean([r[key][stat] for r in Rl], axis=0)
            gains = ', '.join('%.3f' % r['bn_gain_mean'] for r in Rl)
            L += [f"## {NAMES[rec['cond']]}, forward layer {li + 1}", '', f'BN gain (mean of |weight|): {gains}.']
            if Rl[0]['has_soma']:
                L.append('Soma weight per member, mean over channels (std over channels), seeds averaged: '
                         + ', '.join(f"tau {t:g}: {np.mean([r['w_mean'][i] for r in Rl]):.3f} ({np.mean([r['w_std'][i] for r in Rl]):.3f})" for i, t in enumerate(taus)) + '.')
            L += ['', '| value | ' + ' | '.join(str(s) for s in STEPS) + ' | half at step (seeds) |', '|---|' + '--:|' * (len(STEPS) + 1)]

            def row(name, v, sc=1., pr=3, half_of=None):
                cells = ' | '.join(f'{v[s] * sc:.{pr}f}' for s in STEPS)
                hs = ', '.join(str(half(np.asarray(h), N)[0]) for h in half_of) if half_of else ''
                L.append(f'| {name} | {cells} | {hs} |')
            row('I std', avg('I', 'std'), half_of=[r['I']['std'] for r in Rl])
            row('I mean', avg('I', 'mean'))
            for i, t in enumerate(taus):
                row(f'u std, tau {t:g}', avg('u', 'std')[:, i], half_of=[np.asarray(r['u']['std'])[:, i] for r in Rl])
            if Rl[0]['has_soma']:
                for i, t in enumerate(taus):
                    row(f'w u std, tau {t:g}', avg('w_u', 'std')[:, i])
                row('drive std', avg('a', 'std'), half_of=[r['a']['std'] for r in Rl])
                row('drive mean', avg('a', 'mean'))
            row('H std', avg('H', 'std'), half_of=[r['H']['std'] for r in Rl])
            row('H mean', avg('H', 'mean'))
            row('spike rate %', avg('s', 'mean'), 100., 2, half_of=[r['s']['mean'] for r in Rl])
            L.append('')
    path = os.path.join(OUT, 'summary.md')
    open(path, 'w').write('\n'.join(L))
    print(f'wrote {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', choices=list(NAMES))
    ap.add_argument('-nd', '--num_device', type=int, default=3)
    ap.add_argument('--summarize', action='store_true')
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    if a.cond is None:
        raise SystemExit('--cond is required')
    with torch.no_grad():
        run_part(a.cond, f'cuda:{a.num_device}')


if __name__ == '__main__':
    main()
