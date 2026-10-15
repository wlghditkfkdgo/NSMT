#!/usr/bin/env python3
"""The original spikeDE classes against forecasting/neurons.py and the current v3 neuron.

Run with a torch >= 2 interpreter (spikeDE decorates a method with torch.compile at import),
on the CPU, in float64:
    python analysis/spikede_parity.py <spikeDE root at fcd743b> [out.json]

P1  LIFNeuron        == spikeDE LIFNeuron + euler_integrate_tuple
P2  FLIFNeuron pred  == spikeDE LIFNeuron + pred_integrate_tuple   (memory None and 8)
P3  FLIFNeuron gl    == spikeDE LIFNeuron + gl_integrate_tuple     (memory None and 8)
P4  PopulationFLIF(soma='none', K=3) == three independent spikeDE f-LIF neurons
P5  gradient of a weighted spike sum w.r.t. the input current, ours vs spikeDE (pred). spikeDE's
    surrogate returns float32 spikes even in float64 runs, so the original agrees to float32
    precision only; a second reference with that one cast removed (patched forward returning
    the input's dtype) must agree to float64 precision
States are compared by the relative error |ours - spikeDE| / max(1, max|spikeDE|): with tau=0.5
and h=1 the explicit schemes are unstable and the states reach 1e10 and more.
P6  the current v3 neuron (layers.PopulationNeuron, K=1, mode 'full'):
      its branch state == spikeDE pred with firing switched off (threshold 1e30);
      its spikes vs the spikeDE f-LIF with reset -- expected to differ (design, not a bug)
Inputs: constant 1.5, a pulse of 8 at t=0, N(1, 2) noise; 40 steps; alpha {0.3, 0.5, 0.7, 1};
tau {0.5, 2, 8} (the v3 neuron needs tau > 1: {2, 8}); h {1, 0.1}; theta 1, kappa 5.
"""
import sys
import json
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import torch

ROOT = Path(sys.argv[1]).resolve()


def spikede_commit(root):
    """Identify the spikeDE version under `root` without trusting a parent repository (audit 78).

    `git -C root rev-parse HEAD` walks up to the enclosing repository when `root` has no .git of its own and then
    reports THAT repository's HEAD. Use git only when `root` is itself the top level of a clone; otherwise read the
    pinned commit from a source_manifest.json written next to a git-less archive; otherwise say 'unknown'.
    """
    top = subprocess.run(['git', '-C', str(root), 'rev-parse', '--show-toplevel'], capture_output=True, text=True)
    if top.returncode == 0 and Path(top.stdout.strip()).resolve() == root:
        return subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    manifest = root / 'source_manifest.json'
    if manifest.exists():
        import re as _re
        commits = {m for e in json.loads(manifest.read_text()) for m in _re.findall(r'/([0-9a-f]{40})/', e.get('url', ''))}
        if len(commits) == 1:
            return commits.pop() + ' (from source_manifest.json)'
    return 'unknown (no own .git and no single-commit source_manifest.json)'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_suffix('.json')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting'))
from unittest import mock                                        # noqa: E402
from spikeDE.neuron import LIFNeuron as RefLIF                   # noqa: E402
from spikeDE.surrogate import ArctanSurrogate                    # noqa: E402
from spikeDE.solver import euler_integrate_tuple, pred_integrate_tuple, gl_integrate_tuple   # noqa: E402
import neurons as N                                              # noqa: E402
import layers                                                    # noqa: E402

torch.set_default_dtype(torch.float64)
T = 40
rng = np.random.default_rng(7)
INPUTS = {'constant': np.full(T, 1.5), 'pulse': np.r_[8., np.zeros(T - 1)], 'noise': rng.normal(1., 2., T)}
SOLVERS = {'euler': None, 'pred': pred_integrate_tuple, 'gl': gl_integrate_tuple}


def reference(current, solver, tau, h, alpha=1., memory=None, threshold=1., grad=False):
    """spikeDE: run the original neuron under the original solver; returns states v_1..v_T,
    spikes s_0..s_{T-1} and the input tensor (for gradients)."""
    neuron = RefLIF(tau=tau, threshold=threshold, surrogate_grad_scale=5.).double()
    drive = torch.tensor(current, dtype=torch.float64, requires_grad=grad)
    # The pred solver keeps no state history (it stores F), so the state is read inside the
    # right-hand side; one extra step (input 0) exposes v_T and cannot change earlier steps.
    t_grid = torch.arange(T + 2, dtype=torch.float64) * h
    seen = []

    def rhs(t, values):
        n = int(round(float(t) / h))
        seen.append(values[0])
        dv, s = neuron(values[0], drive[n:n + 1] if n < T else torch.zeros(1))
        return dv, s

    y0 = (torch.zeros(1), torch.zeros(1))
    if solver == 'euler':
        hist = euler_integrate_tuple(rhs, y0, t_grid, 1)
    else:
        hist = SOLVERS[solver](rhs, y0, [alpha], t_grid, memory=memory)
    states = torch.cat(seen[1:])                                     # v_1 .. v_T
    spikes = torch.cat(hist[-1][:-1]).double()                       # s_0 .. s_{T-1}

    return states, spikes, drive, float(t_grid[-1] - t_grid[-2])


def ours(cls_name, current, tau, h, alpha=1., method='pred', memory=None, threshold=1., grad=False, **kw):
    neuron = N.build_neuron(cls_name, 1, tau=tau, alpha=alpha, method=method, memory=memory, step=h,
                            threshold=threshold, max_length=T, dtype=torch.float64, **kw)
    drive = torch.tensor(current, dtype=torch.float64, requires_grad=grad)
    out, aux = neuron(drive.reshape(T, 1, 1), return_aux=True)

    return aux['state'][:, 0, 0, :], out[:, 0, :], drive


def compare(name, a_state, a_spike, b_state, b_spike):
    err = float((a_state - b_state).abs().max())
    scale = max(1., float(a_state.abs().max()))
    rel = err / scale
    finite = bool(torch.isfinite(a_state).all() and torch.isfinite(b_state).all())
    return {'case': name, 'state_max_abs_err': err, 'state_max_abs': scale, 'state_rel_err': rel,
            'spikes_equal': bool(torch.equal(a_spike, b_spike)), 'spikes': int(a_spike.sum()),
            'ok': finite and rel < 1e-12 and bool(torch.equal(a_spike, b_spike))}


def dtype_preserving(ctx, input, scale):
    """spikeDE ArctanSurrogate.forward without its .float() cast."""
    ctx.save_for_backward(input)
    ctx.scale = scale
    return (input >= 0).to(input.dtype)


def main():
    rows = []
    # ---- P1-P3 ---------------------------------------------------------------------------
    for solver in ('euler', 'pred', 'gl'):
        for alpha in ((1.,) if solver == 'euler' else (.3, .5, .7, 1.)):
            for tau in (.5, 2., 8.):
                for h in (1., .1):
                    for memory in ((None,) if solver == 'euler' else (None, 8)):
                        for inp, current in INPUTS.items():
                            r_state, r_spike, _, h_used = reference(current, solver, tau, h, alpha, memory)
                            cls = 'lif' if solver == 'euler' else 'flif'
                            o_state, o_spike, _ = ours(cls, current, tau, h_used, alpha,
                                                       'gl' if solver == 'gl' else 'pred', memory)
                            rows.append({'test': {'euler': 'P1', 'pred': 'P2', 'gl': 'P3'}[solver],
                                         'solver': solver, 'alpha': alpha, 'tau': tau, 'h': h, 'memory': memory,
                                         'input': inp, **compare(inp, r_state, r_spike, o_state[:, 0], o_spike[:, 0])})
    # ---- P4 ---------------------------------------------------------------------------------
    taus = (2., 4., 8.)
    for inp, current in INPUTS.items():
        o_state, o_spike, _ = ours('pop_flif', current, taus, 1., .7, soma='none')
        for k, tau in enumerate(taus):
            r_state, r_spike, _, _ = reference(current, 'pred', tau, 1., .7)
            rows.append({'test': 'P4', 'solver': 'pred', 'alpha': .7, 'tau': tau, 'h': 1., 'memory': None,
                         'input': inp, **compare(inp, r_state, r_spike, o_state[:, k], o_spike[:, k])})
    # ---- P5 ---------------------------------------------------------------------------------
    w = torch.linspace(.5, 1.5, T)
    for inp, current in INPUTS.items():
        _, o_spike, o_drive = ours('flif', current, 2., 1., .7, grad=True)
        (o_spike[:, 0] * w).sum().backward()
        _, r_spike, r_drive, _ = reference(current, 'pred', 2., 1., .7, grad=True)
        (r_spike * w).sum().backward()
        with mock.patch.object(ArctanSurrogate, 'forward', staticmethod(dtype_preserving)):
            _, p_spike, p_drive, _ = reference(current, 'pred', 2., 1., .7, grad=True)
            (p_spike * w).sum().backward()
        norm = float(o_drive.grad.norm())
        err_orig = float((r_drive.grad - o_drive.grad).abs().max())
        err_f64 = float((p_drive.grad - o_drive.grad).abs().max())
        rows.append({'test': 'P5', 'solver': 'pred', 'alpha': .7, 'tau': 2., 'h': 1., 'memory': None, 'input': inp,
                     'case': inp, 'grad_norm': norm, 'grad_max_abs_err': err_f64,
                     'grad_max_abs_err_vs_original_float32_spikes': err_orig,
                     'grad_rel_err_vs_original': err_orig / max(norm, 1e-12),
                     'ok': err_f64 < 1e-12 * max(1., norm) and err_orig < 1e-6 * max(1., norm)})
    # ---- P6 ---------------------------------------------------------------------------------
    for alpha in (.5, .7, 1.):
        for tau in (2., 8.):
            for inp, current in INPUTS.items():
                legacy = layers.PopulationNeuron(1, num_population=1, alpha=alpha, tau=(tau,), max_length=T,
                                                 dtype=torch.float64).double()
                out, aux = legacy(torch.tensor(current).reshape(T, 1, 1), mode='full', return_aux=True)
                branch = aux['state'][:, 0, 0, 0]
                silent, _, _, _ = reference(current, 'pred', tau, 1., alpha, threshold=1e30)
                _, fired, _, _ = reference(current, 'pred', tau, 1., alpha)
                err = float((branch - silent).abs().max())
                rows.append({'test': 'P6', 'solver': 'pred', 'alpha': alpha, 'tau': tau, 'h': 1., 'memory': None,
                             'input': inp, 'case': inp, 'branch_vs_silent_spikeDE_err': err, 'ok': err < 1e-10,
                             'v3_spikes': int(out.sum()), 'spikeDE_fLIF_spikes': int(fired.sum()),
                             'spike_trains_equal': bool(torch.equal(out[:, 0, 0], fired))})

    summary = {}
    for row in rows:
        s = summary.setdefault(row['test'], {'cases': 0, 'ok': 0, 'worst': 0.})
        s['cases'] += 1
        s['ok'] += int(row['ok'])
        s['worst'] = max(s['worst'], row.get('state_rel_err', row.get('grad_max_abs_err',
                                                                       row.get('branch_vs_silent_spikeDE_err', 0.))))
    p6 = [r for r in rows if r['test'] == 'P6']
    summary['P6']['spike_trains_equal'] = sum(r['spike_trains_equal'] for r in p6)
    head = spikede_commit(ROOT)
    files = {f: hashlib.sha256((ROOT / 'spikeDE' / f).read_bytes()).hexdigest() for f in ('neuron.py', 'solver.py', 'surrogate.py')}
    task = Path(__file__).resolve().parents[1] / 'forecasting'      # 비교한 우리 쪽 코드
    ours_files = {f: hashlib.sha256((task / f).read_bytes()).hexdigest() for f in ('neurons.py', 'layers.py')}
    record = {'spikeDE_commit': head, 'spikeDE_sha256': files, 'ours_sha256': ours_files, 'torch': torch.__version__,
              'summary': summary, 'rows': rows}
    OUT.write_text(json.dumps(record, indent=1))
    print(f"[parity] spikeDE {head}  torch {torch.__version__}")
    for test, s in summary.items():
        extra = f", v3 spike trains equal to spikeDE f-LIF: {s['spike_trains_equal']}/{s['cases']}" if test == 'P6' else ''
        kind = {'P5': 'gradient |err| vs dtype-preserving spikeDE', 'P6': 'branch |err|'}.get(test, 'relative state err')
        print(f"[parity] {test}: {s['ok']}/{s['cases']} pass, worst {kind} {s['worst']:.2e}{extra}")
    p2 = [r for r in rows if r['test'] in ('P2', 'P3')]
    print(f"[parity] P2/P3 largest state magnitude {max(r['state_max_abs'] for r in p2):.3e} "
          f"(tau 0.5, h 1: explicit scheme unstable); largest absolute err {max(r['state_max_abs_err'] for r in p2):.3e}")
    for r in rows:
        if r['test'] == 'P5':
            print(f"[parity] P5 {r['input']}: vs original spikeDE (float32 spikes) rel err "
                  f"{r['grad_rel_err_vs_original']:.1e}; vs dtype-preserving spikeDE abs err {r['grad_max_abs_err']:.1e}")
    for r in p6:
        if r['input'] == 'constant' and r['alpha'] == .7:
            print(f"[parity] P6 example tau {r['tau']:g} alpha .7 constant 1.5: v3 spikes {r['v3_spikes']}, "
                  f"spikeDE f-LIF spikes {r['spikeDE_fLIF_spikes']}")


if __name__ == '__main__':
    main()
