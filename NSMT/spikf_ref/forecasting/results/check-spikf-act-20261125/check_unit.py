"""Unit checks of the SpikF copy (2026-11-25, exp/f-lif-pop-v3; PROJECT_LOG SpikF-LIF pre-registration). No training.
  U1 GELU path = released model: the local H720 checkpoint of each ETT dataset (/home/yschoi/SpikF/exp/run_ETT/SpikF_<ds>_ftM_sl96_pl720_
     lr0.0005_bs32/checkpoint.pth) loaded strictly into the clone's SpikF (/home/yschoi/SpikF/models/SpikF.py, read only) and into the
     copy with activ='gelu'; the outputs on the first 64 test windows are bit-identical (torch.equal)
  U2a the head neuron alone keeps samples apart exactly: its [T, 8, D, hidden] input (hooked) run through a fresh copy of the neuron as
     one batch and slice by slice gives identical spikes (torch.equal; the neuron is elementwise along T, no reduction)
  U2b end to end, activ='lif' (same checkpoint; the LIF has no parameters), eval mode: each of 8 windows run as one batch vs alone differs
     by no more than 1e-6 and by no more than 10x the GELU model's own batch-vs-alone difference (GEMM rounding changes with the batch
     size; the first version of this check asked for exactly 0 and failed at 2.4e-7 - 4.8e-7, check_unit_v1_strict.txt)
  U3 for contrast, the commented line of the clone (LIF on the [T * B, D, hidden] tensor, T * B taken as time) mixes windows: the same
     test gives a difference > 0 (shows why the copy views the tensor as [T, B, D, hidden])
  U4 the LIF head output is binary (only 0 and 1) and fires (rate strictly between 0 and 1) on these windows
usage (from spikf_ref/forecasting/): python -B results/check-spikf-act-20261125/check_unit.py
"""
import importlib.util
import os
import sys

import numpy as np
import torch
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.dirname(os.path.dirname(HERE))
CLONE = '/home/yschoi/SpikF'
sys.dont_write_bytecode = True
sys.path.insert(0, F)
from models.SpikF import SpikF as Copy                                          # noqa: E402
from data_provider.ETT_data_loader import Dataset_ETT_hour, Dataset_ETT_minute    # noqa: E402
spec = importlib.util.spec_from_file_location('clone_spikf', os.path.join(CLONE, 'models', 'SpikF.py'))
cl = importlib.util.module_from_spec(spec); spec.loader.exec_module(cl)
ROOT = os.path.join(os.path.dirname(os.path.dirname(F)), 'forecasting', 'dataset', 'ETT-small')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
fails = 0


def check(label, ok, detail=''):
    global fails
    fails += not ok
    print(f'{"PASS" if ok else "FAIL"} {label}{": " + detail if detail else ""}')


def run(m, x):
    functional.reset_net(m)
    with torch.no_grad():
        return m(x)


for ds in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
    D = (Dataset_ETT_hour if ds.startswith('ETTh') else Dataset_ETT_minute)(root_path=ROOT, data_path=f'{ds}.csv', flag='test',
                                                                            size=[96, 48, 720], features='M', target='OT')
    x = torch.tensor(np.stack([D[i][0] for i in range(64)])).float().to(dev)
    sd = torch.load(os.path.join(CLONE, 'exp', 'run_ETT', f'SpikF_{ds}_ftM_sl96_pl720_lr0.0005_bs32', 'checkpoint.pth'), map_location=dev)
    a = cl.SpikF(96, 48, 32, 16, 1, 7, 720, 2.0, 2.0, 720).to(dev); a.load_state_dict(sd, strict=True); a.eval()
    b = Copy(96, 48, 32, 16, 1, 7, 720, 2.0, 2.0, 720, activ='gelu').to(dev); b.load_state_dict(sd, strict=True); b.eval()
    ya, yb = run(a, x), run(b, x)
    check(f'U1 {ds}: GELU copy = released model on 64 test windows', torch.equal(ya, yb), f'max |diff| {(ya - yb).abs().max().item():.3g}')
    c = Copy(96, 48, 32, 16, 1, 7, 720, 2.0, 2.0, 720, activ='lif').to(dev); c.load_state_dict(sd, strict=True); c.eval()
    yc = run(c, x[:8])
    d = max((run(c, x[i:i + 1]) - yc[:, i:i + 1]).abs().max().item() for i in range(8))
    yb8 = run(b, x[:8])
    dg = max((run(b, x[i:i + 1]) - yb8[:, i:i + 1]).abs().max().item() for i in range(8))
    check(f'U2b {ds}: LIF head, batch of 8 vs each window alone (GELU model: {dg:.3g})', d <= 1e-6 and d <= 10 * max(dg, 1e-7), f'max |diff| {d:.3g}')
    ins = []
    h = c.activ.register_forward_hook(lambda m, i, o: ins.append(i[0].detach()))
    run(c, x[:8]); h.remove()
    z = ins[0]                                                                   # [T, 8, D, hidden]
    n = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch').to(dev)
    functional.reset_net(n); o8 = n(z)
    same = True
    for i in range(8):
        functional.reset_net(n); same &= torch.equal(n(z[:, i:i + 1]), o8[:, i:i + 1])
    check(f'U2a {ds}: head neuron, [T, 8, D, hidden] as one batch = slice by slice (input shape {list(z.shape)})', bool(same))
    spikes = []
    h = c.activ.register_forward_hook(lambda m, i, o: spikes.append(o.detach()))
    run(c, x[:8]); h.remove()
    s = spikes[0]
    vals = torch.unique(s).tolist()
    check(f'U4 {ds}: LIF head output binary and firing', set(vals) <= {0.0, 1.0} and 0 < s.mean().item() < 1, f'values {vals}, rate {s.mean().item():.4f}')
    w = Copy(96, 48, 32, 16, 1, 7, 720, 2.0, 2.0, 720, activ='gelu').to(dev); w.load_state_dict(sd, strict=True); w.eval()
    w.activ = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch').to(dev)          # the clone's commented line, as written
    yw = run(w, x[:8])
    dw = max((run(w, x[i:i + 1]) - yw[:, i:i + 1]).abs().max().item() for i in range(8))
    check(f'U3 {ds}: the commented line (T * B as time) mixes windows', dw > 0.0, f'max |diff| {dw:.3g}')
print('ALL PASS' if fails == 0 else f'{fails} FAIL')
sys.exit(0 if fails == 0 else 1)
