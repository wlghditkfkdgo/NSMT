"""MIT-BIH synaptic energy under the integer rule (2026-11-01, PROJECT_LOG, exp/f-lif-pop-v3; MIT-BIH tuning of the fusion MUX).

test.py counts synaptic operations with syops (a non-0/1 input to a Linear = MAC). With the ADD residual the SpkMixer block output is
0-3, so the following Linear layers would be counted as MACs; the user's rule (10-29 / 11-01) counts a non-negative integer input as
sum(values) ACs. This script reloads every run of a suite (model_state/config.pt + best+model.pt = the last-epoch weights that
test.py evaluated), re-runs the MIT-BIH test file with energy.SynOpRecorder on the whole model and writes
    syn_syops_uJ (recorder, syops rule), syn_int_uJ (recorder, integer rule),
    total_fft_uJ_corr = total_fft_uJ - syn_energy_uJ + syn_int_uJ   for ADD runs;   = total_fft_uJ (logged) otherwise.
Checks (fail closed): the recomputed test macro-F1 equals the logged one (1e-6); for 0/1 runs the recorder's syops rule equals the
logged syops synaptic energy (relative 1e-3) and the two rules agree; for ADD runs the integer rule has fewer MACs.
usage: python energy_recount_mitbih.py --suite <suite>  ->  results/<suite>/energy_recount.csv
"""
import argparse
import csv
import math
import os
import sys
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spikingjelly.clock_driven import functional                          # noqa: E402
from torcheval.metrics import MulticlassF1Score                           # noqa: E402
from load_model import LOAD_MODEL                                         # noqa: E402
from dataloader import create_loader                                      # noqa: E402
from energy import SynOpRecorder                                          # noqa: E402
from summarize_rhythm import parse                                        # noqa: E402


@torch.no_grad()
def measure(run, device):
    cfg = torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')
    args = SimpleNamespace(**cfg); args.device = device
    model = LOAD_MODEL[args.model](args, True)
    model.load_state_dict(torch.load(os.path.join(run, 'model_state', 'best+model.pt'), map_location='cpu'))
    model = model.to(device).eval(); model.train_mode = 'testing'
    loader, _ = create_loader(train=False, num_classes=args.num_classes, batch_size=args.batch_size, data_root=args.root_path + '/5class_mitbih_test_3_187.csv',
                              data_window_size=187, num_workers=args.num_workers)
    rec = SynOpRecorder(model)
    f1 = MulticlassF1Score(num_classes=args.num_classes, average='macro').to(device)
    functional.reset_net(model)
    for batch in loader:
        x = batch[0].float().to(device); y = batch[1].long().squeeze(-1).to(device)
        out = model(x)
        out = out[0] if isinstance(out, tuple) else out
        rec.add_samples(x.shape[0])
        functional.reset_net(model)
        f1.update(out.argmax(1), y)
    rec.detach()
    return float(f1.compute()), rec.summary(), str(getattr(args, 'connect_f', 'IAND')).upper()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--suite', required=True); ap.add_argument('--device', default='cuda'); a = ap.parse_args()
    res = os.path.join(HERE, 'results', a.suite)
    out, bad = [], []
    for r in parse(os.path.join(res, 'raw.txt')):
        f1, s, cf = measure(r['dir'], a.device)
        add = cf == 'ADD'
        corr = r['total_fft_uJ'] - r['syn_energy_uJ'] + s['syn_energy_int_uJ'] if add else r['total_fft_uJ']
        if abs(f1 - r['f1']) > 1e-6:
            bad.append(f"{r['dir']}: macro-F1 {f1:.6f} != logged {r['f1']:.6f}")
        if not add and (abs(s['rec_syn_uJ'] - r['syn_energy_uJ']) > 1e-3 * r['syn_energy_uJ'] or abs(s['rec_syn_uJ'] - s['syn_energy_int_uJ']) > 1e-9):
            bad.append(f"{r['dir']}: recorder {s['rec_syn_uJ']:.6f} / integer {s['syn_energy_int_uJ']:.6f} vs syops {r['syn_energy_uJ']:.6f} uJ")
        if add and not s['int_MAC_M'] < s['rec_MAC_M']:
            bad.append(f"{r['dir']}: ADD run without integer inputs (int MAC {s['int_MAC_M']}, syops-rule MAC {s['rec_MAC_M']})")
        if not (math.isfinite(corr) and corr > 0):
            bad.append(f"{r['dir']}: corrected energy {corr}")
        out.append(dict(cond=r['cond'], tag=r.get('tag', ''), seed=int(r['seed']), val_fold=int(r.get('val_fold', -1)), f1_logged=f"{r['f1']:.6f}",
                        f1_recomputed=f'{f1:.6f}', total_fft_uJ=f"{r['total_fft_uJ']:.6f}", syn_energy_uJ=f"{r['syn_energy_uJ']:.6f}",
                        syn_syops_uJ=f"{s['rec_syn_uJ']:.6f}", syn_int_uJ=f"{s['syn_energy_int_uJ']:.6f}", rule='integer' if add else 'binary',
                        total_fft_uJ_corr=f'{corr:.6f}'))
        print(f"{r.get('tag', r['cond'])} seed {int(r['seed'])}: f1 {f1:.6f} (logged {r['f1']:.6f}) syn syops-rule {s['rec_syn_uJ']:.3f} / integer "
              f"{s['syn_energy_int_uJ']:.3f} (logged {r['syn_energy_uJ']:.3f}) -> total {corr:.3f} uJ", flush=True)
    path = os.path.join(res, 'energy_recount.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f'{len(out)} runs -> {path}')
    if bad:
        print('FAILED CHECKS:\n  ' + '\n  '.join(bad)); sys.exit(1)
    print('all checks passed')


if __name__ == '__main__':
    main()
