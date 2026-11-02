"""2026-11-02 (exp/f-lif-pop-v3; PROJECT_LOG AD pre-registration): reconstruction anomaly detection with the SpkMixer /
Spikformer-style models of this folder (ad_wrap.Reconstructor), TSLib protocol (neorecall_ad_v1/anomaly_detection of this
repository): the TSLib AD loaders of ./data_provider (byte-identical to neorecall_ad_v1's), window 100, step 1, standardised
with the training file; MSE reconstruction loss; validation = the last 20 % of the training file.

Two modes (user decisions 11-02: tune in the order of the other tasks, select on a validation split, retrain on the whole file):
    --holdout   tuning run: train on the windows inside the first 80 % of the training file, validate on the last 20 % (the TSLib
                validation windows; no shared time step); no test pass (test values cannot be looked at during tuning).
    default     final run (TSLib): train on the whole training file (its last 20 % is also the validation split, as in TSLib),
                test with the best-validation weights: scores = channel mean of the squared error per step over the
                concatenated test windows; AUC-PR (main), AUC-ROC, raw F1 and point-adjusted F1 at the TSLib threshold
                (ad_metrics.py); energy as the other tasks: syops synaptic + neuron energy (energy.FiringRateRecorder), and the
                head recount of energy_recount_ett.py done in the same test pass (the head reads the T_h-averaged spikes; recount =
                T_h binary steps as AC; the ADD residual's body under the integer rule). What syops counted for the head is
                replayed per batch with syops' own rule (an averaged input with <= 2 distinct values in [0, 1] is taken for
                spikes, found in the SMAP smoke run) and checked against syops' per-layer head line before it is replaced.
Training follows train_forecast.py: AdamW, ReduceLROnPlateau on the validation loss, early stopping (--patience).
Outputs: log/best_log_0.csv, train_0 / val_0 TensorBoard events, logargs.txt, final+result.csv, model_state/{config.pt,
best+model.pt (= best-validation weights)}.
usage: python train_ad.py --data MSL --root_path <dataset/MSL> --seq_len 100 --anomaly_ratio 1 [--holdout] --model spkMixer ...
"""
import argparse
import csv
import math
import os
import re
import sys
import time
import types

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from spikingjelly.clock_driven import functional
from syops import get_model_complexity_info
from syops.ops import LIF_syops_counter_hook

from config import Config, parse_arguments, set_random_seed
from load_model import LOAD_MODEL
from utils import EpochLog, get_scheduler, get_energy_consumption
from energy import FiringRateRecorder, SynOpRecorder
from neurons import LIFNeuron, FLIFNeuron
from ad_wrap import Reconstructor
from ad_metrics import ad_metrics
from data_provider.data_factory import data_provider

E_AC, E_MAC = 0.9e-6, 4.6e-6                                          # µJ per operation (utils.get_energy_consumption)
UNIT = {'': 1., 'K': 1e3, 'M': 1e6, 'G': 1e9}


def head_line(out_dir):
    """(AC ops, MAC ops) of the top-level head from the syops per-layer file (as energy_recount_ett.head_line)."""
    for line in open(os.path.join(out_dir, 'model+info+per+layer.txt'), encoding='utf-8'):
        if line.startswith('  (head): Linear('):
            m = re.search(r'%\s*Params,\s*([\d.]+)\s*([KMG]?)\s*Ops,\s*[\d.]+%\s*ACs,\s*([\d.]+)\s*([KMG]?)\s*Ops', line)
            if m:
                return float(m.group(1)) * UNIT[m.group(2)], float(m.group(3)) * UNIT[m.group(4)]
    raise RuntimeError(f'no head line in {out_dir}')


def split_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--anomaly_ratio', type=float, required=True)
    ap.add_argument('--holdout', action='store_true')
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--num_workers', type=int, default=2)
    fc, rest = ap.parse_known_args()
    sys.argv = sys.argv[:1] + rest
    return fc


def data_args(args, fc):
    return types.SimpleNamespace(data=args.data, task_name='anomaly_detection', root_path=args.root_path, seq_len=args.seq_len,
                                 batch_size=args.batch_size, num_workers=fc.num_workers, embed='timeF', freq='h')


class Holdout(Dataset):
    """The training windows that lie inside the first `cut` steps (cut = int(0.8 len), the TSLib validation start)."""
    def __init__(self, ds):
        self.ds, self.cut = ds, int(len(ds.train) * 0.8)
        self.n = (self.cut - ds.win_size) // ds.step + 1

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if not 0 <= i < self.n:
            raise IndexError(i)
        return self.ds[i]


def train_epoch(model, dl, opt, device):
    model.train(); model.train_mode = 'training'
    crit = nn.MSELoss(); tot = 0.; n = 0
    for x, _ in dl:
        x = x.float().to(device)
        loss = crit(model(x), x)
        opt.zero_grad(); loss.backward(); opt.step()
        functional.reset_net(model)
        tot += loss.item(); n += 1
    return {'loss': tot / n}


@torch.no_grad()
def evaluate(model, dl, device):
    """validation MSE over all elements."""
    model.eval(); model.train_mode = 'testing'
    se = 0.; cnt = 0
    functional.reset_net(model)
    for x, _ in dl:
        x = x.float().to(device)
        d = (model(x) - x).double()
        functional.reset_net(model)
        se += float((d ** 2).sum()); cnt += d.numel()
    return {'loss': se / cnt}


@torch.no_grad()
def scores(model, dl, device, probe=None):
    """per-step scores (channel mean of the squared error) and labels over the concatenated windows; probe collects the
    per-step head input statistics of the test pass (energy recount)."""
    model.eval(); model.train_mode = 'testing'
    out_s, out_y = [], []
    functional.reset_net(model)
    for x, y in dl:
        x = x.float().to(device)
        xn, means, stdev = model.norm(x)
        xh = model.step_features(xn.permute(0, 2, 1))                    # [T_h, B, N, D], per-step head input
        f = xh.mean(0)
        rec = model.decode(f) * stdev + means
        functional.reset_net(model)
        if probe is not None:
            for r in probe['recorders']:                               # once per forward, as train_forecast.evaluate
                r.add_samples(x.shape[0])
            probe['n'] += x.shape[0]
            cnt = f.numel() * model.head.out_features                  # the head as syops counts it (syops.ops.spike_rate): an
            if f.unique().numel() <= 2 and f.max() <= 1 and f.min() >= 0:   # averaged input with <= 2 distinct values in [0, 1]
                probe['h_ac'] += cnt * float(f.sum() / f.numel())          # is taken for spikes (AC x rate), otherwise MAC
            else:
                probe['h_mac'] += cnt
            probe['T_h'] = xh.shape[0]
            probe['binary'] &= bool(((xh == 0) | (xh == 1)).all())
            probe['integer'] &= bool((xh >= 0).all() and torch.equal(xh, torch.round(xh)))
            probe['sum'] += float(xh.double().sum()); probe['numel'] += xh.numel()
        out_s.append(((x - rec) ** 2).mean(-1).reshape(-1).cpu().numpy())
        out_y.append(y.reshape(-1).numpy())
    return np.concatenate(out_s), np.concatenate(out_y)


def final_test(args, fc, model, train_dl, test_dl, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    m = model.inner
    fr = FiringRateRecorder(m); syn = SynOpRecorder(m)                 # body only: the head is recounted below
    probe = dict(recorders=(fr, syn), n=0, h_ac=0., h_mac=0., T_h=None, binary=True, integer=True, sum=0., numel=0)
    test_s, test_y = scores(model, test_dl, args.device, probe)
    fr.detach(); syn.detach()
    train_s, _ = scores(model, train_dl, args.device)
    met = ad_metrics(train_s, test_s, test_y, fc.anomaly_ratio)
    model.train_mode = 'testing'
    functional.reset_net(model)
    x0 = next(iter(test_dl))[0]
    with open(os.path.join(out_dir, 'model+info+per+layer.txt'), 'w', encoding='utf-8') as fo:
        ops, params, rate = get_model_complexity_info(model=model, input_res=(tuple(x0.shape[1:]),), dataloader=test_dl, as_strings=False,
                                                      print_per_layer_stat=True, verbose=False, ost=fo,
                                                      custom_modules_hooks={LIFNeuron: LIF_syops_counter_hook, FLIFNeuron: LIF_syops_counter_hook})
    syn_uJ = get_energy_consumption(O_ac=ops[1], O_mac=ops[2], unit='u')
    en = fr.summary(syn_energy_J=get_energy_consumption(O_ac=ops[1], O_mac=ops[2]) * 1e-12)
    fr.write_layers(os.path.join(out_dir, 'neuron_energy.csv'))
    total = float(en['values'][en['columns'].index('total_fft_uJ')])
    s = syn.summary()
    N, D, T_h = model.num_patches, args.embed_dim, probe['T_h']
    out_dim = model.stride * model.C
    head_rate = probe['sum'] / probe['numel']
    head_mac = N * D * out_dim                                         # per window, every multiply-accumulate of the head
    head_ac = T_h * head_rate * head_mac + (T_h - 1) * N * out_dim       # T_h binary steps + summing the T_h outputs
    h_ac, h_mac = probe['h_ac'] / probe['n'], probe['h_mac'] / probe['n']   # what syops counted for the head, per window
    head_syops_uJ = h_ac * E_AC + h_mac * E_MAC
    p_ac, p_mac = head_line(out_dir)                                    # syops' own per-layer line (3 significant digits)
    line_ok = abs(p_ac - h_ac) <= max(1., 0.006 * h_ac) and abs(p_mac - h_mac) <= max(1., 0.006 * h_mac)
    add = str(getattr(args, 'connect_f', '')).upper() == 'ADD' and model.kind == 'spkmix'
    if add:                                                            # ADD residual: body under the integer rule
        corr = total - syn_uJ + s['syn_energy_int_uJ'] + head_ac * E_AC
        ok = probe['integer']
    else:
        corr = total - head_syops_uJ + head_ac * E_AC
        body_ref = syn_uJ - head_syops_uJ                              # syops body = synaptic energy minus syops' head
        ok = probe['binary'] and (model.kind != 'spkmix' or abs(s['rec_syn_uJ'] - body_ref) <= 1e-3 * max(1e-9, syn_uJ))   # as ETT
    ok = ok and line_ok and math.isfinite(corr) and corr > 0
    head = ['weights', 'val_mse_best', 'best_epoch', 'epochs_run'] + list(met) + \
           ['total_op_M', 'AC_op_M', 'MAC_op_M', 'params_M', 'syops_firing_rate', 'syn_energy_uJ'] + en['columns'] + \
           ['head_rate', 'head_MAC', 'head_AC_corr', 'head_syops_AC', 'head_syops_MAC', 'body_syn_uJ', 'body_syn_int_uJ', 'rule',
            'total_fft_uJ_corr', 'energy_check']
    row = ['best', f'{args.best_val_loss:.6f}', str(args.best_epoch), str(args.epochs_run)] + \
          [f'{v:.6f}' if isinstance(v, float) else str(v) for v in met.values()] + \
          [f'{ops[0] / 1e6:.4f}', f'{ops[1] / 1e6:.4f}', f'{ops[2] / 1e6:.4f}', f'{params / 1e6:.6f}', f'{rate:.6f}', f'{syn_uJ:.6f}'] + \
          en['values'] + [f'{head_rate:.6f}', str(head_mac), f'{head_ac:.1f}', f'{h_ac:.3f}', f'{h_mac:.3f}', f"{s['rec_syn_uJ']:.6f}",
                          f"{s['syn_energy_int_uJ']:.6f}", 'integer' if add else 'binary', f'{corr:.6f}', 'ok' if ok else 'FAIL']
    with open(os.path.join(out_dir, 'final+result.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(head); w.writerow(row)
    print(f"[test] aucpr={met['aucpr']:.6f} aucroc={met['aucroc']:.6f} raw_f1={met['raw_f1']:.6f} pa_f1={met['pa_f1']:.6f} "
          f"energy {total:.4f} -> {corr:.4f} uJ (check {'ok' if ok else 'FAIL'})", flush=True)
    return met


def main():
    fc = split_args()
    cfg = parse_arguments()
    args = Config(); args.set_args(cfg)
    for k, v in vars(fc).items():
        setattr(args, k, v)
    set_random_seed(args.seed)
    t0 = time.time()
    da = data_args(args, fc)
    train_set, train_dl = data_provider(da, 'train')
    _, val_dl = data_provider(da, 'val')
    if fc.holdout:
        ho = Holdout(train_set)
        fit_dl = DataLoader(ho, batch_size=args.batch_size, shuffle=True, num_workers=fc.num_workers, drop_last=False)
        print(f'holdout: training windows {len(ho)} inside the first {ho.cut} steps; validation windows {len(val_dl.dataset)}', flush=True)
    else:
        fit_dl = train_dl
    args.num_classes, args.num_channels = 1, int(train_set.train.shape[1])   # channel-mixing tokens; the classification head is dropped
    args.print_info()
    model = Reconstructor(LOAD_MODEL[args.model](args, True), args.embed_dim, args.seq_len, args.num_channels, bias=args.bias).to(args.device)
    functional.reset_net(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'creating model >> number of parameters : {n_params}  tokens {model.num_patches}  channels {model.C}  T_h {model.T}', flush=True)
    args.model_params = n_params
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = get_scheduler(args.scheduler, opt, max_lr=args.lr, min_lr=min(1e-6, args.lr * 1e-2), max_epochs=args.epoch)
    if args.scheduler == 'cosine':
        sched.step(0)
    set_random_seed(args.seed)
    logger = EpochLog(args.save_log_path, kids=0)
    best, best_epoch, wait, epochs_run = float('inf'), -1, 0, 0
    ckpt = os.path.join(args.save_model_state_path, 'best+model.pt')
    for epoch in range(args.epoch):
        tr = train_epoch(model, fit_dl, opt, args.device)
        va = evaluate(model, val_dl, args.device)
        if args.scheduler == 'reduce':
            sched.step(va['loss'])
        elif args.scheduler == 'cosine':
            sched.step(epoch + 1)
        else:
            sched.step()
        logger.write(epoch=epoch, lr=opt.param_groups[0]['lr'], train_result=tr, val_result={'loss': va['loss']})
        epochs_run = epoch + 1
        if va['loss'] < best:
            best, best_epoch, wait = va['loss'], epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
        print(f"epoch {epoch} (lr={opt.param_groups[0]['lr']:.3e}) train MSE {tr['loss']:.6f} val MSE {va['loss']:.6f} "
              f"(best {best:.6f} @ {best_epoch}) {time.time() - t0:.0f}s", flush=True)
        if wait >= fc.patience:
            print('Early stopping', flush=True)
            break
    if not math.isfinite(best):
        raise SystemExit('no finite validation loss')
    args.best_epoch, args.best_val_loss, args.epochs_run = best_epoch, best, epochs_run
    args.save_arg(); logger.close()
    with open(os.path.join(args.save_log_path, 'best_epoch.txt'), 'w') as f:
        f.write(f'{best_epoch} {best:.6f} {epochs_run}\n')
    if fc.holdout:
        with open(os.path.join(args.save_log_path, 'final+result.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['weights', 'val_mse_best', 'best_epoch', 'epochs_run'])
            w.writerow(['best', f'{best:.6f}', best_epoch, epochs_run])
    else:
        model.load_state_dict(torch.load(ckpt, map_location=args.device))
        _, test_dl = data_provider(da, 'test')
        final_test(args, fc, model, train_dl, test_dl, args.save_log_path)
    print(f'done in {time.time() - t0:.0f}s')
    if torch.cuda.is_available():
        print(f'peak_mem_GB {torch.cuda.max_memory_allocated() / 2 ** 30:.2f}')


if __name__ == '__main__':
    main()
