"""MODI 2026-10-28: PPG-DaLiA heart-rate regression (Log-NCDE protocol, ppg_data.py) with the models of this folder as causal
per-token regressors (ppg_model.py). Pre-registration: docs/MODI_PROJECT_LOG.md 2026-10-28.

Training: MSE on the scaled labels (all 390 labels of every window), AdamW, the framework's timm cosine schedule, a fixed number of
epochs; the best-validation-MSE weights (checkpoint.pth) and the last-epoch weights (best+model.pt, framework name) are tested.
Test metrics over all test windows x 390 labels: MSE and MAE on the scaled labels (the Log-NCDE unit; tables use MSE x 1e-2) and
MAE in BPM (scaled error x (max - min) / 2 of the window's subject). Predictions of the best weights on validation and test are
saved (preds_val.npz / preds_test.npz, local) for the smoothing controls (ppg_smooth.py).
Outputs follow train_rhythm.py: best_log_0.csv, train_0 / val_0 TensorBoard events, logargs.txt, model_state/{config.pt,
checkpoint.pth, best+model.pt}, log/final+result.csv (best) and log/last/final+result.csv.
usage: python train_ppg.py --data PPG --model myModel|neoMixer|gru --attn_window W ... (see scripts/run_ppg.sh)
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional
from syops import get_model_complexity_info
from syops.ops import LIF_syops_counter_hook
from torch.utils.data import DataLoader

from config import Config, parse_arguments, set_random_seed
from load_model import LOAD_MODEL
from utils import EpochLog, get_scheduler, get_energy_consumption
from energy import FiringRateRecorder
from neurons import LIFNeuron, FLIFNeuron
from ppg_data import PPGWindows, ROOT
from ppg_model import TokenRegressor, GRURegressor


def split_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--attn_window', type=int, default=0)
    ap.add_argument('--split_seed', type=int, default=0)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--smoke_windows', type=int, default=0, help='smoke tests only: keep the first n windows of every split')
    fc, rest = ap.parse_known_args()
    sys.argv = sys.argv[:1] + rest
    return fc


def build(args):
    if args.model == 'gru':
        return GRURegressor(patch_size=args.patch_size, channels=args.num_channels, hidden=args.embed_dim, seq_len=args.seq_len)
    return TokenRegressor(LOAD_MODEL[args.model](args, True), args.embed_dim, attn_window=args.attn_window)


def loader(ds, bs, shuffle, seed=0, epoch=0, workers=2):
    g = torch.Generator(); g.manual_seed(int(seed) * 1000 + int(epoch))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, generator=g if shuffle else None, num_workers=workers, drop_last=False)


def train_epoch(model, dl, opt, device):
    model.train(); model.train_mode = 'training'
    tot = 0.; n = 0
    for x, y, _ in dl:
        x = x.float().to(device); y = y.float().to(device)
        loss = ((model(x) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        functional.reset_net(model)
        tot += loss.item(); n += 1
    return {'loss': tot / n}


@torch.no_grad()
def evaluate(model, dl, device, bpm_range, recorder=None, keep=False):
    model.eval(); model.train_mode = 'testing'
    se = ae = ab = 0.; cnt = 0; P, Y, S = [], [], []
    functional.reset_net(model)
    for x, y, s in dl:
        x = x.float().to(device); y = y.float().to(device)
        out = model(x)
        if recorder is not None:
            recorder.add_samples(x.shape[0])
        functional.reset_net(model)
        d = (out - y).double()
        half = torch.as_tensor((bpm_range[s.numpy(), 1] - bpm_range[s.numpy(), 0]) / 2., device=device).unsqueeze(1)
        se += float((d ** 2).sum()); ae += float(d.abs().sum()); ab += float((d.abs() * half).sum()); cnt += d.numel()
        if keep:
            P.append(out.float().cpu().numpy()); Y.append(y.float().cpu().numpy()); S.append(s.numpy())
    m = {'mse': se / cnt, 'mae': ae / cnt, 'mae_bpm': ab / cnt, 'n': cnt}
    if keep:
        m['pred'], m['y'], m['subject'] = np.concatenate(P), np.concatenate(Y), np.concatenate(S)
    return m


def final_test(args, model, dl, out_dir, tag, bpm_range):
    os.makedirs(out_dir, exist_ok=True)
    spiking = args.model != 'gru'
    rec = FiringRateRecorder(model.inner) if spiking else None
    m = evaluate(model, dl, args.device, bpm_range, recorder=rec)
    if spiking:
        rec.detach()
        model.train_mode = 'testing'; functional.reset_net(model)
        x0 = next(iter(dl))[0]
        with open(os.path.join(out_dir, 'model+info+per+layer.txt'), 'w', encoding='utf-8') as fo:
            ops, params, fr = get_model_complexity_info(model=model, input_res=(tuple(x0.shape[1:]),), dataloader=dl, as_strings=False,
                                                        print_per_layer_stat=True, verbose=False, ost=fo,
                                                        custom_modules_hooks={LIFNeuron: LIF_syops_counter_hook, FLIFNeuron: LIF_syops_counter_hook})
        syn_uJ = get_energy_consumption(O_ac=ops[1], O_mac=ops[2], unit='u')
        en = rec.summary(syn_energy_J=get_energy_consumption(O_ac=ops[1], O_mac=ops[2]) * 1e-12)
        rec.write_layers(os.path.join(out_dir, 'neuron_energy.csv'))
        e_head, e_vals = ['total_op_M', 'AC_op_M', 'MAC_op_M', 'params_M', 'syops_firing_rate', 'syn_energy_uJ'] + en['columns'], \
            [f'{ops[0] / 1e6:.4f}', f'{ops[1] / 1e6:.4f}', f'{ops[2] / 1e6:.4f}', f'{params / 1e6:.6f}', f'{fr:.6f}', f'{syn_uJ:.6f}'] + en['values']
    else:                                                             # non-spiking reference: no energy accounting
        params = sum(p.numel() for p in model.parameters())
        e_head, e_vals = ['params_M'], [f'{params / 1e6:.6f}']
    head = ['weights', 'mse', 'mae', 'mae_bpm', 'val_mse_best', 'n'] + e_head
    row = [tag, f"{m['mse']:.8f}", f"{m['mae']:.8f}", f"{m['mae_bpm']:.6f}", f"{getattr(args, 'best_val_mse', float('nan')):.8f}", str(m['n'])] + e_vals
    with open(os.path.join(out_dir, 'final+result.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(head); w.writerow(row)
    print(f"[{tag}] test mse={m['mse']:.6f} mae={m['mae']:.6f} mae_bpm={m['mae_bpm']:.4f}", flush=True)
    return m


def main():
    fc = split_args()
    cfg = parse_arguments()
    args = Config(); args.set_args(cfg)
    for k, v in vars(fc).items():
        setattr(args, k, v)
    npz = os.path.join(ROOT, 'processed', f'logncde_seed{fc.split_seed}.npz')
    ds = {s: PPGWindows(npz, s) for s in ('train', 'val', 'test')}
    if fc.smoke_windows:
        for d in ds.values():
            d.idx = d.idx[:fc.smoke_windows]
    bpm_range = ds['train'].bpm_range
    args.num_classes, args.num_channels, args.seq_len = 1, 6, 49920
    args.data_info = {s: len(d) for s, d in ds.items()}
    set_random_seed(args.seed)
    t0 = time.time()
    args.print_info()
    model = build(args).to(args.device)
    functional.reset_net(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'creating model >> number of parameters : {n_params}  tokens {model.num_patches}  T {getattr(model, "T", None)}'
          f'  T_h {getattr(getattr(model, "inner", None), "T_h", None)}  attn_window {fc.attn_window}  windows {args.data_info}', flush=True)
    args.model_params = n_params
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = get_scheduler(args.scheduler, opt, max_lr=args.max_lr, min_lr=min(1e-6, args.lr * 1e-2), max_epochs=args.epoch)
    if args.scheduler == 'cosine':
        sched.step(0)
    logger = EpochLog(args.save_log_path, kids=0)
    best, best_epoch = float('inf'), -1
    ckpt = os.path.join(args.save_model_state_path, 'checkpoint.pth')
    for epoch in range(args.epoch):
        tr = train_epoch(model, loader(ds['train'], args.batch_size, True, args.seed, epoch, fc.num_workers), opt, args.device)
        va = evaluate(model, loader(ds['val'], args.eval_bs, False, workers=fc.num_workers), args.device, bpm_range)
        if not math.isfinite(tr['loss']) or not math.isfinite(va['mse']):
            raise RuntimeError(f'non-finite loss at epoch {epoch}: train {tr["loss"]} val {va["mse"]}')
        if args.scheduler == 'cosine':
            sched.step(epoch + 1)
        else:
            sched.step()
        logger.write(epoch=epoch, lr=opt.param_groups[0]['lr'], train_result=tr, val_result={'mse': va['mse'], 'mae_bpm': va['mae_bpm']})
        if va['mse'] < best:
            best, best_epoch = va['mse'], epoch
            torch.save(model.state_dict(), ckpt)
        print(f"epoch {epoch} (lr={opt.param_groups[0]['lr']:.3e}) train mse {tr['loss']:.5f} val mse {va['mse']:.5f} "
              f"bpm {va['mae_bpm']:.3f} (best {best:.5f} @ {best_epoch}) {time.time() - t0:.0f}s", flush=True)
    torch.save(model.state_dict(), os.path.join(args.save_model_state_path, 'best+model.pt'))   # framework name: last epoch
    args.best_epoch, args.best_val_mse = best_epoch, best
    args.save_arg(); logger.close()
    with open(os.path.join(args.save_log_path, 'best_epoch.txt'), 'w') as f:
        f.write(f'{best_epoch} {best:.8f}\n')
    test_dl = loader(ds['test'], args.eval_bs, False, workers=fc.num_workers)
    final_test(args, model, test_dl, os.path.join(args.save_log_path, 'last'), 'last', bpm_range)
    model.load_state_dict(torch.load(ckpt, map_location=args.device))
    final_test(args, model, test_dl, args.save_log_path, 'best', bpm_range)
    for split in ('val', 'test'):                                       # best weights: predictions for the smoothing controls
        m = evaluate(model, loader(ds[split], args.eval_bs, False, workers=fc.num_workers), args.device, bpm_range, keep=True)
        np.savez(os.path.join(args.save_log_path, f'preds_{split}.npz'), pred=m['pred'], y=m['y'], subject=m['subject'], bpm_range=bpm_range)
    print(f'done in {time.time() - t0:.0f}s')
    if torch.cuda.is_available():
        print(f'peak_mem_GB {torch.cuda.max_memory_allocated() / 2 ** 30:.2f}')


if __name__ == '__main__':
    main()
