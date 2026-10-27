"""MODI 2026-10-27: ETT forecasting with the CinC models of this folder (generalization batch). The forecasting framework
(nsmt_flif_v1/forecasting/train.py) is not imported: its module names (config, layers, model, ...) collide with this folder.
Only its data provider is used (nsmt_flif_v1/forecasting/data_provider, self-contained), so the ETT splits, scaling and
windows are those of every earlier ETT run.

Training follows nsmt_flif_v1/forecasting/train.py for models without the auxiliary decoder: L1 loss on the last pred_len
steps, AdamW, ReduceLROnPlateau on the validation L1 (patience 2, factor 0.1, cooldown 1), early stopping on the validation
L1 (--patience), the best-validation weights (checkpoint.pth) and the last-epoch weights (best+model.pt, framework name) are
both tested. Test metrics are MSE / MAE over all test windows, steps and channels on the scaled data (as test.py).
Outputs follow train_rhythm.py: log/ = best-validation weights, log/last/ = last epoch; best_log_0.csv, train_0 / val_0
TensorBoard events, logargs.txt, model_state/{config.pt, checkpoint.pth, best+model.pt}.
usage: python train_forecast.py --data ETTh2 --data_path ETTh2.csv --root_path <ETT-small> --seq_len 96 --pred_len 96
       --model neoMixer|myModel ... (see nsmt_flif_v1/forecasting/scripts/run_ett_mixer.sh)
"""
import argparse
import csv
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional
from syops import get_model_complexity_info
from syops.ops import LIF_syops_counter_hook

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', 'forecasting'))             # after this folder: only `data_provider` is taken from there

from config import Config, parse_arguments, set_random_seed
from load_model import LOAD_MODEL
from utils import EpochLog, get_scheduler, get_energy_consumption
from energy import FiringRateRecorder
from neurons import LIFNeuron, FLIFNeuron
from forecast_wrap import Forecaster
from data_provider.data_factory import data_provider


def split_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--label_len', type=int, default=48)
    ap.add_argument('--features', default='M')
    ap.add_argument('--target', default='OT')
    ap.add_argument('--freq', default='h')
    ap.add_argument('--embed', default='timeF')
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--num_workers', type=int, default=2)
    fc, rest = ap.parse_known_args()
    sys.argv = sys.argv[:1] + rest
    return fc


def data_args(args, fc):
    return types.SimpleNamespace(data=args.data, embed=fc.embed, batch_size=args.batch_size, freq=fc.freq, task_name='forecasting',
                                 root_path=args.root_path, data_path=args.data_path, seq_len=args.seq_len, label_len=fc.label_len,
                                 pred_len=fc.pred_len, features=fc.features, target=fc.target, seasonal_patterns='Monthly',
                                 num_workers=fc.num_workers, augmentation_ratio=0)


def train_epoch(model, dl, opt, H, device):
    model.train(); model.train_mode = 'training'
    crit = nn.L1Loss()
    tot = 0.; n = 0
    for x, y, _, _ in dl:
        x = x.float().to(device); y = y.float().to(device)[:, -H:, :]
        out = model(x)
        loss = crit(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        functional.reset_net(model)
        tot += loss.item(); n += 1
    return {'loss': tot / n}


@torch.no_grad()
def evaluate(model, dl, H, device, recorder=None):
    """L1 (mean of per-batch L1, as forecasting/train.py val_one_epoch), MSE / MAE over all elements."""
    model.eval(); model.train_mode = 'testing'
    l1 = 0.; nb = 0; se = ae = 0.; cnt = 0
    functional.reset_net(model)
    for x, y, _, _ in dl:
        x = x.float().to(device); y = y.float().to(device)[:, -H:, :]
        out = model(x)
        if recorder is not None:
            recorder.add_samples(x.shape[0])
        functional.reset_net(model)
        d = (out - y).double()
        l1 += d.abs().mean().item(); nb += 1
        se += (d ** 2).sum().item(); ae += d.abs().sum().item(); cnt += d.numel()
    return {'loss': l1 / nb, 'mse': se / cnt, 'mae': ae / cnt, 'n': cnt}


def final_test(args, model, dl, out_dir, H, tag):
    os.makedirs(out_dir, exist_ok=True)
    rec = FiringRateRecorder(model.inner)                             # every neuron is in the body; its own names keep the neocortex prefixes
    m = evaluate(model, dl, H, args.device, recorder=rec)
    rec.detach()
    model.train_mode = 'testing'
    functional.reset_net(model)
    x0 = next(iter(dl))[0]
    with open(os.path.join(out_dir, 'model+info+per+layer.txt'), 'w', encoding='utf-8') as fo:
        ops, params, fr = get_model_complexity_info(model=model, input_res=(tuple(x0.shape[1:]),), dataloader=dl, as_strings=False,
                                                    print_per_layer_stat=True, verbose=False, ost=fo,
                                                    custom_modules_hooks={LIFNeuron: LIF_syops_counter_hook, FLIFNeuron: LIF_syops_counter_hook})
    syn_uJ = get_energy_consumption(O_ac=ops[1], O_mac=ops[2], unit='u')
    en = rec.summary(syn_energy_J=get_energy_consumption(O_ac=ops[1], O_mac=ops[2]) * 1e-12)
    rec.write_layers(os.path.join(out_dir, 'neuron_energy.csv'))
    head = ['weights', 'mse', 'mae', 'val_loss_best', 'n', 'total_op_M', 'AC_op_M', 'MAC_op_M', 'params_M', 'syops_firing_rate',
            'syn_energy_uJ'] + en['columns']
    row = [tag, f"{m['mse']:.6f}", f"{m['mae']:.6f}", f"{getattr(args, 'best_val_loss', float('nan')):.6f}", str(m['n']),
           f'{ops[0] / 1e6:.4f}', f'{ops[1] / 1e6:.4f}', f'{ops[2] / 1e6:.4f}', f'{params / 1e6:.6f}', f'{fr:.6f}', f'{syn_uJ:.6f}'] + en['values']
    with open(os.path.join(out_dir, 'final+result.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(head); w.writerow(row)
    print(f"[{tag}] test mse={m['mse']:.6f} mae={m['mae']:.6f} syn_uJ={syn_uJ:.4f}", flush=True)
    return m


def main():
    fc = split_args()
    cfg = parse_arguments()
    args = Config(); args.set_args(cfg)
    for k, v in vars(fc).items():
        setattr(args, k, v)
    args.num_classes, args.num_channels = 1, 1                        # channel-independent rows; the classification head is dropped
    set_random_seed(args.seed)
    t0 = time.time()
    da = data_args(args, fc)
    _, train_dl = data_provider(da, 'train')
    _, val_dl = data_provider(da, 'val')
    _, test_dl = data_provider(da, 'test')
    args.print_info()
    model = Forecaster(LOAD_MODEL[args.model](args, True), args.embed_dim, fc.pred_len, bias=args.bias).to(args.device)
    functional.reset_net(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'creating model >> number of parameters : {n_params}  tokens {model.num_patches}  T {model.T}'
          f'  T_h {getattr(model.inner, "T_h", model.T)}', flush=True)
    args.model_params = n_params
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = get_scheduler(args.scheduler, opt, max_lr=args.lr, min_lr=min(1e-6, args.lr * 1e-2), max_epochs=args.epoch)
    if args.scheduler == 'cosine':
        sched.step(0)
    set_random_seed(args.seed)
    logger = EpochLog(args.save_log_path, kids=0)
    best, best_epoch, wait = float('inf'), -1, 0
    ckpt = os.path.join(args.save_model_state_path, 'checkpoint.pth')
    epochs_run = 0
    for epoch in range(args.epoch):
        tr = train_epoch(model, train_dl, opt, fc.pred_len, args.device)
        va = evaluate(model, val_dl, fc.pred_len, args.device)
        if args.scheduler == 'reduce':
            sched.step(va['loss'])
        elif args.scheduler == 'cosine':
            sched.step(epoch + 1)
        else:
            sched.step()
        logger.write(epoch=epoch, lr=opt.param_groups[0]['lr'], train_result=tr,
                     val_result={'loss': va['loss'], 'mse': va['mse'], 'mae': va['mae']})
        epochs_run = epoch + 1
        if va['loss'] < best:                                          # forecasting/utils.EarlyStopping with delta 0
            best, best_epoch, wait = va['loss'], epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
        print(f"epoch {epoch} (lr={opt.param_groups[0]['lr']:.3e}) train L1 {tr['loss']:.5f} val L1 {va['loss']:.5f} "
              f"(best {best:.5f} @ {best_epoch}) {time.time() - t0:.0f}s", flush=True)
        if wait >= fc.patience:
            print('Early stopping', flush=True)
            break
    torch.save(model.state_dict(), os.path.join(args.save_model_state_path, 'best+model.pt'))   # framework name: last epoch
    args.best_epoch, args.best_val_loss, args.epochs_run = best_epoch, best, epochs_run
    args.save_arg(); logger.close()
    with open(os.path.join(args.save_log_path, 'best_epoch.txt'), 'w') as f:
        f.write(f'{best_epoch} {best:.6f} {epochs_run}\n')
    final_test(args, model, test_dl, os.path.join(args.save_log_path, 'last'), fc.pred_len, 'last')
    model.load_state_dict(torch.load(ckpt, map_location=args.device))
    final_test(args, model, test_dl, args.save_log_path, fc.pred_len, 'best')
    print(f'done in {time.time() - t0:.0f}s')
    if torch.cuda.is_available():
        print(f'peak_mem_GB {torch.cuda.max_memory_allocated() / 2 ** 30:.2f}')


if __name__ == '__main__':
    main()
