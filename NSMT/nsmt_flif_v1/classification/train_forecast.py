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
MODI 2026-11-19: --aux_alpha > 0 adds NSMT's reconstruction loss for dual-path forecasters (see split_args); best_log_0.csv then
also has train_task (forecast L1) and train_rec (reconstruction MSE); train_loss is the weighted sum. 0 (default) = unchanged.
usage: python train_forecast.py --data ETTh2 --data_path ETTh2.csv --root_path <ETT-small> --seq_len 96 --pred_len 96
       --model neoMixer|myModel ... (see nsmt_flif_v1/forecasting/scripts/run_ett_mixer.sh)
"""
import argparse
import csv
import math
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
from utils import EpochLog, get_scheduler, get_energy_consumption, make_adamw
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
    # MODI 2026-11-19 (aux loss, user request 11-19): NSMT's reconstruction loss for the dual-path forecasters. Training loss =
    # (1 - r) * L1 + r * MSE(weak_decoder(Neocortex output), low-pass target) with r = aux_alpha * pred_len / 720 (the NSMT
    # forecasting recipe, nsmt_flif_v1/forecasting/train.py); the low-pass fraction is --keep_ratio. Validation, early stopping and
    # model selection stay on the forecast L1. 0 (default) = unchanged.
    ap.add_argument('--aux_alpha', type=float, default=0.0)
    # MODI 2026-11-19 (overfitting test): dropout on the flat head's input during training (Forecaster head_dropout); 0 = unchanged.
    ap.add_argument('--head_dropout', type=float, default=0.0)
    # MODI 2026-11-27 (approved plan 1-snuggly-galaxy.md): weight w of the accumulation aux (--aux_past rand, Forecaster.aux_past_loss;
    # named --past_aux_w, not --aux_past_w: this parser's prefix matching read '--aux_past rand' as '--aux_past_w rand', smoke 11-28):
    # training loss = (1 - w) * L1 + w * MSE(past reconstruction); validation, early stopping and model selection stay on the forecast
    # L1. Not with --aux_alpha. 0 (default) = unchanged.
    ap.add_argument('--past_aux_w', type=float, default=0.0)
    fc, rest = ap.parse_known_args()
    sys.argv = sys.argv[:1] + rest
    return fc


def data_args(args, fc):
    return types.SimpleNamespace(data=args.data, embed=fc.embed, batch_size=args.batch_size, freq=fc.freq, task_name='forecasting',
                                 root_path=args.root_path, data_path=args.data_path, seq_len=args.seq_len, label_len=fc.label_len,
                                 pred_len=fc.pred_len, features=fc.features, target=fc.target, seasonal_patterns='Monthly',
                                 num_workers=fc.num_workers, augmentation_ratio=0)


class LinearWarmup:
    """MODI 2026-11-23 (hyperparameter range 2): linear lr warmup. Called once before every optimizer step; step k (0-based) of the
    first W uses lr * (k + 1) / W in every parameter group, so step W - 1 already runs at the full lr and later steps are not touched
    (ReduceLROnPlateau then owns the lr)."""
    def __init__(self, opt, total):
        if total < 1:
            raise ValueError('warmup: at least one step')
        self.opt, self.total, self.k = opt, int(total), 0
        self.base = [g['lr'] for g in opt.param_groups]

    def __call__(self):
        if self.k < self.total:
            f = (self.k + 1) / self.total
            for g, b in zip(self.opt.param_groups, self.base):
                g['lr'] = b * f if self.k + 1 < self.total else b
            self.k += 1


def train_epoch(model, dl, opt, H, device, r=0.0, hook=None, marks=(), warm=None, aux_fn=None):
    """MODI 2026-11-20 (hyperparameter range 1): hook(step) runs after the training steps listed in marks (1-based; the in-epoch
    validations of --val_per_epoch), then training mode is restored. hook None (default): unchanged.
    MODI 2026-11-23 (range 2): warm (a LinearWarmup) is called before every optimizer step. None (default): unchanged.
    MODI 2026-11-27: aux_fn = the aux loss of the r branch (None = model.aux_rec_loss, as before; model.aux_past_loss for --past_aux_w)."""
    model.train(); model.train_mode = 'training'
    crit = nn.L1Loss()
    tot = 0.; n = 0
    if r > 0:                                                          # MODI 2026-11-19: aux loss (see split_args)
        tt = tr_ = 0.
        for x, y, _, _ in dl:
            x = x.float().to(device); y = y.float().to(device)[:, -H:, :]
            out = model(x)
            task = crit(out, y); rec = model.aux_rec_loss() if aux_fn is None else aux_fn()
            loss = task * (1. - r) + rec * r
            if warm is not None:
                warm()
            opt.zero_grad(); loss.backward(); opt.step()
            functional.reset_net(model)
            tot += loss.item(); tt += task.item(); tr_ += rec.item(); n += 1
            if hook is not None and n in marks:
                hook(n); model.train(); model.train_mode = 'training'
        return {'loss': tot / n, 'task': tt / n, 'rec': tr_ / n}
    for x, y, _, _ in dl:
        x = x.float().to(device); y = y.float().to(device)[:, -H:, :]
        out = model(x)
        loss = crit(out, y)
        if warm is not None:
            warm()
        opt.zero_grad(); loss.backward(); opt.step()
        functional.reset_net(model)
        tot += loss.item(); n += 1
        if hook is not None and n in marks:
            hook(n); model.train(); model.train_mode = 'training'
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
    if getattr(model, 'head_kind', 'flat') == 'h6':                    # MODI 2026-11-04 (H6): the shortcut alone, same weights
        model.body_off = True
        ml = evaluate(model, dl, H, args.device)
        model.body_off = False
        head += ['mse_lin', 'mae_lin']
        row += [f"{ml['mse']:.6f}", f"{ml['mae']:.6f}"]
        print(f"[{tag}] shortcut alone: test mse={ml['mse']:.6f} mae={ml['mae']:.6f}", flush=True)
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
    model = Forecaster(LOAD_MODEL[args.model](args, True), args.embed_dim, fc.pred_len, bias=args.bias,
                       head_kind=getattr(args, 'fc_head', 'flat'), head_dim=getattr(args, 'fc_head_dim', 8),
                       seq_len=args.seq_len, head_dropout=fc.head_dropout).to(args.device)   # MODI 2026-11-04 (H6); 11-19 head_dropout
    functional.reset_net(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'creating model >> number of parameters : {n_params}  tokens {model.num_patches}  T {model.T}'
          f'  T_h {getattr(model.inner, "T_h", model.T)}', flush=True)
    args.model_params = n_params
    aux_r = fc.aux_alpha * fc.pred_len / 720.                         # MODI 2026-11-19 (aux loss)
    if fc.aux_alpha != 0:
        if model.kind != 'dual' or not (0. < aux_r < 1.):
            raise ValueError(f'--aux_alpha: dual-path forecasters only, 0 < r < 1 (got kind {model.kind}, r {aux_r})')
        model.aux = True
        args.aux_r = aux_r
        print(f'aux loss on: r = {aux_r:.6f} (aux_alpha {fc.aux_alpha} x {fc.pred_len} / 720), keep_ratio {args.keep_ratio}', flush=True)
    if fc.past_aux_w != 0 or getattr(args, 'aux_past', 'none') != 'none':   # MODI 2026-11-27 (accumulation aux)
        if model.kind != 'dual' or getattr(args, 'aux_past', 'none') != 'rand' or not (0. < fc.past_aux_w < 1.) or fc.aux_alpha != 0:
            raise ValueError(f'--past_aux_w / --aux_past rand: together, dual-path forecasters only, 0 < w < 1, not with --aux_alpha '
                             f'(got kind {model.kind}, aux_past {getattr(args, "aux_past", "none")}, w {fc.past_aux_w}, aux_alpha {fc.aux_alpha})')
        model.aux_past = True
        args.past_aux_w = fc.past_aux_w
        print(f'accumulation aux on: w = {fc.past_aux_w}', flush=True)
    opt = make_adamw(model, args.lr, args.weight_decay, bool(int(getattr(args, 'wd_exclude_1d', 0))))   # MODI 2026-11-20 (range 1)
    sched = get_scheduler(args.scheduler, opt, max_lr=args.lr, min_lr=min(1e-6, args.lr * 1e-2), max_epochs=args.epoch)
    if args.scheduler == 'cosine':
        sched.step(0)
    set_random_seed(args.seed)
    logger = EpochLog(args.save_log_path, kids=0)
    best, best_epoch, wait = float('inf'), -1, 0
    ckpt = os.path.join(args.save_model_state_path, 'checkpoint.pth')
    epochs_run = 0
    # MODI 2026-11-20 (hyperparameter range 1): K = --val_per_epoch validations per epoch; the K - 1 inner ones run after the steps
    # ceil(n * j / K), j = 1 .. K - 1, and take part in the best checkpoint and in early stopping (an epoch with any improvement
    # resets the wait); the scheduler and best_log_0.csv keep the end-of-epoch validation. K = 1: no hook, unchanged.
    K = int(getattr(args, 'val_per_epoch', 1))
    if K < 1:
        raise ValueError('--val_per_epoch >= 1')
    n_steps = len(train_dl)
    marks = {int(math.ceil(n_steps * j / K)) for j in range(1, K)}
    if K > 1 and (len(marks) != K - 1 or max(marks) >= n_steps):
        raise ValueError(f'--val_per_epoch {K}: too many for {n_steps} steps per epoch')
    inner, best_step = [], None
    # MODI 2026-11-23 (hyperparameter range 2): --lr_warmup v in (0, 1] = linear lr warmup over the first round(v * n_steps) optimizer
    # steps (at most the first epoch, so the plateau scheduler never acts during it). < 0 (default): none, unchanged.
    wu = float(getattr(args, 'lr_warmup', -1.0)); warm = None
    if wu >= 0:
        if not (0. < wu <= 1.) or args.scheduler != 'reduce':
            raise ValueError('--lr_warmup (train_forecast.py): 0 < value <= 1 epoch, with --scheduler reduce')
        warm = LinearWarmup(opt, int(round(wu * n_steps)))
        print(f'lr warmup on: {warm.total} steps of {n_steps} per epoch, lr {args.lr}', flush=True)
    for epoch in range(args.epoch):
        improved = False

        def hook(step, epoch=epoch):
            nonlocal best, best_epoch, best_step, improved
            vi = evaluate(model, val_dl, fc.pred_len, args.device)
            inner.append((epoch, step, vi['loss'], vi['mse'], vi['mae']))
            if vi['loss'] < best:
                best, best_epoch, best_step, improved = vi['loss'], epoch, step, True
                torch.save(model.state_dict(), ckpt)
        hk = hook if K > 1 else None
        tr = train_epoch(model, train_dl, opt, fc.pred_len, args.device, r=aux_r, hook=hk, marks=marks, warm=warm) if fc.aux_alpha != 0 else \
            train_epoch(model, train_dl, opt, fc.pred_len, args.device, r=fc.past_aux_w, hook=hk, marks=marks, warm=warm,
                        aux_fn=model.aux_past_loss) if fc.past_aux_w != 0 else \
            train_epoch(model, train_dl, opt, fc.pred_len, args.device, hook=hk, marks=marks, warm=warm)
        va = evaluate(model, val_dl, fc.pred_len, args.device)
        if K > 1:
            inner.append((epoch, n_steps, va['loss'], va['mse'], va['mae']))
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
            best_step = n_steps
            torch.save(model.state_dict(), ckpt)
        elif improved:                                                 # MODI 2026-11-20: an inner validation improved this epoch
            wait = 0
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
        f.write(f'{best_epoch} {best:.6f} {epochs_run}' + (f' {best_step}/{n_steps}' if K > 1 else '') + '\n')
    if K > 1:                                                          # MODI 2026-11-20: every validation, in order
        with open(os.path.join(args.save_log_path, 'val_inner.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['epoch', 'step', 'steps_per_epoch', 'val_loss', 'val_mse', 'val_mae'])
            for e_, s_, l_, m_, a_ in inner:
                w.writerow([e_, s_, n_steps, f'{l_:.6f}', f'{m_:.6f}', f'{a_:.6f}'])
    final_test(args, model, test_dl, os.path.join(args.save_log_path, 'last'), fc.pred_len, 'last')
    model.load_state_dict(torch.load(ckpt, map_location=args.device))
    final_test(args, model, test_dl, args.save_log_path, fc.pred_len, 'best')
    print(f'done in {time.time() - t0:.0f}s')
    if torch.cuda.is_available():
        print(f'peak_mem_GB {torch.cuda.max_memory_allocated() / 2 ** 30:.2f}')


if __name__ == '__main__':
    main()
