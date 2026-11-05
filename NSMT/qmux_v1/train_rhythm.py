"""Stage 9 training entry for CinC 2017 and SHD (MODI_PROJECT_LOG 2026-10-19). The MIT-BIH entry train.py is untouched.

Same model zoo (LOAD_MODEL: myModel = NSMT / Hippo-only, neoMixer = attention replaced by the neocortex network), same loss
form as train.py (CE x (1 - alpha) + aux MSE x alpha when the model has a neocortex time_block, else CE), AdamW + the
framework's timm cosine schedule. Differences from train.py, all forced by the new protocols:
  * separate validation / test sets; the best-validation epoch is tested once on the test set (log/), the last epoch too
    (log/last/, for reporting);
  * the RNG is seeded once; epoch e shuffles with seed*1000+e and CinC crops depend on (seed, epoch, record), so every
    epoch sees new crops (train.py re-seeds every epoch, which repeats the same order);
  * CinC: CAL class weights from the training part (--loss_kind cal) or plain CE (--loss_kind ce).
Aux-loss options (MODI 2026-10-20): --alpha 0 = aux loss off (plain CE), --alpha 0.5 / 0.9 = its weight, --keep_ratio 0 =
reconstruct the full patched signal instead of its DCT low-pass part.
usage: python train_rhythm.py --data CINC2017|SHD --model myModel|neoMixer ... (see scripts/run_rhythm.sh)
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional

from config import Config, parse_arguments, set_random_seed
from load_model import LOAD_MODEL
from utils import EpochLog, get_scheduler
import rhythm_data as rd
import rhythm_eval as re_


def cal_weights(counts):
    c = torch.as_tensor(counts, dtype=torch.float64).clamp(min=1)
    w = torch.log(c.max() / c)
    return torch.maximum(torch.ones_like(w), w).float()


def train_epoch(model, dl, criterion, opt, ratio, device):
    model.train(); model.train_mode = 'training'
    tot = ce_s = mse_s = 0.; n = 0
    for x, y in dl:
        x = x.float().to(device); y = y.long().to(device)
        out = model(x)
        if isinstance(out, tuple):
            z, _, _, org_x, rec_x = out
            mse = ((org_x - rec_x) ** 2).mean()
        else:
            z, mse = out, torch.zeros((), device=device)
        ce = criterion(z, y)
        if hasattr(model, 'time_block') and ratio > 0:
            loss = ce * (1 - ratio) + mse * ratio
        else:                                    # no neocortex, or the aux loss switched off (--alpha 0, MODI 2026-10-20):
            loss = ce                            # plain CE; the decoder gets no gradient (mse is still logged)
        opt.zero_grad(); loss.backward(); opt.step()
        functional.reset_net(model)
        tot += loss.item(); ce_s += ce.item(); mse_s += float(mse); n += 1
    return {'loss': tot / n, 'ce': ce_s / n, 'mse': mse_s / n}


def main():
    cfg = parse_arguments()
    args = Config(); args.set_args(cfg)
    set_random_seed(args.seed)
    t0 = time.time()
    data = rd.build(args, args.seed)
    args.num_classes, args.num_channels, args.seq_len = data['num_classes'], data['num_channels'], data['seq_len']
    args.data_info = data['info']
    print('data', data['info'])
    args.print_info()
    model = LOAD_MODEL[args.model](args, True).to(args.device)
    functional.reset_net(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'creating model >> number of parameters : {n_params}  tokens {getattr(model, "num_patches", "?")}  T {getattr(model, "T", "?")}'
          f'  T_h {getattr(model, "T_h", getattr(model, "T", "?"))}')
    args.model_params = n_params
    if str(args.data).upper() == 'CINC2017' and args.loss_kind == 'cal':
        w = cal_weights(data['counts']); print('CAL weights', w.tolist())
        criterion = nn.CrossEntropyLoss(weight=w).to(args.device)
    else:
        criterion = nn.CrossEntropyLoss().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = get_scheduler(args.scheduler, opt, max_lr=args.max_lr, min_lr=min(1e-6, args.lr * 1e-2), max_epochs=args.epoch)
    if args.scheduler == 'cosine':
        sched.step(0)
    logger = EpochLog(args.save_log_path, kids=0)
    best, best_epoch = -1., -1
    for epoch in range(args.epoch):
        if hasattr(data['train'], 'epoch'):
            data['train'].epoch = epoch
        dl = rd.loader(data['train'], args.batch_size, shuffle=True, seed=args.seed, epoch=epoch)
        tr = train_epoch(model, dl, criterion, opt, args.alpha, args.device)
        P, Y, _ = re_.predict(model, data['val'], args.device, bs=args.eval_bs)
        vm, _ = re_.metrics(P, Y, args.data)
        val = {'score': vm['score'], 'acc': vm['acc'], 'macro_f1': vm['macro_f1']}
        if args.scheduler == 'cosine':
            sched.step(epoch + 1)
        else:
            sched.step()
        logger.write(epoch=epoch, lr=opt.param_groups[0]['lr'], train_result=tr, val_result=val)
        if vm['score'] > best:
            best, best_epoch = vm['score'], epoch
            torch.save(model.state_dict(), os.path.join(args.save_model_state_path, 'checkpoint.pth'))
        print(f'epoch {epoch} val score {vm["score"]:.4f} (best {best:.4f} @ {best_epoch}) {time.time() - t0:.0f}s', flush=True)
    torch.save(model.state_dict(), os.path.join(args.save_model_state_path, 'best+model.pt'))   # framework name: last epoch
    args.best_epoch, args.best_val_score = best_epoch, best
    args.save_arg(); logger.close()
    with open(os.path.join(args.save_log_path, 'best_epoch.txt'), 'w') as f:
        f.write(f'{best_epoch} {best:.6f}\n')
    re_.final_test(args, model, data['test'], os.path.join(args.save_log_path, 'last'), tag='last')
    model.load_state_dict(torch.load(os.path.join(args.save_model_state_path, 'checkpoint.pth'), map_location=args.device))
    re_.final_test(args, model, data['test'], args.save_log_path, tag='best')
    print(f'done in {time.time() - t0:.0f}s')
    if torch.cuda.is_available():                                       # MODI 2026-10-21: memory planning for shared GPUs
        print(f'peak_mem_GB {torch.cuda.max_memory_allocated() / 2 ** 30:.2f}')


if __name__ == '__main__':
    main()
