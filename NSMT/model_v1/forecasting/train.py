import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
# snn
from spikingjelly.clock_driven import functional
# timm
from timm.optim import create_optimizer_v2
from timm.models import create_model, load_checkpoint
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy
from einops import rearrange

from typing import Optional, Callable, Tuple
import os
from utils import *
from config import *
from data_provider.data_factory import data_provider
# import model as model
from model import LOAD_MODEL
from test import test
# os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
# torch.autograd.set_detect_anomaly(True)

class EntropyLoss(nn.Module):
    def __init__(self, eps = 1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        b = x * torch.log(x + self.eps)
        b = -1.0 * b.sum(dim=1)
        b = b.mean()
        return b

def train_one_epoch(model, data_loader:DataLoader, optimizer, ratio:float, device:torch.device):
    
    model.train()
    # weighted_sum_loss.train()
    model.train_mode = 'training'

    epoch_totloss = 0
    epoch_rec = 0
    epoch_distill = 0
    epoch_ce = 0
    epoch_consol = 0
    epoch_kd = 0
    pred_len = model.pred_len
    
    criterion = nn.L1Loss()
    Rec = nn.L1Loss()
    entropy_loss = EntropyLoss()
    
    for batch in data_loader:
        x, y, _, _ = batch
        x = x.float().to(device)
        y = y.float().to(device)
        y = y[:, -pred_len:, :]
        
        functional.reset_net(model)
        output = model(x)
        
        z_neo = None
        trend_pred = None
        y0 = y  # [B, pred_len, M] kept before any T-repeat below (for trend supervision)
        if isinstance(output, tuple):

            if len(output) >= 6:
                if getattr(model, 'use_trend_decomp', False):
                    # trend/detail decomposition: tail is the Neocortex trend [BC, pred_len]
                    out, x_data, x_time, org_x, rec_x, trend_pred = output[:6]
                else:
                    # teacher-student: extra Student forecast z_neo at the tail
                    out, x_data, x_time, org_x, rec_x, z_neo = output[:6]
            elif len(output) > 3:
                out, x_data, x_time, org_x, rec_x = output
            rec = (org_x - rec_x).abs().mean() if getattr(model, 'aux_l1', False) else ((org_x - rec_x)**2).mean()
                #t b 1 c p
                # rec_x = rec_x.squeeze(2).permute(1, 2, 0, 3).contiguous()
                # rec += F.margin_ranking_loss(rec_x[:, :, :-1], rec_x[:, :, 1:], target=torch.ones_like(rec_x[:, :, 1:]))
                # rec = ((org_x - rec_x)**2).mean(-1, keepdim=True)
                # rec = (rec * model.filter_mask).mean() / model.filter_mask.mean()
                # x_time = rearrange(x_time, 't (b c) l p -> t b l p c', l=1, b=x.shape[0])
                # rec = entropy_loss(x_time.reshape(-1, x_time.shape[-1]))
            distill = ((x_data - x_time)**2).mean()

        else:
            out = output
            rec = torch.tensor([0]).to(device)

        if out.dim() == 4:
            y = y.repeat(model.T, 1, 1, 1)
            out = out[:, :, -pred_len:, :]
        if out.dim() == 3:
            out = out[:, -pred_len:, :]

        ce = criterion(out, y)

        # loss = ce + rec * ratio if isinstance(output, tuple) else ce
        # loss = weighted_sum_loss(ce, rec * ratio) if isinstance(output, tuple) else ce
        loss = ce * (1. - ratio) + (rec * ratio) if isinstance(output, tuple) else ce

        # Teacher(Hippocampus) -> Student(Neocortex) distillation (training only).
        # Student matches the teacher forecast (KD) and also learns the task.
        kd = torch.tensor(0.0, device=device)
        if z_neo is not None:
            z_neo_h = z_neo[:, :, -pred_len:, :] if z_neo.dim() == 4 else z_neo[:, -pred_len:, :]
            kd_mimic = ((z_neo_h - out.detach()) ** 2).mean()   # mimic teacher
            kd_task = criterion(z_neo_h, y)                      # solve the task
            kd = kd_mimic + kd_task
            loss = loss + getattr(model, 'kd_weight', 0.5) * kd

        # Trend/detail decomposition: supervise the Neocortex trend against the
        # low-frequency component of the target so the trend can only come from
        # the Neocortex (makes the memory causally necessary).
        if trend_pred is not None:
            Blow = model.B_low                                   # [K, pred_len]
            y_norm = (y0 - model.means) / model.stdev            # [B, pred_len, M]
            yb = rearrange(y_norm, 'b l c -> (b c) l')           # [BC, pred_len]
            y_low = (yb @ Blow.t()) @ Blow                       # [BC, pred_len] low-freq target
            loss = loss + getattr(model, 'lam_trend', 0.5) * ((trend_pred - y_low) ** 2).mean()

        if isinstance(optimizer, tuple):
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            loss.backward()
            optimizer[0].step()
            optimizer[1].step()
            
            epoch_rec += rec.item()
            epoch_ce += ce.item()
            
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # get_grad_norm(model)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            epoch_ce += ce.item()
            if isinstance(output, tuple):
                epoch_rec += rec.item()
                epoch_distill += distill.item()
            if z_neo is not None:
                epoch_kd += kd.item()
            epoch_totloss += loss.item()



    if epoch_rec > 0:
        result = {'loss' : epoch_totloss/len(data_loader),
                'ce' : epoch_ce/len(data_loader),
                'rec' : epoch_rec/len(data_loader),
                'distill' : epoch_distill/len(data_loader),
                }
        if epoch_kd > 0:
            result['kd'] = epoch_kd/len(data_loader)
        return result

    else:
        return {'loss' : epoch_totloss/len(data_loader),
                'ce' : epoch_ce/len(data_loader),
                'rec' : 1.,}


def val_one_epoch(model, data_loader:DataLoader, ratio:float, device:torch.device):
    
    model.eval()
    # weighted_sum_loss.eval()
    # model.train_mode = 'testing'
    epoch_ce = 0
    epoch_rec = 0
    epoch_totloss= 0
    epoch_distill = 0
                
    mse = MeanSquaredError()
    mae = MeanAbsoluteError()
    criterion = nn.L1Loss()
    Rec = nn.L1Loss()
    pred_len = model.pred_len
    entropy_loss = EntropyLoss()
    
    with torch.no_grad():
        for batch in data_loader:
            x, y, _, _ = batch
            x = x.float().to(device)
            y = y.float().to(device)
            y = y[:, -pred_len:, :]
            
            # emb = temporal_model(x)
            functional.reset_net(model)
            output = model(x)
         
            if isinstance(output, tuple):

                if len(output) >= 6:
                    out, x_data, x_time, org_x, rec_x, _z_neo = output[:6]
                elif len(output) > 3:
                    out, x_data, x_time, org_x, rec_x = output
                rec = (org_x - rec_x).abs().mean() if getattr(model, 'aux_l1', False) else ((org_x - rec_x)**2).mean()
                    # t b l c p
                    # rec_x = rec_x.squeeze(2).permute(1, 2, 0, 3).contiguous()
                    # rec += F.margin_ranking_loss(rec_x[:, :, :-1], rec_x[:, :, 1:], target=torch.ones_like(rec_x[:, :, 1:]))
                    # rec = ((org_x - rec_x)**2).mean(-1, keepdim=True) # [t b c l]
                    # rec = (rec * model.filter_mask).mean() / model.filter_mask.mean()
                    # x_time = rearrange(x_time, 't (b c) l p -> t b l p c', l=1, b=x.shape[0])
                    # rec = entropy_loss(x_time.reshape(-1, x_time.shape[-1]))
                distill = ((x_data - x_time)**2).mean()

            else:
                out = output
                
            # review 7.6: align validation metric/model-selection with the FINAL test metric,
            # which averages the T spiking-step predictions first (horizon_stats reduce_T='mean').
            # The T-step L1 (ce) is kept only for logging / training-consistency.
            if out.dim() == 4:
                out_step = out[:, :, -pred_len:, :]                    # [T, B, pred_len, C]
                ce = criterion(out_step, y.repeat(model.T, 1, 1, 1))   # T-step loss (logging only)
                out_eval = out_step.mean(0)                            # T-mean prediction == test-time metric
            else:
                out_eval = out[:, -pred_len:, :]
                ce = criterion(out_eval, y)

            epoch_ce += ce.item()
            if isinstance(output, tuple): epoch_rec += rec.item()
            if isinstance(output, tuple): epoch_distill += distill.item()

            loss = ce * (1. - ratio) + (rec * ratio) if isinstance(output, tuple) else ce

            epoch_totloss += loss.item()

            pred = out_eval.detach().cpu()                             # T-mean, matches final metric
            true = y.detach().cpu()

            mse.update(pred.contiguous(), true.contiguous())
            mae.update(pred.contiguous(), true.contiguous())
            
    return {
        'loss' : epoch_totloss/len(data_loader),
        'ce' : epoch_ce/len(data_loader), 
        'rec' : epoch_rec/len(data_loader) if epoch_rec > 0 else 0., 
        'distill' : epoch_distill/len(data_loader) if epoch_distill > 0 else 0., 
        'mse' : mse.compute().item(),
        'mae' : mae.compute().item(),
            }
        

def train(args:Config):
    set_random_seed(args.seed)

    logger = EpochLog(args.save_log_path, kids=0)
    early_stopping = EarlyStopping(verbose=True, patience=args.patience)

    _, train_loader = data_provider(args, flag='train')
    _, val_loader = data_provider(args, flag='val')

    model = LOAD_MODEL[args.model](args, True)

    # expose KD weight to train_one_epoch (only used when --use_kd is set)
    setattr(model, "kd_weight", getattr(args, "kd_weight", 0.5))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"creating model >> number of parameters : {n_params}")
    setattr(args, "model_params", n_params)

    model = model.to(args.device)
    functional.reset_net(model)

    optimizer1 = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler1 = get_scheduler(args.scheduler, optimizer1, max_lr=args.lr, min_lr=min(1e-6, args.lr * 1e-2), max_epochs=args.epoch)
    if args.scheduler == 'cosine' : scheduler1.step(0)

    alpha = args.alpha * (args.pred_len/720)
    # R1 ablation: load a trained checkpoint and go straight to test (no retraining).
    _load_ckpt = getattr(args, 'load_ckpt', '')
    if _load_ckpt:
        model.load_state_dict(torch.load(_load_ckpt, map_location=args.device))
        print(f"[load_ckpt] loaded `{_load_ckpt}` (wrec_mode={getattr(args,'wrec_mode','full')}), skipping training")
        args.save_arg()
        if args.test:
            test(args=args, model=model)
        return
    print(f"fine-tuning ...")
    for epoch in range(args.epoch):
        train_result = train_one_epoch(model, train_loader, optimizer1, alpha, args.device)
        val_result = val_one_epoch(model, val_loader, alpha, args.device)

        if args.scheduler == 'reduce':
            scheduler1.step(val_result['loss'])
        elif args.scheduler == 'cosine':
            scheduler1.step(epoch+1)
        else:
            scheduler1.step()

        logger.write(epoch=epoch, lr=optimizer1.param_groups[0]['lr'], train_result=train_result, val_result=val_result)
        print(f"Current log saved to `{args.save_log_path}`")

        early_stopping(val_result['mse'], model, args.save_model_state_path)   # review 7.6: select on the T-mean MSE that we actually report

        if early_stopping.early_stop:
            print("Early stopping")
            break

    best_ckpt_path = os.path.join(args.save_model_state_path, "best+model.pt")
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=args.device))
        print(f"Best model loaded from `{best_ckpt_path}`")
    else:
        torch.save(model.state_dict(), best_ckpt_path)
        print(f"Model saved to `{best_ckpt_path}`")

    args.save_arg()
    logger.close()
    if args.test:
        test(args=args, model=model)
        
        
if __name__ == '__main__' : 
    
    
    config = parse_arguments()
    args = Config()
    
    args.set_args(config)
    set_random_seed(args.seed) # 42
    args.print_info()

    train(args)