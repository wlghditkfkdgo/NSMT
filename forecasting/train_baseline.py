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
import model as model
from test import test

# os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
torch.autograd.set_detect_anomaly(True)

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
    model.train_mode = 'training'

    pred_len = model.pred_len
    
    criterion = nn.MSELoss()
    entropy_loss = EntropyLoss()
    
    for batch in data_loader:
        x, y, _, _ = batch
        x = x.float().to(device)
        y = y.float().to(device)
        y = y[:, -pred_len:, :]
        
        functional.reset_net(model)
        output = model(x)
        
        if isinstance(output, tuple):
            
            if len(output) > 3:
                out, x_data, x_time, org_x, rec_x = output
                rec = ((org_x - rec_x)**2).mean()
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
            
        ce = criterion(out[:, :, -pred_len:, :], y)
        
        # loss = ce + rec * ratio if isinstance(output, tuple) else ce
        # loss = weighted_sum_loss(ce, rec * ratio) if isinstance(output, tuple) else ce
        loss = ce + rec * ratio if isinstance(output, tuple) else ce
        
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
            epoch_totloss += loss.item()

        

    if epoch_rec > 0:
        return {'loss' : epoch_totloss/len(data_loader),
                'ce' : epoch_ce/len(data_loader),
                'rec' : epoch_rec/len(data_loader),
                'distill' : epoch_distill/len(data_loader),
                }

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
    criterion = nn.MSELoss()
    pred_len = model.pred_len
    entropy_loss = EntropyLoss()
    
    has_tb = model.gating == 'original'
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
                    
                if len(output) > 3:
                    out, x_data, x_time, org_x, rec_x = output
                    rec = ((org_x - rec_x)**2).mean()
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
                
            out = out[:, :, -pred_len:, :]
            
            if out.dim() == 4:
                y = y.repeat(model.T, 1, 1, 1)
            
            ce = criterion(out, y)

            epoch_ce += ce.item()
            if isinstance(output, tuple): epoch_rec += rec.item()
            if isinstance(output, tuple): epoch_distill += distill.item()
        
            loss = ce + rec * ratio if isinstance(output, tuple) else ce

            epoch_totloss += loss.item()
            
            pred = out.detach().cpu()
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

    for kids in range(args.n_folds):
        
        logger = EpochLog(args.save_log_path, kids=kids)
        early_stopping_pre = EarlyStopping(verbose=True, patience=args.patience)
        early_stopping = EarlyStopping(verbose=True, patience=args.patience)
        
        _, train_loader = data_provider(args, flag='train')
        _, val_loader = data_provider(args, flag='val')
        # _. train_loader = data_provider(args, flag='test')

        # window_size = args.segment_len if args.segment_len > 0 else args.window_size
        
        spikformer = create_model(
            'spikformer',
            pretrained=False,
            pretrained_cfg=None,
            pretrained_cfg_overlay=None,
            drop_rate=0.,
            drop_path_rate=0.,
            drop_block_rate=None,
            keep_ratio = args.keep_ratio,
            gating=args.gating,
            train_mode='training',
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            pred_len=args.pred_len,
            seq_len=args.seq_len,
            qkv_bias=False, 
            mlp_ratios=args.mlp_ratios,
            depths=args.num_layers, 
            sr_ratios=1,
            time_num_layers=args.time_num_layers,
            c_in=args.c_in,
            bias=args.bias, 
            tau=args.tau,
            perm=args.perm,
            spk_encoding=args.spk_encoding,
        )
        
        n_params = sum(p.numel() for p in spikformer.parameters() if p.requires_grad)
        print(f"creating model >> number of parameters : {n_params}")
        setattr(args, "model_params", n_params)
        
        spikformer = spikformer.to(args.device)
        functional.reset_net(spikformer)
        
        # weighted_sum_loss = AutomaticWeightedLoss(num=2).to(args.device)
        
        # optimizer1 = create_optimizer_v2(spikformer.parameters(), opt='adamw', lr=args.lr, weight_decay=args.weight_decay)
        # optimizer2 = create_optimizer_v2(spikformer.parameters(), opt='adamw', lr=min(0.001, args.lr * 10), weight_decay=args.weight_decay)
        optimizer1 = torch.optim.AdamW([{'params' : spikformer.parameters()}, 
                                        # {'params' : weighted_sum_loss.parameters()}
                                        ],
                                       lr=args.lr, weight_decay=args.weight_decay)

        optimizer2 = torch.optim.AdamW(spikformer.parameters(), lr=min(0.001, args.lr * 10))

        scheduler1 = get_scheduler(args.scheduler, optimizer1, max_lr=args.lr, min_lr=min(1e-6, args.lr * 1e-2), max_epochs=args.epoch)
        if args.scheduler == 'cosine' : scheduler1.step(0)

        set_random_seed(args.seed)
        
        print(f"pre-training ...")
        for epoch in range(args.warm_up_epoch):

            pre_train_loss = pre_training_one_epoch(spikformer, train_loader, optimizer2, True, args.device)
            pre_train_val_loss = pre_training_one_epoch(spikformer, val_loader, optimizer2, False, args.device)

            logger.logging(epoch=epoch, 
                           train_result={'pre_train_loss' : pre_train_loss['rec']}, 
                           val_result={'pre_train_val_loss' : pre_train_val_loss['rec']})

            logger.verbose(epoch=epoch,
                           lr=optimizer2.param_groups[0]['lr'],
                           train_result=pre_train_loss,
                           val_result=pre_train_val_loss)
            
            early_stopping_pre(pre_train_val_loss['rec'], spikformer, None)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
        print(f"fine-tuning ...")
        for epoch in range(args.epoch):
            
            train_result = train_one_epoch(spikformer, train_loader, optimizer1, args.alpha, args.device) 
            val_result = val_one_epoch(spikformer, val_loader, args.alpha, args.device) 
            
            if args.scheduler == 'reduce':
                scheduler1.step(val_result['ce'])
            elif args.scheduler == 'cosine':
                scheduler1.step(epoch+1)
            else:
                scheduler1.step()

            logger.write(epoch=epoch, lr=optimizer1.param_groups[0]['lr'], train_result=train_result, val_result=val_result)
            print(f"Current log saved to `{args.save_log_path}`")
                
            early_stopping(val_result['ce'], spikformer, args.save_model_state_path)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
        # args.saved_epoch.append(epoch)
            
        best_model = spikformer.state_dict() # inference mode
        
        torch.save(best_model, args.save_model_state_path + f"/best+model.pt")
        print(f"Model saved to `{args.save_model_state_path}`")

                    
        # final arguments save
        args.save_arg()
        logger.close()
        if args.test:
            
            last_epoch = args.epoch - 1
            last_saved_epoch = args.saved_epoch[-1]
            
            if last_saved_epoch == last_epoch:
                test(args=args, model=spikformer)
                
            else:
                test(args=args)
        
        
if __name__ == '__main__' : 
    
    
    config = parse_arguments()
    args = Config()
    
    args.set_args(config)
    set_random_seed(args.seed) # 42
    args.print_info()

    train(args)