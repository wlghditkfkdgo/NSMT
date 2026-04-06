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

from typing import Optional, Callable, Tuple
import os
from utils import *
from config import *
from data_provider.data_factory import data_provider
from load_model import LOAD_MODEL
from test import test

from copy import deepcopy   

# os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
# torch.autograd.set_detect_anomaly(True)

@torch.no_grad()
def ema_update(target, source, decay=0.999):
    for pt, ps in zip(target.parameters(), source.parameters()):
        pt.mul_(decay).add_(ps, alpha=1.0 - decay)

def pre_training_one_epoch(model, data_loader:DataLoader, optimizer, train:bool, device:torch.device):
    
    if train:
        model.train()
        model.train_mode = 'pre_training'
    else:
        model.eval()

    # has_tb = hasattr(model, 'time_block')
    epoch_rec = 0
    epoch_consol = 0
    # if hasattr(model, 'time_block'): 
        # prev_weights = deepcopy(model.time_block.state_dict())
        # tb_ema = deepcopy(model.time_block).eval()
        # for p in tb_ema.parameters():
        #     p.requires_grad = False
    
    for batch in data_loader:
        x, _ = batch
        x = x.float().to(device)
        
        rec = model.pretrain(x)

        # if has_tb: 
        #     consol_loss = consolidation_loss(model.time_block.state_dict(), prev_weights)
            # ema_update(tb_ema, model.time_block, decay=0.999)
            # consol_loss = sum((p1 - p2).pow(2).sum() for p1, p2 in zip(model.time_block.parameters(), tb_ema.parameters())) 
            
        loss = rec# + consol_loss * 0.01 if has_tb else rec
        
        if train:
            if isinstance(optimizer, tuple):
                optimizer[0].zero_grad()
                optimizer[1].zero_grad()
                loss.backward(retain_graph=True)
                optimizer[0].step()
                optimizer[1].step()
                
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        # if has_tb: prev_weights = deepcopy(model.time_block.state_dict())
        
        epoch_rec += rec.item()
        # if has_tb: epoch_consol += consol_loss.item()
        functional.reset_net(model)
        
    return {
            'rec' : epoch_rec/len(data_loader),
            # 'consol' : epoch_consol/len(data_loader) if hasattr(model, 'time_block') else 1.,
            }

def train_one_epoch(model, data_loader:DataLoader, weighted_sum_loss, criterion, optimizer, ratio:float, device:torch.device):
    
    model.train()
    weighted_sum_loss.train()
    model.train_mode = 'training'

    epoch_totloss = 0
    epoch_rec = 0
    epoch_distill = 0
    epoch_ce = 0
    epoch_consol = 0
    has_tb = hasattr(model, 'time_block')

    # if hasattr(model, 'time_block'): 
        # prev_weights = deepcopy(model.time_block.state_dict())

    
    for batch in data_loader:
        x, y = batch
        x = x.float().to(device)
        y = y.float().to(device)

        output = model(x)
         
        if isinstance(output, tuple):
            
            out, x_data, x_time, org_x, rec_x = output
            rec = ((org_x - rec_x)**2).mean()
            distill = ((x_data - x_time)**2).mean()
            
        else:
            out = output
            rec = torch.tensor([0]).to(device)
            
        f_dim=-1 if model.features == 'MS' else 0
        out = out[:, :, f_dim:]
        score = criterion(out, x).mean()
        
        # if has_tb: 
        #     consol_loss = consolidation_loss(model.time_block.state_dict(), prev_weights)
            # ema_update(tb_ema, model.time_block, decay=0.999)
            # consol_loss = sum((p1 - p2).pow(2).sum() for p1, p2 in zip(model.time_block.parameters(), tb_ema.parameters()))
            
        loss = score * (1-ratio) + rec * ratio if has_tb else score
        
        
        if isinstance(optimizer, tuple):
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            loss.backward()
            optimizer[0].step()
            optimizer[1].step()
            
            epoch_rec += rec.item()
            epoch_ce += score.item()
            
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_ce += score.item()
            if isinstance(output, tuple): 
                epoch_rec += rec.item()
                epoch_distill +=distill.item()
            epoch_totloss += loss.item()

        # if has_tb: prev_weights = deepcopy(model.time_block.state_dict())
        functional.reset_net(model)

    if epoch_distill > 0:
        return {'loss' : epoch_totloss/len(data_loader),
                'score' : epoch_ce/len(data_loader),
                'rec' : epoch_rec/len(data_loader),
                'distill' : epoch_distill/len(data_loader),
                }

    else:
        return {'loss' : epoch_totloss/len(data_loader),
                'score' : epoch_ce/len(data_loader),
                }


def val_one_epoch(model, data_loader:DataLoader, weighted_sum_loss, criterion, ratio:float, device:torch.device):
    
    model.eval()
    weighted_sum_loss.eval()
    # model.train_mode = 'testing'
    epoch_score = 0
    epoch_rec = 0
    epoch_totloss= 0
    epoch_distill = 0
                
    score = MeanSquaredError()
    mae = MeanAbsoluteError()
    has_tb = hasattr(model, 'time_block')
    
    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            x = x.float().to(device)
            y = y.float().to(device)
            
            # emb = temporal_model(x)
            
            output = model(x)
         
            if isinstance(output, tuple):
                out, x_data, x_time, org_x, rec_x = output
                rec = ((org_x - rec_x)**2).mean()
                distill = ((x_data - x_time)**2).mean()
                    
            else:
                out = output
                
            f_dim=-1 if model.features == 'MS' else 0
            
            out = out[:, :, f_dim:]
            score = criterion(out.detach().cpu(), x.detach().cpu()).mean()
            loss = score * (1-ratio) + rec * ratio if has_tb else score

            epoch_score += score.item()
            if isinstance(output, tuple): epoch_rec += rec.item()
            if isinstance(output, tuple): epoch_distill += distill.item()
            
            epoch_totloss += loss.item()

            functional.reset_net(model)
            
    return {
        'loss' : epoch_totloss/len(data_loader),
        'score' : epoch_score/len(data_loader), 
        'rec' : epoch_rec/len(data_loader) if epoch_rec > 0 else 0., 
        'distill' : epoch_distill/len(data_loader) if epoch_distill > 0 else 0., 

            }
        

def train(args:Config):
    set_random_seed(args.seed)  
    for kids in range(args.n_folds):
        
        logger = EpochLog(args.save_log_path, kids=kids)
        early_stopping = EarlyStopping(verbose=True)
        
        _, train_loader = data_provider(args, flag='train')
        _, val_loader = data_provider(args, flag='val')
        # _. train_loader = data_provider(args, flag='test')

        # window_size = args.segment_len if args.segment_len > 0 else args.window_size
        
        model = LOAD_MODEL[args.model](args, train=True)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"creating model >> number of parameters : {n_params}")
        setattr(args, "model_params", n_params)
        
        model = model.to(args.device)
        functional.reset_net(model)
        weighted_sum_loss = AutomaticWeightedLoss(num=2).to(args.device)
        
        criterion = nn.MSELoss(reduce=False)
        # criterion = nn.L1Loss(reduce=False)

        optimizer1 = torch.optim.AdamW([{'params' : model.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
            
        scheduler1 = get_scheduler(args.scheduler, optimizer1, max_lr=args.lr, min_lr=0, max_epochs=args.epoch)
        if args.scheduler == 'cosine' : scheduler1.step(0)

        set_random_seed(args.seed)
        
        print(f"fine-tuning ...")
        for epoch in range(args.epoch):
            
            train_result = train_one_epoch(model, train_loader, weighted_sum_loss, criterion, optimizer1, args.alpha, args.device) 
            val_result = val_one_epoch(model, val_loader, weighted_sum_loss, criterion, args.alpha, args.device) 
            
            if args.scheduler == 'reduce':
                scheduler1.step(val_result['score'])
            elif args.scheduler == 'cosine':
                scheduler1.step(epoch+1)
            else:
                scheduler1.step()

            logger.write(epoch=epoch, lr=optimizer1.param_groups[0]['lr'], train_result=train_result, val_result=val_result)
            print(f"Current log saved to `{args.save_log_path}`")
                
            early_stopping(val_result['loss'], model, args.save_model_state_path)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
        # args.saved_epoch.append(epoch)
            
        best_model = model.state_dict() # inference mode
        
        torch.save(best_model, args.save_model_state_path + f"/best+model.pt")
        print(f"Model saved to `{args.save_model_state_path}`")

                    
        # final arguments save
        args.save_arg()
        logger.close()
        if args.test:
            
            last_epoch = args.epoch - 1
            last_saved_epoch = args.saved_epoch[-1]
            
            if last_saved_epoch == last_epoch:
                test(args=args, model=model)
                
            else:
                test(args=args)
        
        
if __name__ == '__main__' : 
    
    
    config = parse_arguments()
    args = Config()
    
    args.set_args(config)
    set_random_seed(args.seed) # 42
    args.print_info()

    train(args)