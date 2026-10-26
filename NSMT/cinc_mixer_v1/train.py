import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter, summary
from torchvision import transforms
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision

# snn
from spikingjelly.clock_driven import functional

# timm
from timm.optim import create_optimizer_v2
from timm.models import create_model, load_checkpoint
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy

import os

from utils import *
from config import *
from dataloader import create_loader

import model as model
from load_model import LOAD_MODEL

from test import test, evaluation

torch.jit.fuser("fuser0")
# nsmt_flif_v1: debugging switches off (they only synchronise kernels / check backward for NaN; the computation is
# the same -- check_cls.py c2 compares parameters after training steps with and without them). Opt back in with
# NSMT_DEBUG=1.
if os.environ.get('NSMT_DEBUG', '0') == '1':
    os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    torch.autograd.set_detect_anomaly(True)

def pre_training_one_epoch(model, data_loader:DataLoader, optimizer, train:bool, device:torch.device):
    
    if train:
        model.train()
        model.train_mode = 'pre_training'
    else:
        model.eval()

    epoch_mse = 0
    epoch_consol = 0
    for batch in data_loader:

        data = batch[0].float().to(device)
        label = batch[1].to(device)

        mse = model(data)

        loss = mse if hasattr(model, 'time_block') else mse
        
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
                
        epoch_mse += mse.item()
        functional.reset_net(model)
        
    return {
            'mse' : epoch_mse/len(data_loader),
            }

def train_one_epoch(model, data_loader:DataLoader, criterion, optimizer, ratio:float, device:torch.device):
    
    model.train()
    model.train_mode = 'training'

    epoch_totloss = 0
    epoch_mse = 0
    epoch_ce = 0

    for batch in data_loader:
        data = batch[0].float().to(device)
        label = batch[1].long().squeeze(-1).to(device)

        output = model(data)
         
        if isinstance(output, tuple):
            out, x_data, x_time, org_x, rec_x = output
            mse = ((org_x - rec_x)**2).mean()
        else:
            out = output
            mse = torch.tensor([0]).to(device)
            
        ce = criterion(out, label)
        
        loss = ce * (1 - ratio) + mse * ratio  if hasattr(model, 'time_block') else ce
        
        
        if isinstance(optimizer, tuple):
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            ce.backward(retain_graph=True)
            mse.backward(retain_graph=True)
            optimizer[0].step()
            optimizer[1].step()
            
            epoch_mse += mse.item()
            epoch_ce += ce.item()
            
        else:
            optimizer.zero_grad()
            loss.backward()
            _gc = getattr(model, 'grad_clip', 0.0)
            if _gc and _gc > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=_gc)
            optimizer.step()
            epoch_ce += ce.item()
            if isinstance(output, tuple): 
                epoch_mse += mse.item()
            epoch_totloss += loss.item()

        functional.reset_net(model)

    if epoch_mse > 0:
        return {'loss' : epoch_totloss/len(data_loader),
                'ce' : epoch_ce/len(data_loader),
                'mse' : epoch_mse/len(data_loader),
                }

    else:
        return {'loss' : epoch_totloss/len(data_loader),
                'ce' : epoch_ce/len(data_loader),
                'mse' : 1.,}


def val_one_epoch(model, data_loader:DataLoader, criterion, num_classes:int, ratio:float, device:torch.device):
    
    model.eval()
    # model.train_mode = 'testing'
    epoch_ce = 0
    epoch_mse = 0
    epoch_totloss= 0

    tot_acc = MulticlassAccuracy().to(device)
    acc = MulticlassAccuracy(num_classes=num_classes, average=None).to(device)
    f1 = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
    pre = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    _all_pred = []; _all_label = []                                   # R5: per-class F1 + confusion
    with torch.no_grad():
        for batch in data_loader:
            data = batch[0].float().to(device)
            label = batch[1].long().squeeze(-1).to(device)
            
            output = model(data)
         
            if isinstance(output, tuple):
                out, x_data, x_time, org_x, rec_x = output
                mse = ((org_x - rec_x)**2).mean()
                    
            else:
                out = output
                
            ce = criterion(out, label)

            epoch_ce += ce.item()
            if isinstance(output, tuple): epoch_mse += mse.item()
            
            loss = ce * (1 - ratio) + mse * ratio if isinstance(output, tuple) else ce
            epoch_totloss += loss.item()

            functional.reset_net(model)

            pred = get_pred(out)
            
            tot_acc.update(pred, label)
            acc.update(pred, label)
            f1.update(pred, label)
            pre.update(pred, label)
            _all_pred.append(pred.detach().cpu()); _all_label.append(label.detach().cpu())

    import numpy as _np
    from sklearn.metrics import f1_score as _f1, confusion_matrix as _cm
    _p = torch.cat(_all_pred).numpy(); _l = torch.cat(_all_label).numpy()
    _f1pc = _f1(_l, _p, average=None, labels=list(range(num_classes)), zero_division=0)
    print("[PERCLASS_F1] " + " ".join(f"{v:.4f}" for v in _f1pc))
    print("[CONFUSION] " + " | ".join(" ".join(map(str, row)) for row in _cm(_l, _p, labels=list(range(num_classes)))))
    return {
        'loss' : epoch_totloss/len(data_loader),
        'ce' : epoch_ce/len(data_loader), 
        'mse' : epoch_mse/len(data_loader) if epoch_mse > 0 else 0., 
        'tot_acc' : tot_acc.compute().item(),
        'acc' : acc.compute(), 
        'f1' : f1.compute().item(), 
        'pre' : pre.compute().item(),
            }
        

def train(args:Config):

    logger = EpochLog(args.save_log_path, kids=0)
    early_stopping_pre = EarlyStopping(verbose=True)
    early_stopping_fine = EarlyStopping(verbose=True)
    
    loader, class_num_samples = create_loader(train=True, batch_size=args.batch_size, num_classes=args.num_classes, data_root=args.root_path+'/5class_mitbih_train_7_187.csv', train_val_ratio=None, data_window_size=187, sampling='None', num_workers=args.num_workers)

    if isinstance(loader, tuple):
        train_loader = loader[0]
        val_loader = loader[1]
        
    else:
        train_loader = loader
        val_loader, _ = create_loader(train=False, batch_size=args.batch_size, num_classes=args.num_classes, data_root=args.root_path+'/5class_mitbih_test_3_187.csv', data_window_size=187, num_workers=args.num_workers)


    model = LOAD_MODEL[args.model](args, True)
    model.grad_clip = getattr(args, 'grad_clip', 0.0)   # P2: gradient clipping for recurrent BPTT stability
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"creating model >> number of parameters : {n_params}")
    setattr(args, "model_params", n_params)
    
    model = model.to(args.device)
    functional.reset_net(model)
    # # time_block = time_block.to(args.device)
    
    cal_weights = get_class_weights(class_num_samples=class_num_samples)

    criterion = nn.CrossEntropyLoss(weight=cal_weights).to(args.device)
    test_criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer1 = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler1 = get_scheduler(args.scheduler, optimizer1, max_lr=args.max_lr, min_lr=min(1e-6, args.lr * 1e-2), max_epochs=args.epoch)
    if args.scheduler == 'cosine' : scheduler1.step(0)
    
    for epoch in range(args.epoch):
        
        set_random_seed(args.seed)
        
        train_result = train_one_epoch(model, train_loader, criterion, optimizer1, args.alpha, args.device) 
        val_result = val_one_epoch(model, val_loader, test_criterion, args.num_classes, args.alpha, args.device) 
        
        if args.scheduler == 'reduce':
            scheduler1.step(val_result['loss'])
        elif args.scheduler == 'cosine':
            scheduler1.step(epoch+1)
        else:
            scheduler1.step()
        
        logger.write(epoch=epoch, lr=optimizer1.param_groups[0]['lr'], train_result=train_result, val_result=val_result)
        print(f"Current log saved to `{args.save_log_path}`")
        
        if args.scheduler == 'reduce':
            early_stopping_fine(val_result['ce'], model, None)
            
            if early_stopping_fine.early_stop:
                print("Early stopping")
                break
        
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
            evaluation(args=args, model=model)
            
        else:
            evaluation(args=args, model=model)
    
if __name__ == '__main__' : 
    
    
    config = parse_arguments()
    args = Config()
    
    args.set_args(config)
    set_random_seed(args.seed) # 42
    args.print_info()

    train(args)