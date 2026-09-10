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
from model import myModel
from test import test

# os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
torch.autograd.set_detect_anomaly(True)


class Exp_CL_Spikformer():
    
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.model = myModel(args).to(self.device)

    def _get_data(self, flag):
        
        args = self.args
        
        if flag  == 'test':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.test_bsz;
            freq = args.freq
        elif flag == 'val':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.batch_size;
            freq = args.detail_freq
        elif flag == 'pred':
            shuffle_flag = False;
            drop_last = False;
            batch_size = 1;
            freq = args.detail_freq
            args.data = 'pred'
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq
            
            
            
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
            gating=args.gating,
            train_mode='training',
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            pred_len=args.pred_len,
            seq_len=args.seq_len,
            qkv_bias=False, 
            mlp_ratios=args.mlp_emb,
            depths=args.num_layers, 
            sr_ratios=1,
            time_num_layers=args.time_num_layers,
            c_in=args.c_in,
            bias=args.bias, 
            tau=args.tau,
            spk_encoding=args.spk_encoding,
        )
        
        n_params = sum(p.numel() for p in spikformer.parameters() if p.requires_grad)
        print(f"creating model >> number of parameters : {n_params}")
        setattr(args, "model_params", n_params)
        
        spikformer = spikformer.to(args.device)
        functional.reset_net(spikformer)
        
       
        optimizer1 = torch.optim.AdamW([{'params' : spikformer.parameters()}, 
                                        # {'params' : weighted_sum_loss.parameters()}
                                        ],
                                       lr=args.lr, weight_decay=args.weight_decay)

        optimizer2 = torch.optim.AdamW(spikformer.parameters(), lr=min(0.001, args.lr * 10))

        scheduler1 = get_scheduler(args.scheduler, optimizer1, max_lr=args.lr, min_lr=min(1e-6, args.lr * 1e-2), max_epochs=args.epoch)
        if args.scheduler == 'cosine' : scheduler1.step(0)

        set_random_seed(args.seed)
        
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