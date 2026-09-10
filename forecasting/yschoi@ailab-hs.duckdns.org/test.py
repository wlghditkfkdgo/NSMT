from argparse import ArgumentParser
import torch
import torch.nn as nn
import numpy as np
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall

import os
from sys import stdout

# snn
from spikingjelly.clock_driven import functional
from syops import get_model_complexity_info

# model
from timm import create_model
from timm.optim.optim_factory import create_optimizer_v2

from config import set_random_seed, parse_arguments, Config
from utils import get_energy_consumption
from model import LOAD_MODEL
from data_provider.data_factory import data_provider
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score
from analysis import *

import matplotlib.pyplot as plt
# model loading

def R2(pred, true):
    true = true.ravel()
    pred = pred.ravel()
    mean = np.mean(true, axis=0)
    return 1-(((true-pred)**2).sum()/(((true-mean)**2).sum()))

    
def load_model(args):
    saved_model_path = os.path.join(args.save_result_path, "model_state", f"best+model.pt")
    
    spikformer = create_model(
        'spikformer',
        pretrained=False,
        pretrained_cfg=None,
        pretrained_cfg_overlay=None,
        checkpoint_path=saved_model_path,
        drop_rate=0.,
        keep_ratio = args.keep_ratio,
        drop_path_rate=0.,
        drop_block_rate=None,
        gating=args.gating,
        train_mode='training',
        pred_len=args.pred_len,
        seq_len=args.seq_len,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        qkv_bias=False, 
        mlp_ratios=args.mlp_ratios,
        depths=args.num_layers, 
        sr_ratios=1,
        time_num_layers=args.time_num_layers,
        c_in=args.c_in,
        bias=args.bias, 
        tau=args.tau,
        spk_encoding=args.spk_encoding,
    )

    spikformer = spikformer.to(args.device)

    
    print(f"Model was successfully loaded. (epoch = {args.saved_epoch[-1]:03d})")
    
    return spikformer


def plot_predicted_time_series(x, pred, true, args,
                               sample_idx=0,
                               channel_idx=0,
                               filename=None):
    """
    x      : 입력 시계열 (Tensor)  [B, seq_len, C] 가정
    pred   : 예측 시계열 (Tensor)  [B, pred_len, C]
    true   : GT 시계열 (Tensor)    [B, pred_len, C]
    args   : Config (save_log_path, pred_len, seq_len 등 포함)
    sample_idx, channel_idx: 몇 번째 샘플 / 채널을 그릴지
    """


    # ---- 텐서 -> numpy ----
    # x: [B, L, C]
    x_np = x[sample_idx].detach().cpu().numpy()        # [L, C] (또는 [L])
    pred_np = pred[sample_idx].detach().cpu().numpy()  # [pred_len, C]
    true_np = true[sample_idx].detach().cpu().numpy()  # [pred_len, C]

    # 채널 선택
    if x_np.ndim == 2:
        hist = x_np[:, channel_idx]          # [L]
    else:
        hist = x_np                          # [L]

    if pred_np.ndim == 2:
        fut_pred = pred_np[:, channel_idx]   # [pred_len]
        fut_true = true_np[:, channel_idx]   # [pred_len]
    else:
        fut_pred = pred_np
        fut_true = true_np

    seq_len = hist.shape[0]
    pred_len = fut_pred.shape[0]

    # ---- 시간축 설정 ----
    t_hist = np.arange(seq_len)
    # 마지막 입력 타임스텝에서부터 예측/GT를 이어서 그림
    t_future = np.arange(seq_len - 1, seq_len - 1 + pred_len)

    # ---- Figure 생성: 가로로 길고 세로로 짧게 ----
    fig, ax = plt.subplots(figsize=(10, 2.5))

    # 입력 시그널
    ax.plot(t_hist, hist, label="Input", linewidth=1.0, color="black")

    # 경계선 (입력/예측 구분)
    ax.axvline(seq_len - 1, linestyle="--", linewidth=0.8, color="gray")

    # GT & 예측
    ax.plot(t_future, fut_true, label="Ground Truth", linewidth=0.8)
    ax.plot(t_future, fut_pred, label="Prediction", linewidth=1.0)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")

    ax.legend(loc="upper left", fontsize=5)
    ax.set_xlim(0, t_future[-1])

    # 여백 최소화 (논문용)
    plt.tight_layout()

    # ---- 저장 ----
    os.makedirs(args.save_log_path, exist_ok=True)
    if filename is None:
        filename = f"predicted_time_series_s{sample_idx}_c{channel_idx}.png"
    save_path = os.path.join(args.save_log_path, filename)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Predicted time series figure saved to `{save_path}`")
    

def test(args:Config, model=None):
    set_random_seed(args.seed)
    
    if (args.model == 'myModel') and (args.gating == 'original') and (args.analysis):
        
        torch.cuda.empty_cache()
        neo_causal_eval(args)
        
        torch.cuda.empty_cache()
        run_token_aligned_path_analyses(args)
        
        torch.cuda.empty_cache()
        run_all_path_analyses(args)
        torch.cuda.empty_cache()
        # pass
        
    set_random_seed(args.seed)
        
    _, loader = data_provider(args, flag='test')
            
    mse = MeanSquaredError()
    mae = MeanAbsoluteError()
    r2 = R2Score()
    
    if model is None:
        model = LOAD_MODEL[args.model](args, train=False)

    model.eval()

    criterion = nn.MSELoss()
    with torch.no_grad():
        epoch_loss = 0

        preds = []
        trues = []
        for i, batch in enumerate(loader):
            x, y, _, _ = batch
            x = x.float().to(args.device)
            y = y.float().to(args.device)
            
                
            functional.reset_net(model)
            output = model(x)
            
            if isinstance(output, tuple):
                
                output, org_x, rec_x = output

            if output.dim() == 4:
                y = y.repeat(model.T, 1, 1, 1)[:, :, -args.pred_len:, :]
                output = output[:, :, -args.pred_len:, :]
            if output.dim() == 3:
                output = output[:, -args.pred_len:, :]
                y = y[:, -args.pred_len:, :]
            loss = criterion(output, y)
            epoch_loss += loss.item()
            
            
            pred = output.mean(0).detach().cpu() if output.dim() == 4 else output.detach().cpu()
            true = y.mean(0).detach().cpu() if y.dim() == 4 else y.detach().cpu()
            
            preds.append(pred)
            trues.append(true)
            
            if i == 0:
                # 첫 배치의 첫 샘플, 첫 채널을 논문용 figure로 저장
                plot_predicted_time_series(x, pred, true, args,
                                        sample_idx=0,
                                        channel_idx=0,
                                        filename="forecast_example.png")
        preds = torch.cat(preds)
        trues = torch.cat(trues)
        
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        mse.update(preds.contiguous(), trues.contiguous())
        mae.update(preds.contiguous(), trues.contiguous())
        r2 = R2(preds.numpy(), trues.numpy())
        
        input_res = x[0].shape
        
        model_info_per_layer_path = os.path.join(args.save_log_path, 'model+info+per+layer.txt') 
        file_out = open(model_info_per_layer_path, 'w', encoding='utf-8')
       
        model.train_mode = 'testing'
        functional.reset_net(model)
        ops, params, fr = get_model_complexity_info(
                                    model=model,
                                    input_res=(input_res,), 
                                    dataloader=loader,
                                    as_strings=False,
                                    print_per_layer_stat=True,
                                    # custom_modules_hooks=modules,
                                    # ignore_modules=ignore_modules,
                                    verbose=False,
                                    ost=file_out,
                                )
        
        file_out.close()
        
    test_result = {
        'loss' : epoch_loss/len(loader),
        'mse' : mse.compute().item(),
        'mae' : mae.compute().item(),
        'R2' : r2.item(),
    }
    
    print("Test was successfully done")

    with open(args.save_log_path + '/final+result.csv', 'a', encoding='utf-8') as log_csv:
        print("loss", "mse", "mae", "R2", "total_op", "ACop", "MACop", "capacity", "firing_rate", "energy",
              sep=", ", end="\n", file=log_csv)
        
        print(f"{test_result['loss']:.6f}", 
              f"{test_result['mse']:.6f}", 
              f"{test_result['mae']:.6f}",
              f"{test_result['R2']:.6f}", 
            #   f"{test_result['sim']:.6f}",
              f"{ops[0] / 1e6:.2f} M Ops",
              f"{ops[1] / 1e6:.2f} M Ops",
              f"{ops[2] / 1e6:.2f} M Ops",
              f"{params / 1e6:.4f} M",
              f"{fr:.4f} %",
              f"{get_energy_consumption(O_ac=ops[1], O_mac=ops[2], unit='u'):.2f} uJ",
              sep=", ", end="\n", file=log_csv)
        
    print(f"Final result saved to `{args.save_log_path}`")
    
    # savefig_path = args.save_log_path
    
    # spikformer.eval()
    # spikformer.train_mode = 'visual'
    # spikformer.head = nn.Identity()
    
    # plot_eval(model=spikformer, loader=loader, num_classes=args.num_classes, save_path=savefig_path, device=args.device)
    
    # print(f"Final tsne result saved to `{savefig_path}`")
    

    
        
if __name__ == '__main__':
    
    
    config = parse_arguments()
    args = Config()
    
    if config.config:
        config_path = config.config
        args.load_args(config_path, config)
    
    else:
        args.set_args(config)
        
    set_random_seed(args.seed)
    args.print_info()
    
    test(args)
    torch.set_grad_enabled(True)
    
    
    