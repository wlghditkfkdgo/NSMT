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
from noise_injector import apply_gaussian_noise

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
        filename = f"predicted_time_series_s{sample_idx}_c{channel_idx}"
    save_path = os.path.join(args.save_log_path, filename)

    plt.savefig(save_path+'.png', dpi=300, bbox_inches="tight")
    plt.savefig(save_path+'.pdf', dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Predicted time series figure saved to `{save_path}`")
    

def test(args:Config, model=None):
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)  # 이 프로세스에 보이는 GPU를 1개로 제한
    torch.cuda.set_device(args.device)

    set_random_seed(args.seed)
    
    # if (args.model == 'myModel') and (args.gating == 'original') and (args.analysis):
        
    #     torch.cuda.empty_cache()
    #     neo_causal_eval(args)
        
    #     torch.cuda.empty_cache()
    #     run_token_aligned_path_analyses(args)
        
    #     torch.cuda.empty_cache()
    #     run_all_path_analyses(args)
    #     torch.cuda.empty_cache()
    #     # pass
        
    set_random_seed(args.seed)
        
    _, loader = data_provider(args, flag='test')
            
    mse = MeanSquaredError()
    mse_long = MeanSquaredError()
    mae = MeanAbsoluteError()
    mae_long = MeanAbsoluteError()
    r2 = R2Score()

    long_pred_len = 192
    
    if model is None:
        model = LOAD_MODEL[args.model](args, train=False)


    criterion = nn.L1Loss()
    with torch.no_grad():
        epoch_loss = 0
        model.eval()

        preds = []
        trues = []

        preds_long = []
        trues_long = []
        hstats = HorizonStats(pred_len=args.pred_len, device="cpu")
        for i, batch in enumerate(loader):
            x, y, _, _ = batch
            x = x.float().to(args.device)
            y = y.float().to(args.device)
            # ==========================================================
            # [추가된 코드] Inference 직전, 설정된 SNR 값이 있다면 노이즈 주입
            # ==========================================================
            if hasattr(args, 'snr') and (args.snr != 100):
                x = apply_gaussian_noise(x, args.snr)
            # ==========================================================
            functional.reset_net(model)
            output = model(x)
            pred_h, true_h, pred_long_h, true_long_h = extract_pred_true_for_metrics(
                output=output, y=y, pred_len=args.pred_len, long_pred_len=long_pred_len, reduce_T="mean"
            )

            loss = criterion(pred_h, true_h)
            epoch_loss += loss.item()

            preds.append(pred_h.cpu())
            trues.append(true_h.cpu())
            preds_long.append(pred_long_h.cpu())
            trues_long.append(true_long_h.cpu())

            hstats.update(pred_h, true_h)
            
            if i == 0:
                plot_predicted_time_series(x, pred_h, true_h, args,
                                        sample_idx=1,
                                        channel_idx=0,
                                        filename=f"forecast_example_{args.model}_{args.data}")
        mae_h, mse_h = hstats.compute()
        bin_summary = summarize_bins_from_horizon(mae_h, mse_h, pred_len=args.pred_len, far_k=192)

        # 저장 (그래프+CSV)
        save_horizon_artifacts(mae_h, mse_h, save_dir=args.save_log_path, prefix=f"h{args.pred_len}", make_plot=True)

        # 콘솔 출력
        print("[Horizon summary]")
        for k, v in bin_summary.items():
            print(f"  {k}: {v:.6f}")
        
        preds = torch.cat(preds)
        trues = torch.cat(trues)

        preds_long = torch.cat(preds_long)
        trues_long = torch.cat(trues_long)
        
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        preds_long = preds_long.reshape(-1, preds_long.shape[-2], preds_long.shape[-1])
        trues_long = trues_long.reshape(-1, trues_long.shape[-2], trues_long.shape[-1])
        
        mse.update(preds.contiguous(), trues.contiguous())
        mae.update(preds.contiguous(), trues.contiguous())

        mse_long.update(preds_long.contiguous(), trues_long.contiguous())
        mae_long.update(preds_long.contiguous(), trues_long.contiguous())
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
        'mse_long' : mse_long.compute().item(),
        'mae_long' : mae_long.compute().item(),
        'R2' : r2.item(),
    }
    
    print("Test was successfully done")

    head = ','.join(
        ["snr", "loss", "mse", "mae", "total_op", "ACop", "MACop", "capacity", "firing_rate", "energy", "mse_long", "mae_long", "R2",]
    )
    results_csv = ','.join([
        f"{args.snr if hasattr(args, 'snr') else 'None'}",
        f"{test_result['loss']:.6f}", 
        f"{test_result['mse']:.6f}", 
        f"{test_result['mae']:.6f}",
    #   f"{test_result['sim']:.6f}",
        f"{ops[0] / 1e6:.2f} M Ops",
        f"{ops[1] / 1e6:.2f} M Ops",
        f"{ops[2] / 1e6:.2f} M Ops",
        f"{params / 1e6:.4f} M",
        f"{fr:.4f} %",
        f"{get_energy_consumption(O_ac=ops[1], O_mac=ops[2], unit='u'):.2f} uJ",
        f"{test_result['mse_long']:.6f}", 
        f"{test_result['mae_long']:.6f}",
        f"{test_result['R2']:.6f}", ]
    )

    for k, v in test_result.items():
        value =  v if isinstance(v, float) else v.mean()
        print(f" > {k:10s}:{value:>5.3f}")

    with open(args.save_log_path + '/final+result.csv', 'a', encoding='utf-8') as log_csv:
        print(head, end="\n", file=log_csv)
        
        print(results_csv, end="\n", file=log_csv)
        
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

    for snr in config.snr_list:
        setattr(args, 'snr', snr)
        test(args)

    torch.set_grad_enabled(True)
    
    
    