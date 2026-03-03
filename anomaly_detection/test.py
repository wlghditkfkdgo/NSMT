from argparse import ArgumentParser
import torch
import torch.nn as nn
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
import numpy as np

import os
from sys import stdout

# snn
from spikingjelly.clock_driven import functional
from syops import get_model_complexity_info

# model
from timm import create_model
from timm.optim.optim_factory import create_optimizer_v2

from config import set_random_seed, parse_arguments, Config
from utils import get_energy_consumption, adjustment
import model as model
from model import TemporalBlock
from data_provider.data_factory import data_provider
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt


def load_model(args):
    saved_model_path = os.path.join(args.save_result_path, "model_state", f"best+model.pt")
    
    spikformer = create_model(
        'spikformer',
        pretrained=False,
        pretrained_cfg=None,
        pretrained_cfg_overlay=None,
        checkpoint_path=saved_model_path,
        drop_rate=0.,
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
        c_out=args.c_out,
        mlp_ratios=args.mlp_ratios,
        depths=args.num_layers, 
        sr_ratios=1,
        time_num_layers=args.time_num_layers,
        bias=args.bias, 
        tau=args.tau,
        features=args.features,
        spk_encoding=args.spk_encoding,
    )

    spikformer = spikformer.to(args.device)

    
    print(f"Model was successfully loaded. (epoch = {args.saved_epoch[-1]:03d})")
    
    return spikformer


def visualize_anomaly_example(args, model, threshold, test_loader, criterion):
    """
    test_loader에서 anomaly가 예측된 샘플 하나를 골라서

    - 입력 시계열 (가장 에러가 큰 채널 1개)
    - 그 채널의 reconstruction (모델 출력)
    - reconstruction error + threshold
    - 예측 anomaly 포인트

    를 함께 시각화해서 저장한다.
    (x, output shape 가 [B, T, C] 라는 가정)
    """
    model.eval()
    model.train_mode = 'testing'
    functional.reset_net(model)

    chosen_x = None        # [T, C]
    chosen_out = None      # [T, C]
    chosen_err = None      # [T, C]
    chosen_score = None    # [T]
    chosen_pred = None     # [T]

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch           # x: [B, T, C], y: [B, T]
            x = x.float().to(args.device)
            y = y.float().to(args.device)

            out = model(x)
            if isinstance(out, tuple):
                out = out[0]

            # per-feature squared error: [B, T, C]
            per_feat_err = criterion(x, out)       # MSELoss(reduction='none')
            # per-timestep score (평균): [B, T]
            score = per_feat_err.mean(-1)

            # anomaly flag: [B, T]
            pred_flags = (score > threshold).int()

            # anomaly가 하나라도 있는 샘플을 찾는다
            for i in range(x.shape[0]):
                if pred_flags[i].sum() > 0:
                    chosen_x = x[i].detach().cpu().numpy()          # [T, C]
                    chosen_out = out[i].detach().cpu().numpy()      # [T, C]
                    chosen_err = per_feat_err[i].detach().cpu().numpy()  # [T, C]
                    chosen_score = score[i].detach().cpu().numpy()  # [T]
                    chosen_pred = pred_flags[i].detach().cpu().numpy()  # [T]
                    break
            if chosen_x is not None:
                break

    if chosen_x is None:
        print("No predicted anomaly found. Skip visualization.")
        return

    T, C = chosen_x.shape
    time_axis = np.arange(T)

    # --- 가장 '이상한' 채널 고르기 ---
    anomaly_idx = np.where(chosen_pred == 1)[0]
    if len(anomaly_idx) > 0:
        # anomaly 구간에서 채널별 평균 에러 -> 가장 큰 채널 선택
        mean_err_per_channel = chosen_err[anomaly_idx].mean(axis=0)   # [C]
        feat_idx = int(np.argmax(mean_err_per_channel))
    else:
        # 혹시 모를 fallback: 전체 평균 에러 기준
        mean_err_per_channel = chosen_err.mean(axis=0)
        feat_idx = int(np.argmax(mean_err_per_channel))

    values = chosen_x[:, feat_idx]        # 입력
    recon = chosen_out[:, feat_idx]       # 복원
    score_1d = chosen_score               # [T]
    pred_1d = chosen_pred                 # [T]

    anomaly_idx = np.where(pred_1d == 1)[0]

    # --- 그림 그리기 ---
    os.makedirs(args.save_log_path, exist_ok=True)
    fig_path = os.path.join(args.save_log_path, "anomaly_example.png")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)

    # (1) 입력 시계열 + 복원 + anomaly 위치
    ax1.plot(time_axis, values, label=f"Input (feat {feat_idx})")
    ax1.plot(time_axis, recon, linestyle='--', label="Reconstruction")
    if len(anomaly_idx) > 0:
        ax1.scatter(time_axis[anomaly_idx],
                    values[anomaly_idx],
                    s=25, c='red', label='Predicted anomaly')
    ax1.set_ylabel("Value")
    ax1.legend(loc="upper right")

    # (2) reconstruction error + threshold
    ax2.plot(time_axis, score_1d, label="Reconstruction error")
    ax2.axhline(threshold, linestyle='--', label="Threshold")
    if len(anomaly_idx) > 0:
        ax2.scatter(time_axis[anomaly_idx],
                    score_1d[anomaly_idx],
                    s=25, c='red')
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Error")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"Anomaly example figure saved to `{fig_path}`")

def test(args:Config, model=None):
    set_random_seed(args.seed)
    _, test_loader = data_provider(args, flag='test')
    _, train_loader = data_provider(args, flag='train')
            
    mse = MeanSquaredError()
    mae = MeanAbsoluteError()
    
    if model is None:
        spikformer = load_model(args)
        
    else:
        spikformer = model

    spikformer.eval()
    spikformer.train_mode = 'testing'
    criterion = nn.MSELoss(reduction='none')
    functional.reset_net(spikformer)

    # if hasattr(spikformer, 'weak_decoder'): delattr(spikformer, 'weak_decoder')
    # if hasattr(spikformer, 'replay') : delattr(spikformer, 'replay')
    
    attens_energy = []
    # (1) static on the train set
    with torch.no_grad():
        epoch_loss = 0

        preds = []
        trues = []
        for batch in train_loader:
            x, _ = batch
            x = x.float().to(args.device)
                
            output = spikformer(x)
            if isinstance(output, tuple):
                output = output[0]

            score = criterion(x, output).mean(-1)
            attens_energy.append(score.detach().cpu().numpy())
            
            functional.reset_net(spikformer)
            
    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    train_energy = np.array(attens_energy)
        
    attens_energy = []
    test_labels = []
    # (2) find the threshold on the test set
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.float().to(args.device)
            y = y.float().to(args.device)
                
            output = spikformer(x)
            if isinstance(output, tuple):
                output = output[0]

            score = criterion(x, output).mean(-1)
            attens_energy.append(score.detach().cpu().numpy())
            test_labels.append(y.detach().cpu().numpy())
            
            functional.reset_net(spikformer)
            
    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    test_energy = np.array(attens_energy)
    combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    threshold = np.percentile(combined_energy, 100 - args.anomaly_ratio)
    print("Threshold :", threshold)
    
    # (3) evaluation on the test set
    
    pred = (test_energy > threshold).astype(int)
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    test_labels = np.array(test_labels)
    gt = test_labels.astype(int)
    
    print("pred:", pred.shape, " gt:", gt.shape)
    
    # (4) detection adjustment
    
    gt, pred = adjustment(gt, pred)
    
    pred = np.array(pred)
    gt = np.array(gt)
    
    print("pred:", pred.shape, " gt:", gt.shape)
    
    accuracy = accuracy_score(gt, pred)
    
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
    
            
    test_result = {
        'accuracy' : accuracy,
        'precision' : precision,
        'recall' : recall,
        'f-score' : f_score,
    }
    
    print("Test was successfully done")

    with open(args.save_log_path + '/final+result.csv', 'a', encoding='utf-8') as log_csv:
        print("accuracy", "precision", "recall", "f-score",
              sep=", ", end="\n", file=log_csv)
        
        print(f"{test_result['accuracy']:.6f}", 
              f"{test_result['precision']:.6f}", 
              f"{test_result['recall']:.6f}",
              f"{test_result['f-score']:.6f}",
              sep=", ", end="\n", file=log_csv)
        
    print(f"Final result saved to `{args.save_log_path}`")
    
    visualize_anomaly_example(args, spikformer, threshold, test_loader, criterion)
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
    
    