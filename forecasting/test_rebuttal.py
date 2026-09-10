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

import os
import csv
from collections import defaultdict

import torch
import torch.nn.functional as F

from noise_injector import apply_gaussian_noise
from spikingjelly.activation_based import functional

# ----------------------------------------------------------
# 가장 좋은 방법:
# training 때 쓰던 "exact low-frequency target 생성 함수"를 그대로 import
# 예) from utils.freq_utils import build_low_freq_target
# ----------------------------------------------------------
try:
    from utils.freq_utils import build_low_freq_target  # TODO: 실제 경로로 수정
except Exception:
    build_low_freq_target = None

# fallback: 분석용 DCT 구현
# 가능하면 위의 exact 함수를 재사용하는 것을 권장
try:
    from scipy.fft import dct, idct
except Exception:
    from scipy.fftpack import dct, idct


def build_low_freq_target_from_clean(x: torch.Tensor, k: float) -> torch.Tensor:
    """
    clean input x: [B, L, C]
    low-frequency target 생성
    가능한 한 training 때 사용한 함수와 동일해야 함.
    """
    if build_low_freq_target is not None:
        return build_low_freq_target(x, k=k)

    x_np = x.detach().cpu().numpy()  # [B, L, C]
    coeff = dct(x_np, type=2, axis=1, norm='ortho')

    L = x_np.shape[1]
    keep = max(1, int(round(L * k)))
    coeff[:, keep:, :] = 0.0

    low = idct(coeff, type=2, axis=1, norm='ortho')
    return torch.from_numpy(low).to(x.device, dtype=x.dtype)


def get_submodule_by_name(model, module_name: str):
    """
    "neocortex" / "decoder" / "module.submodule" 같은 문자열로 module 접근
    """
    m = model
    for part in module_name.split('.'):
        m = getattr(m, part)
    return m


class NeocortexProbe:
    """
    model forward를 안 바꾸고, Neocortex 출력을 hook으로 잡기 위한 helper
    """
    def __init__(self, model, neo_module_name: str):
        self.cache = {}
        self.handle = None

        neo_module = get_submodule_by_name(model, neo_module_name)
        self.handle = neo_module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        self.cache["x_neo"] = output.detach()

    def clear(self):
        self.cache.clear()

    def close(self):
        if self.handle is not None:
            self.handle.remove()


def batch_pearson_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    """
    a, b: [B, ...]
    sample-wise Pearson correlation
    """
    a = a.flatten(1)
    b = b.flatten(1)

    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)

    num = (a * b).sum(dim=1)
    den = torch.sqrt((a.pow(2).sum(dim=1) + eps) * (b.pow(2).sum(dim=1) + eps))
    return num / den


def align_recon_to_target_shape(recon: torch.Tensor, target: torch.Tensor, model=None):
    """
    decoder 출력 shape과 target shape이 다를 때 맞춰주는 곳
    가장 중요:
    training 시 aux loss 계산 전에 사용하던 reshape / unpatchify와 동일하게 맞춰야 함.
    """
    if recon.shape == target.shape:
        return recon

    # TODO:
    # 예를 들어 decoder 출력이 [B, N, P*C] 이고 target이 [B, L, C] 라면
    # 여기서 model.unpatchify(recon) 또는 기존 training 코드의 reshape를 재사용해야 함.
    #
    # 예시:
    # if hasattr(model, "unpatchify"):
    #     recon = model.unpatchify(recon)
    #     if recon.shape == target.shape:
    #         return recon

    raise ValueError(
        f"Recon shape {recon.shape} != target shape {target.shape}. "
        "Use the same reshape/unpatchify logic as the training aux loss."
    )


def update_repr_stats(stats, prefix: str, recon: torch.Tensor, target: torch.Tensor):
    """
    prefix: 'clean' or 'noisy'
    """
    stats[f"{prefix}_mse_sum"] += F.mse_loss(recon, target, reduction="sum").item()
    stats[f"{prefix}_mae_sum"] += F.l1_loss(recon, target, reduction="sum").item()

    recon_f = recon.flatten(1)
    target_f = target.flatten(1)

    stats[f"{prefix}_cos_sum"] += F.cosine_similarity(recon_f, target_f, dim=1).sum().item()
    stats[f"{prefix}_pearson_sum"] += batch_pearson_corr(recon, target).sum().item()

    stats[f"{prefix}_numel"] += target.numel()
    stats[f"{prefix}_nsamples"] += target.shape[0]


def append_repr_csv(save_path: str, row: dict):
    file_exists = os.path.exists(save_path)
    with open(save_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)



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

    enable_repr_analysis = getattr(args, "analysis_repr", False)

    probe = None
    recon_decoder = None
    repr_stats = defaultdict(float)

    if enable_repr_analysis:
        # ------------------------------------------------------
        # TODO: 아래 두 이름은 네 실제 model 구조에 맞게 바꿔야 함
        # 예시:
        #   args.neo_hook_module = "neocortex"
        #   args.recon_decoder_module = "decoder"
        # ------------------------------------------------------
        probe = NeocortexProbe(model, args.neo_hook_module)
        recon_decoder = get_submodule_by_name(model, args.recon_decoder_module)


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
            clean_x = x.clone()

            # -----------------------------------------
            # noisy input 생성
            # -----------------------------------------
            if hasattr(args, "snr") and args.snr is not None:
                noisy_x = apply_gaussian_noise(clean_x, args.snr)
            else:
                noisy_x = clean_x

            # ======================================================
            # [A] Representation-level analysis
            # clean target vs noisy recon
            # ======================================================
            if enable_repr_analysis:
                # 1) clean low-frequency target
                lowfreq_clean = build_low_freq_target_from_clean(
                    clean_x,
                    k=getattr(args, "retention_ratio", 0.15)
                )

                # -----------------------------
                # 2) clean pass
                # -----------------------------
                probe.clear()
                functional.reset_net(model)
                _ = model(clean_x)

                if "x_neo" not in probe.cache:
                    raise RuntimeError(
                        "Failed to capture Neocortex output. "
                        "Check args.neo_hook_module."
                    )

                xneo_clean = probe.cache["x_neo"]
                recon_clean = recon_decoder(xneo_clean)
                recon_clean = align_recon_to_target_shape(recon_clean, lowfreq_clean, model=model)

                update_repr_stats(repr_stats, prefix="clean", recon=recon_clean, target=lowfreq_clean)

                # -----------------------------
                # 3) noisy pass
                # -----------------------------
                probe.clear()
                functional.reset_net(model)
                output = model(noisy_x)

                if "x_neo" not in probe.cache:
                    raise RuntimeError(
                        "Failed to capture Neocortex output on noisy input. "
                        "Check args.neo_hook_module."
                    )

                xneo_noisy = probe.cache["x_neo"]
                recon_noisy = recon_decoder(xneo_noisy)
                recon_noisy = align_recon_to_target_shape(recon_noisy, lowfreq_clean, model=model)

                update_repr_stats(repr_stats, prefix="noisy", recon=recon_noisy, target=lowfreq_clean)

                # clean/noisy X_neo similarity
                repr_stats["xneo_cos_sum"] += F.cosine_similarity(
                    xneo_clean.flatten(1),
                    xneo_noisy.flatten(1),
                    dim=1
                ).sum().item()
                repr_stats["xneo_nsamples"] += clean_x.shape[0]

            else:
                # ==================================================
                # [B] 기존 test 동작
                # ==================================================
                functional.reset_net(model)
                output = model(noisy_x)

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
        
    # ==========================================================
    # [추가] representation analysis 결과 정리
    # ==========================================================
    repr_result = None
    if enable_repr_analysis:
        repr_result = {
            "snr": getattr(args, "snr", None),

            "clean_recon_mse": repr_stats["clean_mse_sum"] / max(repr_stats["clean_numel"], 1.0),
            "clean_recon_mae": repr_stats["clean_mae_sum"] / max(repr_stats["clean_numel"], 1.0),
            "clean_recon_cos": repr_stats["clean_cos_sum"] / max(repr_stats["clean_nsamples"], 1.0),
            "clean_recon_pearson": repr_stats["clean_pearson_sum"] / max(repr_stats["clean_nsamples"], 1.0),

            "noisy_recon_mse_to_clean_target": repr_stats["noisy_mse_sum"] / max(repr_stats["noisy_numel"], 1.0),
            "noisy_recon_mae_to_clean_target": repr_stats["noisy_mae_sum"] / max(repr_stats["noisy_numel"], 1.0),
            "noisy_recon_cos_to_clean_target": repr_stats["noisy_cos_sum"] / max(repr_stats["noisy_nsamples"], 1.0),
            "noisy_recon_pearson_to_clean_target": repr_stats["noisy_pearson_sum"] / max(repr_stats["noisy_nsamples"], 1.0),

            "xneo_clean_noisy_cos": repr_stats["xneo_cos_sum"] / max(repr_stats["xneo_nsamples"], 1.0),
        }

        print(
            f"[Repr Analysis | SNR={repr_result['snr']}] "
            f"clean_MSE={repr_result['clean_recon_mse']:.6f}, "
            f"noisy_MSE_to_clean_target={repr_result['noisy_recon_mse_to_clean_target']:.6f}, "
            f"clean_COS={repr_result['clean_recon_cos']:.4f}, "
            f"noisy_COS_to_clean_target={repr_result['noisy_recon_cos_to_clean_target']:.4f}, "
            f"Xneo_clean_noisy_COS={repr_result['xneo_clean_noisy_cos']:.4f}"
        )

        if getattr(args, "repr_csv_path", None) is not None:
            append_repr_csv(args.repr_csv_path, repr_result)

        probe.close()

        
        
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
    
    
    