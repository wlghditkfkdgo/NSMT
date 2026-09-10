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
# import model as model
from ours import TemporalBlock
from data_provider.data_factory import data_provider
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
# ==============================
# Path effectiveness evaluation
# ==============================
import copy, json, math
import torch.nn.functional as F
import matplotlib.pyplot as plt

def load_model(args):
    saved_model_path = os.path.join(args.save_result_path, "model_state", f"best+model.pt")
    
    model = create_model(
        'mymodel',
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
        mlp_ratios=args.mlp_ratios,
        depths=args.num_layers, 
        sr_ratios=1,
        time_num_layers=args.time_num_layers,
        c_in=args.c_in,
        bias=args.bias, 
        tau=args.tau,
        spk_encoding=args.spk_encoding,
    )

    model = model.to(args.device)

    
    print(f"Model was successfully loaded. (epoch = {args.saved_epoch[-1]:03d})")
    
    return model


# utils
def safe_partial_load(model, ckpt_path: str, verbose: bool = True):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)

    model_state = model.state_dict()
    keep = {}
    unexpected = []
    missing = []

    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            keep[k] = v
        else:
            unexpected.append(k)

    model.load_state_dict(keep, strict=False)

    # 남은 키들(모델에 있는데 못 채운 것)
    for k in model_state.keys():
        if k not in keep:
            missing.append(k)

    if verbose:
        def _short(lst, n=12):
            return lst[:n] + (['...'] if len(lst) > n else [])
        print(f"[safe_load] loaded: {len(keep)} keys")
        if unexpected:
            print(f"[safe_load] ignored(unexpected): {len(unexpected)} keys -> {_short(unexpected)}")
        if missing:
            print(f"[safe_load] missing(not in ckpt): {len(missing)} keys -> {_short(missing)}")
            
class ScaleWrapper(nn.Module):
    """모듈 출력에 스칼라 α를 곱해 임시 개입"""
    def __init__(self, module, alpha=1.0):
        super().__init__()
        self.module = module
        self.alpha = alpha
    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        return out * self.alpha

def cosine_sim(a, b, eps=1e-8):
    a = a.flatten(1); b = b.flatten(1)
    cs = F.cosine_similarity(a, b, dim=1, eps=eps)  # L2 기반
    return cs.mean().item()

def rel_change(a, b, eps=1e-8):
    num = (a-b).pow(2).sum().sqrt()
    den = (b.pow(2).sum().sqrt() + eps)
    return (num/den).item()

@torch.no_grad()
def forward_training_tuple(model, x):
    """분석용: training 스타일 출력 (z, hip_avg, x_neo, org_x, rec_x)"""
    mode_backup = model.train_mode
    model.train_mode = 'training'
    functional.reset_net(model)
    out = model(x)
    model.train_mode = mode_backup
    return out

def replace_encoding_neo(model, alpha=None, permute_time=False):
    """encoding_neo를 스케일/퍼뮤트 개입으로 임시 대체"""
    # 원래 모듈 백업
    if not hasattr(model, "_enc_neo_backup"):
        model._enc_neo_backup = model.encoding_neo

    base = model._enc_neo_backup

    if alpha is None and not permute_time:
        model.encoding_neo = base
        return

    class Wrap(nn.Module):
        def __init__(self, inner, alpha=1.0, permute=False):
            super().__init__()
            self.inner = inner
            self.alpha = alpha
            self.permute = permute
        def forward(self, *args, **kwargs):
            x = self.inner(*args, **kwargs)  # [T, B, 1, D]
            if self.permute:
                T = x.shape[0]
                idx = torch.randperm(T, device=x.device)
                x = x.index_select(0, idx)
            if self.alpha is not None:
                x = x * self.alpha
            return x

    model.encoding_neo = Wrap(base, alpha=alpha, permute=permute_time)

def restore_encoding_neo(model):
    if hasattr(model, "_enc_neo_backup"):
        model.encoding_neo = model._enc_neo_backup

def collect_gate_energy(model):
    """각 Block의 mca 게이트 출력을 후크로 수집 (x * (1 - mca(...))) 에서 mca부분 크기)"""
    energies = []
    hooks = []
    def hook_fn(mod, inp, out):
        # mod는 MutualCrossAttention. out과 inp로부터 출력 크기만 사용
        # out: 게이트 값 (가정) → 평균 절대값
        val = out.detach()
        energies.append(val.float().abs().mean().item())

    for m in model.modules():
        if m.__class__.__name__ == 'MutualCrossAttention':
            hooks.append(m.register_forward_hook(lambda mod, i, o: hook_fn(mod, i, o)))
    return energies, hooks

# ---------- 주 평가 루틴 ----------
def neo_causal_eval(args, num_batches: int = 8):
    """
    여러 test 배치를 샘플링해서 평균적인 값을 리포트.
    num_batches: 최대 몇 개 배치까지 볼지 (loader가 더 작으면 거기까지만)
    """
    # 1) 데이터/모델
    _, loader = data_provider(args, flag='test')
    
    spikformer = load_model(args)
    spikformer = spikformer.to(args.device)
    spikformer.eval()

    # 2) 안전 로드 (한 번만)
    ckpt_path = os.path.join(args.save_result_path, "model_state", "best+model.pt")
    safe_partial_load(spikformer, ckpt_path, verbose=True)

    # alpha 설정 (모든 배치에서 동일하게 사용)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    # ----- 누적 버퍼 초기화 -----
    count = 0
    baseline_mse_sum = 0.0

    # alpha별로 cos_sim/rel_change/mse 누적
    alpha_acc = {
        a: {"cos_sim": 0.0, "rel_change": 0.0, "mse": 0.0}
        for a in alphas
    }

    # permute case 누적
    perm_acc = {"cos_sim": 0.0, "rel_change": 0.0, "mse": 0.0}

    # grad norm / share 누적
    hip_gn_sum = 0.0
    neo_gn_sum = 0.0
    neo_share_sum = 0.0

    # gate energy 누적
    gate_energy_sum = 0.0
    gate_energy_count = 0

    # ---------- 배치 루프 ----------
    for b_idx, batch in enumerate(loader):
        if b_idx >= num_batches:
            break

        x, y, _, _ = batch
        x = x.float().to(args.device)
        y = y.float().to(args.device)

        # 2) 베이스라인
        functional.reset_net(spikformer)
        z_base, hip_base, neo_base, *_ = forward_training_tuple(spikformer, x)
        zb = z_base.mean(0) if z_base.dim() == 4 else z_base
        mse_base = F.mse_loss(zb[:, -args.pred_len:, :], y[:, -args.pred_len:, :]).item()
        baseline_mse_sum += mse_base

        # 3) α-개입 곡선
        for a in alphas:
            replace_encoding_neo(spikformer, alpha=a, permute_time=False)
            functional.reset_net(spikformer)
            z, *_ = forward_training_tuple(spikformer, x)
            z = z.mean(0) if z.dim() == 4 else z
            cs = cosine_sim(z, zb)
            rc = rel_change(z, zb)
            mse = F.mse_loss(z[:, -args.pred_len:, :], y[:, -args.pred_len:, :]).item()

            alpha_acc[a]["cos_sim"]   += cs
            alpha_acc[a]["rel_change"] += rc
            alpha_acc[a]["mse"]       += mse
        restore_encoding_neo(spikformer)

        # 4) 시간축 퍼뮤트(정렬 파괴)
        replace_encoding_neo(spikformer, alpha=1.0, permute_time=True)
        functional.reset_net(spikformer)
        z_perm, *_ = forward_training_tuple(spikformer, x)
        z_perm = z_perm.mean(0) if z_perm.dim() == 4 else z_perm
        cos_perm = cosine_sim(z_perm, zb)
        rc_perm  = rel_change(z_perm, zb)
        mse_perm = F.mse_loss(z_perm[:, -args.pred_len:, :], y[:, -args.pred_len:, :]).item()
        restore_encoding_neo(spikformer)

        perm_acc["cos_sim"]    += cos_perm
        perm_acc["rel_change"] += rc_perm
        perm_acc["mse"]        += mse_perm

        # 5) 그래디언트 기여도 (Neo vs Hip)
        torch.set_grad_enabled(True)
        spikformer.eval()

        hip_buf = {}
        neo_buf = {}

        def _save_hip(mod, inp, out):
            hip_buf['act'] = out

        def _save_neo(mod, inp, out):
            neo_buf['act'] = out

        h1 = spikformer.data_block[-1].register_forward_hook(_save_hip)
        h2 = spikformer.time_block.register_forward_hook(_save_neo)

        functional.reset_net(spikformer)
        z_out = spikformer(x)
        if isinstance(z_out, tuple):
            z = z_out[0]
        else:
            z = z_out
        if z.dim() == 4:
            z = z.mean(0)

        loss = F.mse_loss(z[:, -args.pred_len:, :], y[:, -args.pred_len:, :])

        hip_act = hip_buf['act']
        neo_act = neo_buf['act']

        if isinstance(neo_act, tuple): neo_act = neo_act[1]
        grads = torch.autograd.grad(loss, [hip_act, neo_act],
                                    retain_graph=False, allow_unused=True)

        g_hip = grads[0] if grads[0] is not None else torch.zeros(1, device=z.device)
        g_neo = grads[1] if grads[1] is not None else torch.zeros(1, device=z.device)

        hip_gn = g_hip.pow(2).mean().sqrt().item()
        neo_gn = g_neo.pow(2).mean().sqrt().item()
        neo_grad_share = neo_gn / (neo_gn + hip_gn + 1e-12)

        hip_gn_sum += hip_gn
        neo_gn_sum += neo_gn
        neo_share_sum += neo_grad_share

        h1.remove(); h2.remove()

        # 6) 게이트-영향 상관(선택): 훅으로 mca 출력 모으기
        gate_vals, hooks = collect_gate_energy(spikformer)
        functional.reset_net(spikformer)
        _ = forward_training_tuple(spikformer, x)
        for h in hooks: h.remove()
        if gate_vals:
            gate_energy_sum += float(torch.tensor(gate_vals).mean())
            gate_energy_count += 1

        count += 1  # 실제 사용한 배치 수

    if count == 0:
        raise RuntimeError("neo_causal_eval: test loader가 비어 있습니다.")

    # ----- 평균 내기 -----
    baseline_mse = baseline_mse_sum / count

    alpha_results = []
    for a in alphas:
        alpha_results.append({
            "alpha": a,
            "cos_sim":   alpha_acc[a]["cos_sim"]   / count,
            "rel_change": alpha_acc[a]["rel_change"] / count,
            "mse":       alpha_acc[a]["mse"]       / count,
        })

    permute_report = {
        "cos_sim":    perm_acc["cos_sim"]    / count,
        "rel_change": perm_acc["rel_change"] / count,
        "mse":        perm_acc["mse"]        / count,
    }

    hip_gn_mean = hip_gn_sum / count
    neo_gn_mean = neo_gn_sum / count
    neo_grad_share_mean = neo_share_sum / count

    gate_energy = (gate_energy_sum / gate_energy_count) if gate_energy_count > 0 else None

    grad_report = {
        "hip_norm": hip_gn_mean,
        "neo_norm": neo_gn_mean,
        "neo_grad_share": neo_grad_share_mean,
    }
    print("[grad-share(avg)]", grad_report)

    report = {
        "baseline_mse": baseline_mse,
        "alpha_curve": alpha_results,
        "permute": permute_report,
        "grad": grad_report,
        "gate_energy_mean": gate_energy,
        "num_batches": count,
    }
    print(json.dumps(report, indent=2))

    os.makedirs(args.save_log_path, exist_ok=True)
    with open(os.path.join(args.save_log_path, "neo_causal_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return report

# --- DCT utils (시간축 L에 대해 DCT/IDCT) ---

# ---------------- DCT / filtering utils ----------------
def _dct_matrix(T, device=None, dtype=torch.float32):
    device = device or 'cpu'
    n = torch.arange(T, device=device, dtype=dtype).reshape(1, T)
    k = torch.arange(T, device=device, dtype=dtype).reshape(T, 1)
    C = torch.cos(math.pi * (n + 0.5) * k / T) * math.sqrt(2.0 / T)
    C[0, :] *= 1 / math.sqrt(2.0)
    return C

def _dct_time(x, C, time_dim=1):
    if time_dim != 1:
        perm = list(range(x.dim())); perm[1], perm[time_dim] = perm[time_dim], perm[1]
        x = x.permute(*perm).contiguous()
        need_inv = (perm != list(range(len(perm))))
    else:
        need_inv = False
    Y = torch.matmul(x.transpose(1,2).contiguous(), C.t()).transpose(1,2)
    if need_inv:
        inv = list(range(len(perm))); inv[1], inv[time_dim] = inv[time_dim], inv[1]
        Y = Y.permute(*inv).contiguous()
    return Y

def _idct_time(xc, C, time_dim=1):
    if time_dim != 1:
        perm = list(range(xc.dim())); perm[1], perm[time_dim] = perm[time_dim], perm[1]
        xc = xc.permute(*perm).contiguous()
        need_inv = (perm != list(range(len(perm))))
    else:
        need_inv = False
    Y = torch.matmul(xc.transpose(1,2).contiguous(), C).transpose(1,2)
    if need_inv:
        inv = list(range(len(perm))); inv[1], inv[time_dim] = inv[time_dim], inv[1]
        Y = Y.permute(*inv).contiguous()
    return Y

def _lp_hp_split(x, keep_ratio=0.15, time_dim=1):
    """x:[B,L,C] -> (x_lp, x_hp) by DCT masking"""
    B,L,Cn = x.shape
    Cmt = _dct_matrix(L, device=x.device, dtype=x.dtype)
    Xc = _dct_time(x, Cmt, time_dim=time_dim)
    k = max(1, int(L * keep_ratio))
    m_lp = torch.zeros_like(Xc); m_lp[:, :k, :] = 1.
    m_hp = torch.zeros_like(Xc); m_hp[:, k:, :] = 1.
    x_lp = _idct_time(Xc * m_lp, Cmt, time_dim=time_dim)
    x_hp = _idct_time(Xc * m_hp, Cmt, time_dim=time_dim)
    return x_lp, x_hp, k

def _smooth_1d(x_1d, kernel=5):
    T = x_1d.shape[0]
    if T <= 2: return x_1d
    k = min(kernel, T | 1)
    pad = k // 2
    filt = torch.ones(1,1,k, device=x_1d.device, dtype=x_1d.dtype) / k
    y = F.conv1d(x_1d.view(1,1,T), filt, padding=pad)
    return y.view(T)
# --- DCT (시간축) ---
def _dct_matrix(T, device=None, dtype=torch.float32):
    device = device or 'cpu'
    n = torch.arange(T, device=device, dtype=dtype).reshape(1, T)
    k = torch.arange(T, device=device, dtype=dtype).reshape(T, 1)
    C = torch.cos(math.pi * (n + 0.5) * k / T) * math.sqrt(2.0 / T)
    C[0, :] *= 1 / math.sqrt(2.0)
    return C

def _dct_time(x, C, time_dim=1):
    if time_dim != 1:
        perm = list(range(x.dim())); perm[1], perm[time_dim] = perm[time_dim], perm[1]
        x = x.permute(*perm).contiguous()
        need_inv = (perm != list(range(len(perm))))
    else:
        need_inv = False
    Y = torch.matmul(x.transpose(1,2).contiguous(), C.t()).transpose(1,2)
    if need_inv:
        inv = list(range(len(perm))); inv[1], inv[time_dim] = inv[time_dim], inv[1]
        Y = Y.permute(*inv).contiguous()
    return Y

# --- 모델과 동일한 패치화 (토큰화) ---
def _tokenize_sequence(x, patch_size, stride):
    """
    x: [B, L, C]
    return:
      patches: [T, B, C, P]  (P=patch_size)
      T = 1 + floor((L - P)/stride)
    """
    B, L, Cn = x.shape
    # unfold over L -> [B, C, T, P]
    patches = x.transpose(1,2).unfold(dimension=2, size=patch_size, step=stride)  # [B, C, T, P]
    T = patches.shape[2]
    patches = patches.permute(2, 0, 1, 3).contiguous()  # [T, B, C, P]
    return patches, T

def _token_lp_hp_energy(patches, keep_ratio=0.15):
    """
    patches: [T, B, C, P]
    return:
      lp_e: [T]  # per-token LP energy
      hp_e: [T]  # per-token HP energy
    """
    T, B, Cn, P = patches.shape
    Cmt = _dct_matrix(P, device=patches.device, dtype=patches.dtype)
    # DCT over patch-axis P
    Xc = _dct_time(patches.view(T*B*Cn, P).view(T*B, Cn, P), Cmt, time_dim=2)  # [T*B, C, P]
    Xc = Xc.view(T, B, Cn, P)

    k = max(1, int(P * keep_ratio))
    m_lp = torch.zeros_like(Xc); m_lp[..., :k] = 1.
    m_hp = torch.zeros_like(Xc); m_hp[..., k:] = 1.

    lp = (Xc * m_lp).pow(2).sum(dim=(1,2,3)) / (B*Cn*P)  # [T]
    hp = (Xc * m_hp).pow(2).sum(dim=(1,2,3)) / (B*Cn*P)  # [T]
    return lp, hp  # [T], [T]

# --------------- hooks & helpers ---------------
def _register_neo_prespike_hook(model):
    box = {'pre': None}
    def hook_fn(module, inp, out):
        box['pre'] = inp[0].detach()  # [T, B, ..., D]
    h = None
    if hasattr(model, 'encoding_neo') and hasattr(model.encoding_neo, 'linear'):
        h = model.encoding_neo.linear.register_forward_hook(hook_fn)
    elif hasattr(model, 'encoding_neo'):
        h = model.encoding_neo.register_forward_hook(hook_fn)
    return box, h

def _run_model_paths(spikformer, x):
    """returns: (z, hippo_avg[T,...], neo_pre[T,...])"""
    functional.reset_net(spikformer)
    out = spikformer(x)
    if isinstance(out, (list, tuple)):
        z, hip, neo, *_ = out
        hippo = hip
    else:
        z, hippo = out, None
    # neo pre-spike
    box, h = _register_neo_prespike_hook(spikformer)
    _ = spikformer(x)  # forward once more to capture pre
    neo_pre = box['pre'];  h.remove() if h is not None else None
    return z, hippo, neo_pre

def _token_power_from_any(x: torch.Tensor, T: int) -> torch.Tensor:
    """
    x: 임의 모양의 텐서 (토큰축 T가 맨 앞이거나 섞여 있어도 OK)
    반환: [T] (토큰별 평균 파워)
    """
    if x is None:
        return None
    x = x.float()
    # 전체 요소 수가 T로 나누어떨어지지 않으면 오류 방지
    if x.numel() % T != 0:
        # 마지막 축이 토큰일 가능성 대비: 전치 후 시도
        x = x.transpose(0, -1).contiguous()
        if x.numel() % T != 0:
            raise RuntimeError(f"[token_power] Can't reshape to [T=-1,*], got shape {tuple(x.shape)} with T={T}")
    # [T, -1] 로 강제 reshape 후 토큰별 파워
    x = x.reshape(T, -1)
    return (x**2).mean(dim=1)  # [T]

def _register_neo_prespike_hook(model):
    box = {'pre': None}
    def hook_fn(module, inp, out):
        box['pre'] = inp[0].detach()  # pre-spike current into spk_neuron
    h = None
    if hasattr(model, 'encoding_neo'):
        # SpikLinearLayer 내부에 bn, spk_neuron이 있으면 spk_neuron 입력을 훅
        if hasattr(model.encoding_neo, 'spk_neuron') and hasattr(model.encoding_neo, 'bn'):
            # bn의 forward output이 곧 spk_neuron input
            def _bn_hook(mod, inp, out):
                box['pre'] = out.detach()
            h = model.encoding_neo.bn.register_forward_hook(_bn_hook)
        else:
            h = model.encoding_neo.register_forward_hook(hook_fn)
    return box, h

def _run_model_paths(spikformer, x):
    """returns: (z, hippo_avg[T,...], neo_pre[T,...])"""
    functional.reset_net(spikformer)
    out = spikformer(x)
    if isinstance(out, (list, tuple)):
        z, hip, neo, *_ = out
        hippo = hip
    else:
        z, hippo = out, None

    # neo pre-spike capture
    box, h = _register_neo_prespike_hook(spikformer)
    _ = spikformer(x)  # run once more to fill hook
    neo_pre = box['pre']  # expect [T, B, 1, D] or [T, B, D]
    if h is not None:
        h.remove()
    return z, hippo, neo_pre


# ----------------- 1) cutoff sweep -----------------
@torch.no_grad()
def analyze_cutoff_sweep(args, model, save_dir):
    """
    keep_ratio를 변화시키며 LP 입력을 실제 모델에 넣어
    hip/neo의 path power 변화를 측정
    """
    import torch.nn.functional as F
    os.makedirs(save_dir, exist_ok=True)
    _, loader = data_provider(args, flag='test')
    x, y, _, _ = next(iter(loader))
    x = x.float().to(args.device)

    keep_list = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40]
    hip_curve, neo_curve = [], []

    model.eval()

    for kr in keep_list:
        # ---- 실제 LP 필터링 입력 ----
        x_lp, x_hp, _ = _lp_hp_split(x, keep_ratio=kr)
        
        # forward on LP-filtered input
        z, hippo, neo_pre = _run_model_paths(model, x_lp)
        
        # 경로별 평균 파워
        hip_pow = float(_token_power_from_any(hippo, model.T).mean().cpu()) if hippo is not None else 0.0
        neo_pow = float(_token_power_from_any(neo_pre, model.T).mean().cpu()) if neo_pre is not None else 0.0
        hip_curve.append(hip_pow)
        neo_curve.append(neo_pow)

    # ---- 정규화 및 저장 ----
    H = np.array(hip_curve)
    N = np.array(neo_curve)
    if H.max() > 0:
        Hn = H / (H.max() + 1e-12)
    else:
        Hn = H
    if N.max() > 0:
        Nn = N / (N.max() + 1e-12)
    else:
        Nn = N

    with open(os.path.join(save_dir, "cutoff_sweep.json"), "w") as f:
        json.dump({
            "keep_ratio": keep_list,
            "hip_norm": Hn.tolist(),
            "neo_norm": Nn.tolist(),
            "hip_raw": H.tolist(),
            "neo_raw": N.tolist()
        }, f, indent=2)

    plt.figure()
    plt.plot(keep_list, Hn, "-o", label="hip (norm)")
    plt.plot(keep_list, Nn, "-o", label="neo (norm)")
    plt.xlabel("low-pass keep_ratio")
    plt.ylabel("normalized path power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cutoff_sweep.png"))
    plt.close()
    print("[cutoff_sweep] Done: LP-filtered inputs actually fed to model.")


# ----------------- 2) coherence -----------------
@torch.no_grad()
def analyze_spectral_coherence(args, model, save_dir):
    """
    토큰 축(T)에서 얻은 경로 파워(hip/neo)와
    원 시퀀스 축(L)에서 얻은 LP/HP 입력 파워의 코히어런스를 계산.
    길이 불일치(T != L)를 선형보간으로 해결.
    """
    import torch.nn.functional as F
    os.makedirs(save_dir, exist_ok=True)
    _, loader = data_provider(args, flag='test')
    x, y, _, _ = next(iter(loader))
    x = x.float().to(args.device)

    model.eval()
    z, hippo, neo_pre = _run_model_paths(model, x)

    # --- 경로별 파워: [T] ---
    hip_pw_T = _token_power_from_any(hippo, model.T)    # [T] or None
    neo_pw_T = _token_power_from_any(neo_pre, model.T)  # [T] or None
    if hip_pw_T is None or neo_pw_T is None:
        print("[Coherence] skip: path tensors are None")
        return

    # --- 입력 LP/HP 파워: [L] ---
    L = x.shape[1]
    Cmt = _dct_matrix(L, device=x.device, dtype=x.dtype)
    Xc = _dct_time(x, Cmt, time_dim=1)
    k = int(0.15 * L)
    m_lp = torch.zeros_like(Xc); m_lp[:, :k, :] = 1.
    m_hp = torch.zeros_like(Xc); m_hp[:, k:, :] = 1.
    x_lp = _idct_time(Xc*m_lp, Cmt, time_dim=1)
    x_hp = _idct_time(Xc*m_hp, Cmt, time_dim=1)
    lp_pw_L = (x_lp**2).mean(dim=(0,2)).detach()  # [L]
    hp_pw_L = (x_hp**2).mean(dim=(0,2)).detach()  # [L]

    # --- T → L 리샘플 (선형 보간) ---
    def _resample_to_L(vec_T, L):
        # vec_T: [T] → [L]
        v = vec_T.view(1, 1, -1).float()
        vL = F.interpolate(v, size=L, mode='linear', align_corners=False)
        return vL.view(-1)

    hip_pw_L = _resample_to_L(hip_pw_T, L)  # [L]
    neo_pw_L = _resample_to_L(neo_pw_T, L)  # [L]

    # --- 정규화 상관 (cosine similarity on centered vectors) ---
    def _corr(a, b):
        a = a.float(); b = b.float()
        a = (a - a.mean()); b = (b - b.mean())
        denom = (a.norm() * b.norm() + 1e-12)
        return float((a*b).sum().cpu() / denom)

    coh_neo_lp = _corr(neo_pw_L, lp_pw_L)
    coh_hip_hp = _corr(hip_pw_L, hp_pw_L)

    res = {"coherence": {"neo_vs_lp": coh_neo_lp, "hip_vs_hp": coh_hip_hp},
           "lengths": {"T": int(hip_pw_T.numel()), "L": int(L)}}
    with open(os.path.join(save_dir, "coherence.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[Coherence] neo~LP={coh_neo_lp:.3f} | hip~HP={coh_hip_hp:.3f} | T→L resampled ({hip_pw_T.numel()}→{L})")
    
@torch.no_grad()
def token_coherence_mi_tau(args, model, save_dir, keep_ratio=0.15, bins=32, eps=0.01):
    os.makedirs(save_dir, exist_ok=True)
    _, loader = data_provider(args, flag='test')
    x, y, _, _ = next(iter(loader))
    x = x.float().to(args.device)

    model.eval()
    # 토큰화(모델과 동일)
    patches, T = _tokenize_sequence(x, patch_size=args.patch_size, stride=args.patch_size//2)
    # 입력의 토큰별 LP/HP 에너지
    lp_tok, hp_tok = _token_lp_hp_energy(patches, keep_ratio=keep_ratio)  # [T], [T]

    # 경로 관측
    z, hippo, neo_pre = _run_model_paths(model, x)
    hip_pw = _token_power_from_any(hippo, T)      # [T]
    neo_pw = _token_power_from_any(neo_pre, T)    # [T]
    if hip_pw is None or neo_pw is None:
        print("[token] skip: path tensors are None")
        return

    # --- coherence (cosine on centered vectors) ---
    def _corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        denom = (a.norm()*b.norm() + 1e-12)
        return float((a*b).sum().cpu() / denom)

    coh_neo_lp = _corr(neo_pw, lp_tok)
    coh_hip_hp = _corr(hip_pw, hp_tok)

    # --- MI (histogram) on tokens ---
    def _to_np(z):
        z = z.detach().float().cpu().numpy()
        z = (z - z.mean()) / (z.std() + 1e-12)
        return z
    neo_np, hip_np = _to_np(neo_pw), _to_np(hip_pw)
    lp_np, hp_np   = _to_np(lp_tok), _to_np(hp_tok)

    def _mi(a, b, bins):
        H, _, _ = np.histogram2d(a, b, bins=bins)
        P = H / (H.sum() + 1e-12)
        Px = P.sum(1, keepdims=True); Py = P.sum(0, keepdims=True)
        I = 0.0
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                p = P[i, j]
                if p > 0:
                    I += p * (np.log(p + 1e-12) - np.log(Px[i, 0] + 1e-12) - np.log(Py[0, j] + 1e-12))
        return float(I)

    mi_neo_lp = _mi(neo_np, lp_np, bins)
    mi_hip_hp = _mi(hip_np, hp_np, bins)

    # --- 수치 DCT-occlusion (토큰 영역) ---
    # 각 토큰 패치의 DCT bin을 구간으로 나눠, 입력 패치를 ε만큼 증폭 → 경로 파워 변화
    T, B, Cn, P = patches.shape
    CmtP = _dct_matrix(P, device=x.device, dtype=x.dtype)
    Xc = _dct_time(patches.view(T*B, Cn, P), CmtP, time_dim=2).view(T, B, Cn, P)
    base_hip = float(hip_pw.mean().cpu()); base_neo = float(neo_pw.mean().cpu())
    edges = torch.linspace(0, P-1, steps=bins+1, device=x.device).long()
    xs, s_hip, s_neo = [], [], []
    for i in range(bins):
        lo, hi = edges[i].item(), edges[i+1].item()
        mask = torch.zeros_like(Xc); mask[..., lo:hi] = 1.0
        Xc_pert = Xc * (1.0 + eps * mask)
        patches_pert = torch.matmul(Xc_pert, _dct_matrix(P, device=x.device, dtype=x.dtype).t())  # IDCT
        # (간단히) 재조립하여 원래 x와 동일 차원으로 합치기보다는,
        # perturb된 패치를 시퀀스로 복원
        # 오버랩 평균 복원
        stride = args.patch_size//2
        L = x.shape[1]; recon = torch.zeros_like(x); weight = torch.zeros_like(x)
        for t in range(T):
            start = t*stride
            end = start + P
            if end > L: break
            seg = patches_pert[t].permute(0,2,1)  # [B, P, C]
            recon[:, start:end, :] += seg
            weight[:, start:end, :] += 1
        recon = recon / (weight + 1e-12)

        # 경로 파워 재측정
        z1, hip1, neo1 = _run_model_paths(model, recon)
        hip_new = float(_token_power_from_any(hip1, T).mean().cpu())
        neo_new = float(_token_power_from_any(neo1, T).mean().cpu())
        s_hip.append((hip_new - base_hip) / (eps + 1e-12))
        s_neo.append((neo_new - base_neo) / (eps + 1e-12))
        xs.append(0.5*(lo+hi))

    # --- τ 추정 (토큰 도메인 step) ---
    # 토큰 인덱스 t0 이후 패치 평균이 1인 시퀀스를 만들어 응답 적합
    def _fit_tau_from_curve(y):
        y = y - y.min()
        if y.max() > 0: y = y / (y.max() + 1e-12)
        t = torch.arange(y.numel(), device=y.device, dtype=y.dtype)
        eps = 1e-6
        mask = (1 - y) > eps
        if mask.sum() < 3: return float('nan')
        X = t[mask].unsqueeze(1).float()
        zlin = torch.log((1 - y[mask]).clamp_min(eps)).unsqueeze(1).float()
        tau = - ( (X.T @ X + 1e-6*torch.eye(1, device=y.device)) ).inverse() @ (X.T @ zlin)
        return float(tau.item())

    tau_hip = _fit_tau_from_curve(hip_pw)
    tau_neo = _fit_tau_from_curve(neo_pw)

    # --- 저장 ---
    out = {
        "T": int(T),
        "coherence": {"neo_vs_lp_T": coh_neo_lp, "hip_vs_hp_T": coh_hip_hp},
        "MI": {"neo_lp_T": mi_neo_lp, "hip_hp_T": mi_hip_hp},
        "tau_token": {"hip": tau_hip, "neo": tau_neo},
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "token_aligned_stats.json"), "w") as f:
        json.dump(out, f, indent=2)

    # occlusion 곡선 저장
    np.save(os.path.join(save_dir, "token_dct_occ_hip.npy"), np.array(s_hip))
    np.save(os.path.join(save_dir, "token_dct_occ_neo.npy"), np.array(s_neo))
    plt.figure(); plt.plot(xs, s_hip, label='hip (num)')
    plt.plot(xs, s_neo, label='neo-pre (num)')
    plt.xlabel("DCT bin center (low→high) in token"); plt.ylabel("Δpath power / ε")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "token_dct_occlusion.png")); plt.close()

    print(f"[token] coh neo~LP={coh_neo_lp:.3f} | hip~HP={coh_hip_hp:.3f} | MI neo={mi_neo_lp:.3f} hip={mi_hip_hp:.3f} | τ hip={tau_hip:.3f}, neo={tau_neo:.3f}")

# ----------------- 3) Mutual Information -----------------
@torch.no_grad()
def analyze_mutual_information(args, model, save_dir, bins=20):
    """
    입력 LP/HP 에너지와 경로 파워 간 상호정보량 추정.
    길이 불일치(T != L)는 선형 보간(F.interpolate)로 T→L 리샘플 후 MI 계산.
    """
    import numpy as np
    import torch.nn.functional as F
    os.makedirs(save_dir, exist_ok=True)

    # --- data ---
    _, loader = data_provider(args, flag='test')
    x, y, _, _ = next(iter(loader))
    x = x.float().to(args.device)

    # --- forward & path tensors ---
    model.eval()
    z, hippo, neo_pre = _run_model_paths(model, x)
    hip_pw_T = _token_power_from_any(hippo, model.T)     # [T] or None
    neo_pw_T = _token_power_from_any(neo_pre, model.T)   # [T] or None
    if hip_pw_T is None or neo_pw_T is None:
        print("[MI] skip: path tensors are None")
        return

    # --- input LP/HP power on L ---
    L = x.shape[1]
    Cmt = _dct_matrix(L, device=x.device, dtype=x.dtype)
    Xc = _dct_time(x, Cmt, time_dim=1)
    k = int(0.15 * L)
    m_lp = torch.zeros_like(Xc); m_lp[:, :k, :] = 1.
    m_hp = torch.zeros_like(Xc); m_hp[:, k:, :] = 1.
    x_lp = _idct_time(Xc*m_lp, Cmt, time_dim=1)
    x_hp = _idct_time(Xc*m_hp, Cmt, time_dim=1)
    lp_pw_L = (x_lp**2).mean(dim=(0,2)).detach()  # [L]
    hp_pw_L = (x_hp**2).mean(dim=(0,2)).detach()  # [L]

    # --- T -> L resample ---
    def _resample_to_L(vec_T, L):
        v = vec_T.view(1, 1, -1).float()
        vL = F.interpolate(v, size=L, mode='linear', align_corners=False)
        return vL.view(-1)

    hip_pw_L = _resample_to_L(hip_pw_T, L)  # [L]
    neo_pw_L = _resample_to_L(neo_pw_T, L)  # [L]

    # --- numpy arrays ---
    neo_np = neo_pw_L.detach().cpu().numpy()
    hip_np = hip_pw_L.detach().cpu().numpy()
    lp_np  = lp_pw_L.detach().cpu().numpy()
    hp_np  = hp_pw_L.detach().cpu().numpy()

    # --- standardize (수치 안정) ---
    def _standardize(a):
        a = a.astype(np.float64)
        m, s = a.mean(), a.std()
        return (a - m) / (s + 1e-12)
    neo_np = _standardize(neo_np)
    hip_np = _standardize(hip_np)
    lp_np  = _standardize(lp_np)
    hp_np  = _standardize(hp_np)

    # --- histogram-based MI ---
    def _mi(a, b, bins):
        # a,b: same length 1D
        H, xedges, yedges = np.histogram2d(a, b, bins=bins)
        P = H / (H.sum() + 1e-12)
        Px = P.sum(axis=1, keepdims=True)  # [bx,1]
        Py = P.sum(axis=0, keepdims=True)  # [1,by]
        valid = P > 0
        # 안전한 log
        logP  = np.log(P[valid])
        logPx = np.log(Px[np.where(P.sum(axis=1)>0)[0], 0][:, None] + 1e-12)  # broadcast-safe
        logPy = np.log(Py[0, np.where(P.sum(axis=0)>0)[0]][None, :] + 1e-12)
        # 선택된 인덱스에 맞춰 Px,Py 재정렬은 위 한 줄로 처리했으므로,
        # 아래는 간단화를 위해 P를 다시 사용
        # 직접 합치려면 mask 인덱스 매핑이 필요하지만, 수치적으로는 아래 단순식이 잘 동작
        I = 0.0
        Px_vec = Px.squeeze(-1)
        Py_vec = Py.squeeze(0)
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                p = P[i, j]
                if p > 0:
                    I += p * (np.log(p + 1e-12) - np.log(Px_vec[i] + 1e-12) - np.log(Py_vec[j] + 1e-12))
        return float(I)

    mi_neo_lp = _mi(neo_np, lp_np, bins)
    mi_hip_hp = _mi(hip_np, hp_np, bins)

    with open(os.path.join(save_dir, "mutual_info.json"), "w") as f:
        json.dump({"MI": {"neo_lp": mi_neo_lp, "hip_hp": mi_hip_hp},
                   "lengths": {"L": int(L)}}, f, indent=2)
    print(f"[MI] neo↔LP={mi_neo_lp:.4f} | hip↔HP={mi_hip_hp:.4f} | lengths matched to L={L}")
    
# ----------------- 4) Step response (τ estimate) -----------------
@torch.no_grad()
def analyze_step_response_tau(args, model, save_dir):
    """
    Heaviside step 입력 → 경로 파워의 시간 응답을 지수로 적합해 유효 τ 비교.
    기대: τ_neo >> τ_hip
    """
    os.makedirs(save_dir, exist_ok=True)
    L = args.seq_len; B=8; C=args.c_in
    device = args.device
    t = torch.arange(L, device=device).float()
    step = (t >= L//3).float()  # 한 지점 이후 1
    x = step.view(1, L, 1).repeat(B,1,C).float().to(device)

    model.eval()
    z, hippo, neo_pre = _run_model_paths(model, x)
    hip_pw = _token_power_from_any(hippo, model.T)  # [T]
    neo_pw = _token_power_from_any(neo_pre, model.T)

    def _fit_tau(y):
        # y: [T], 정규화 후 (1 - exp(-t/tau)) 형태를 최소제곱 적합
        y = y - y.min()
        if y.max() > 0: y = y / (y.max() + 1e-12)
        t = torch.arange(y.numel(), device=y.device, dtype=y.dtype)
        # 선형화: 1 - y = exp(-t/tau) → ln(1 - y + eps) = -t/tau
        eps = 1e-6
        mask = (1 - y) > eps
        if mask.sum() < 3: 
            return float('nan')
        X = t[mask].float().unsqueeze(1)  # [n,1]
        z = torch.log((1 - y[mask]).clamp_min(eps)).float().unsqueeze(1)  # [n,1]
        # z = -X / tau → tau = - X / z (LS)
        tau = - ( (X.T @ X + 1e-6*torch.eye(1, device=y.device)) ).inverse() @ (X.T @ z)
        return float(tau.item())

    tau_hip = _fit_tau(hip_pw)
    tau_neo = _fit_tau(neo_pw)

    with open(os.path.join(save_dir, "tau_estimate.json"), "w") as f:
        json.dump({"tau": {"hip": tau_hip, "neo": tau_neo}}, f, indent=2)
    print(f"[Tau] hip≈{tau_hip:.2f} | neo≈{tau_neo:.2f} (expect neo >> hip)")

# ----------------- 5) Numerical DCT occlusion (fast) -----------------
@torch.no_grad()
def analyze_dct_occlusion(args, model, save_dir, bins=64, eps=0.01):
    os.makedirs(save_dir, exist_ok=True)
    _, loader = data_provider(args, flag='test')
    x, y, _, _ = next(iter(loader))
    x = x.float().to(args.device)

    model.eval()
    # 기준 경로 파워
    z0, hip0, neo0 = _run_model_paths(model, x)
    base_hip = float(_token_power_from_any(hip0, model.T).mean().cpu()) if hip0 is not None else 0.0
    base_neo = float(_token_power_from_any(neo0, model.T).mean().cpu()) if neo0 is not None else 0.0

    # 구간화
    L = x.shape[1]
    edges = torch.linspace(0, L-1, steps=bins+1, device=x.device).long()
    xs, s_hip, s_neo = [], [], []

    Cmt = _dct_matrix(L, device=x.device, dtype=x.dtype)
    Xc = _dct_time(x, Cmt, time_dim=1)

    for i in range(bins):
        lo, hi = edges[i].item(), edges[i+1].item()
        mask = torch.zeros_like(Xc); mask[:, lo:hi, :] = 1.0
        x_pert = _idct_time(Xc * (1.0 + eps * mask), Cmt, time_dim=1)

        z1, hip1, neo1 = _run_model_paths(model, x_pert)
        hip_new = float(_token_power_from_any(hip1, model.T).mean().cpu()) if hip1 is not None else 0.0
        neo_new = float(_token_power_from_any(neo1, model.T).mean().cpu()) if neo1 is not None else 0.0

        s_hip.append( (hip_new - base_hip) / (eps + 1e-12) )
        s_neo.append( (neo_new - base_neo) / (eps + 1e-12) )
        xs.append(0.5*(lo+hi))

    np.save(os.path.join(save_dir, "dct_occ_hip.npy"), np.array(s_hip))
    np.save(os.path.join(save_dir, "dct_occ_neo.npy"), np.array(s_neo))

    plt.figure(); plt.plot(xs, s_hip, label='hip (num)')
    plt.plot(xs, s_neo, label='neo-pre (num)')
    plt.xlabel("DCT bin center (low→high)"); plt.ylabel("Δpath power / ε")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dct_occlusion.png")); plt.close()
    print("[DCT-Occ] saved curves.")
    
# ----------------- Orchestrator -----------------
def run_all_path_analyses(args, model=None):
    """
    test() 이후 호출 권장. model 인스턴스가 있으면 인자로 전달.
    출력 폴더: args.save_log_path + '/path_analysis'
    """
    save_dir = os.path.join(args.save_log_path, "path_analysis")
    os.makedirs(save_dir, exist_ok=True)

    spikformer = model if model is not None else load_model(args)
    spikformer = spikformer.to(args.device)

    analyze_cutoff_sweep(args, spikformer, save_dir)
    analyze_spectral_coherence(args, spikformer, save_dir)
    analyze_mutual_information(args, spikformer, save_dir)
    analyze_step_response_tau(args, spikformer, save_dir)
    analyze_dct_occlusion(args, spikformer, save_dir)
    print(f"[Done] Additional path-frequency analyses saved to: {save_dir}")
    
def run_token_aligned_path_analyses(args, model=None):
    save_dir = os.path.join(args.save_log_path, "path_analysis_token")
    os.makedirs(save_dir, exist_ok=True)
    spikformer = model if model is not None else load_model(args)
    spikformer = spikformer.to(args.device)
    token_coherence_mi_tau(args, spikformer, save_dir, keep_ratio=0.15, bins=32, eps=0.01)
    print(f"[Done] Token-aligned analyses saved to {save_dir}")


# ==============================
# SSA high-frequency amplification test
# ==============================

def _dct_N(x: torch.Tensor):
    """
    DCT along N-axis for hippo path.
    x: [T, B, N, D]  (N이 원래 시계열 time 축에 정렬된 토큰 축)
    return:
        Xc: [T, B, N, D]  (N축에 대해 DCT된 계수들)
        C:  [N, N]        (DCT 변환 행렬)
    """
    assert x.dim() == 4, f"expected [T,B,N,D], got {x.shape}"
    T, B, N, D = x.shape

    # DCT matrix on length N (not T!)
    C = _dct_matrix(N, device=x.device, dtype=x.dtype)  # [N,N]

    # reshape so that N이 마지막 차원인 2D 행렬로 만들어 한 번에 곱하기
    # x: [T,B,N,D] -> [T,B,D,N] -> [T*B*D, N]
    x_flat = x.permute(0, 1, 3, 2).contiguous().view(-1, N)      # [T*B*D, N]

    # DCT: (T*B*D, N) @ (N, N) -> (T*B*D, N)
    Xc_flat = x_flat @ C.t()

    # 원래 모양으로 되돌리기: [T*B*D, N] -> [T,B,D,N] -> [T,B,N,D]
    Xc = Xc_flat.view(T, B, D, N).permute(0, 1, 3, 2).contiguous()  # [T,B,N,D]
    return Xc, C


def _idct_N(Xc: torch.Tensor, C: torch.Tensor):
    """
    inverse DCT along N-axis.
    Xc: [T,B,N,D]
    C : [N,N]  (same matrix as in _dct_N)
    return:
        x: [T,B,N,D]
    """
    assert Xc.dim() == 4, f"expected [T,B,N,D], got {Xc.shape}"
    T, B, N, D = Xc.shape
    assert C.shape[0] == N, f"DCT matrix size {C.shape} doesn't match N={N}"

    # [T,B,N,D] -> [T,B,D,N] -> [T*B*D,N]
    Xc_flat = Xc.permute(0, 1, 3, 2).contiguous().view(-1, N)   # [T*B*D, N]

    # IDCT: (T*B*D, N) @ (N, N) -> (T*B*D, N)
    x_flat = Xc_flat @ C                                        # [T*B*D, N]

    # -> [T,B,D,N] -> [T,B,N,D]
    x = x_flat.view(T, B, D, N).permute(0, 1, 3, 2).contiguous()  # [T,B,N,D]
    return x


def _split_lp_hp_on_N(x: torch.Tensor, keep_ratio: float = 0.15):
    """
    hippo path input x: [T, B, N, D]
    N축을 '본래 시계열 시간축'으로 보고, 그 축에 대해 DCT를 적용한 뒤
    저주파/고주파 성분으로 나눈다.

    return:
        x_lp: [T,B,N,D]  (N축 저주파만 남긴 재구성 신호)
        x_hp: [T,B,N,D]  (N축 고주파만 남긴 재구성 신호)
    """
    assert x.dim() == 4, f"expected [T,B,N,D], got {x.shape}"
    T, B, N, D = x.shape

    # 1) DCT along N
    Xc, C = _dct_N(x)  # [T,B,N,D], [N,N]

    # 2) low/high mask in frequency index (0..N-1)
    k = max(1, int(N * keep_ratio))
    mask_lp = torch.zeros_like(Xc)
    mask_lp[:, :, :k, :] = 1.0           # 저주파 (낮은 index)
    mask_hp = 1.0 - mask_lp              # 나머지 = 고주파

    Xc_lp = Xc * mask_lp
    Xc_hp = Xc * mask_hp

    # 3) inverse DCT to go back to token domain
    x_lp = _idct_N(Xc_lp, C)             # [T,B,N,D]
    x_hp = _idct_N(Xc_hp, C)             # [T,B,N,D]

    return x_lp, x_hp


def _scalar_norm(t):
    if t is None:
        return 0.0
    return float(t.detach().float().pow(2).mean().sqrt().cpu())


def _cos_rel_mse(z, zref, y, pred_len):
    # z, zref: [T,B,L,C] or [B,L,C]
    if z.dim() == 4: z = z.mean(0)
    if zref.dim() == 4: zref = zref.mean(0)
    cs = cosine_sim(z, zref)
    rc = rel_change(z, zref)
    mse = F.mse_loss(z[:, -pred_len:, :], y[:, -pred_len:, :]).item()
    return cs, rc, mse


def analyze_ssa_highfreq(args, base_model=None, keep_ratio=0.15, save_key="ssa_hf"):
    """
    각 Block(SSA_rel_scl)에 대해 baseline/LP/HP 세 조건을 돌려
    - gate_mean: SSA gate 평균(abs mean)
    - response_gain: ||out|| / ||in||
    - 전체 출력의 변동(cs, rc, mse)
    를 기록.

    저장 위치: args.save_log_path / save_key / results.json
    """
    os.makedirs(os.path.join(args.save_log_path, save_key), exist_ok=True)
    save_dir = os.path.join(args.save_log_path, save_key)
    
    model = base_model if base_model is not None else load_model(args)
    model = model.to(args.device)

    # --- 데이터 한 배치 로드
    _, loader = data_provider(args, flag='test')
    x, y, _, _ = next(iter(loader))
    x = x.float().to(args.device)
    y = y.float().to(args.device)

    model.eval()

    # --- baseline 참조 출력
    functional.reset_net(model)
    z_base = model(x)
    if isinstance(z_base, (list, tuple)):
        z_base = z_base[0]

    # --- 유틸: 특정 Block에만 개입/계측하는 훅들 생성
    def make_attn_gate_hook(box):
        def _hook(mod, inp, out):
            # SSA_rel_scl의 출력(게이트)을 평균 절대값으로 집계
            box['gate_mean'] = float(out.detach().abs().mean().cpu())
        return _hook

    def make_pre_hook(box, mode, keep_ratio):
        """
        mode: 'none' | 'lp' | 'hp'
        Block.forward(x, mx=None) 형태라 pre-hook 입력은 (x, mx)
        첫 번째 인자 x만 LP/HP로 치환하고 mx는 그대로 통과시킨다.
        """
        def _pre(mod, args):
            # args는 tuple. (x,) 또는 (x, mx)
            if len(args) == 1:
                xin, = args
                mx = None
            elif len(args) == 2:
                xin, mx = args
            else:
                # 혹시라도 추가 인자가 생겨도 첫 번째만 x로 보고 나머지는 그대로 유지
                xin = args[0]
                mx  = args[1] if len(args) > 1 else None

            # 기록: 원본 입력 노름
            box['in_norm'] = _scalar_norm(xin)

            # baseline이면 그대로 통과
            if mode == 'none':
                return (xin,) if mx is None else (xin, mx)

            # N축(L 원시 시간축이 정렬된 축)을 따라 LP/HP 분해
            x_lp, x_hp = _split_lp_hp_on_N(xin, keep_ratio=keep_ratio)
            x_mod = x_lp if mode == 'lp' else x_hp

            # return signature는 원래 forward 인자 서명과 동일해야 함
            return (x_mod,) if mx is None else (x_mod, mx)
        return _pre

    def make_post_hook(box):
        def _post(mod, inp, out):
            box['out_norm'] = _scalar_norm(out)
        return _post

    # --- 실행 루틴 (block_idx, condition) => metrics
    def run_once(block_idx, condition):
        """
        condition: 'baseline' | 'lp' | 'hp'
        """
        blk = model.data_block[block_idx]  # Block: .attn == SSA_rel_scl

        boxes = {'pre': {}, 'attn': {}, 'post': {}}
        # hooks
        h_pre  = blk.register_forward_pre_hook(make_pre_hook(boxes['pre'],
                                                             mode=('none' if condition=='baseline' else condition),
                                                             keep_ratio=keep_ratio))
        h_attn = blk.attn.register_forward_hook(make_attn_gate_hook(boxes['attn']))
        h_post = blk.register_forward_hook(make_post_hook(boxes['post']))

        # forward
        functional.reset_net(model)
        z = model(x)
        if isinstance(z, (list, tuple)):
            z = z[0]

        # remove hooks
        h_pre.remove(); h_attn.remove(); h_post.remove()

        # compute deltas to baseline output
        cs, rc, mse = _cos_rel_mse(z, z_base, y, args.pred_len)

        return {
            "gate_mean": boxes['attn'].get('gate_mean', None),
            "in_norm": boxes['pre'].get('in_norm', None),
            "out_norm": boxes['post'].get('out_norm', None),
            "response_gain": (boxes['post'].get('out_norm', 0.0) / (boxes['pre'].get('in_norm', 1e-12) + 1e-12)),
            "cos_sim_vs_base": cs,
            "rel_change_vs_base": rc,
            "mse": mse
        }

    # --- 모든 블록에 대해 baseline/lp/hp 측정
    results = []
    for bidx in range(len(model.data_block)):
        rec = {"block": bidx}
        for cond in ["baseline", "lp", "hp"]:
            rec[cond] = run_once(bidx, cond)
        results.append(rec)

    # 저장
    with open(os.path.join(save_dir, "Anal_SSA_high_freq_results.json"), "w") as f:
        json.dump({"keep_ratio": keep_ratio, "per_block": results}, f, indent=2)

    # 간단 요약 프린트
    print("[SSA-HF] keep_ratio =", keep_ratio)
    for r in results:
        b = r["block"]
        g_lp = r["lp"]["gate_mean"];    g_hp = r["hp"]["gate_mean"]
        rg_lp = r["lp"]["response_gain"]; rg_hp = r["hp"]["response_gain"]
        print(f"[SSA-HF][Block {b}] gate_mean  HP:{g_hp:.4e}  LP:{g_lp:.4e} | response_gain  HP:{rg_hp:.4f}  LP:{rg_lp:.4f}")

@torch.no_grad()
def analyze_ssa_hf_sensitivity(
    args,
    base_model=None,
    keep_ratio=0.15,
    gammas=(0.0, 0.5, 1.0, 1.5, 2.0),
    max_batches=16,
    save_key="ssa_hf_sweep"
):
    """
    SSA_rel_scl가 고주파 에너지에 얼마나 민감한지 보기 위한 실험.
    - hippo 입력 x_hippo: [T, B, N, D], 여기서 N이 원래 time-axis에 해당하는 token 축.
    - N축에 대해 LP/HP 분해한 뒤, x = x_lp + gamma * x_hp로 조작.
    - 여러 gamma에 대해 Block별 out_norm, 전체 MSE를 여러 batch에 걸쳐 평균.
    - 결과를 JSON과 곡선 플롯으로 저장.

    keep_ratio: LP에 남길 비율 (예: 0.15면 N 중 하위 15% DCT 모드만 저주파로 간주)
    gammas: 고주파 배율 리스트 (0 → HP 완전 제거, 1 → 원본, >1 → 고주파 증폭)
    max_batches: 실험에 사용할 test batch 개수
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    save_dir = os.path.join(args.save_log_path, save_key)
    os.makedirs(save_dir, exist_ok=True)
    model = base_model if base_model is not None else load_model(args)
    model = model.to(args.device)

    _, loader = data_provider(args, flag='test')
    spikformer = model
    spikformer.eval()

    num_blocks = len(spikformer.data_block)

    # gamma별로 통계 누적용 버퍼
    agg = {
        float(gamma): {
            "in_norm": np.zeros(num_blocks, dtype=np.float64),
            "out_norm": np.zeros(num_blocks, dtype=np.float64),
            "mse": 0.0,
            "count": 0
        }
        for gamma in gammas
    }

    # --- hook factory ---
    def make_pre_hook(block_idx, gamma, buf):
        def _pre(mod, inputs):
            # inputs: (x_hippo, x_neo)
            xin, mx = inputs  # xin: [T, B, N, D]
            # N축 기준 LP/HP 분해
            x_lp, x_hp = _split_lp_hp_on_N(xin, keep_ratio=keep_ratio)
            xin_new = x_lp + gamma * x_hp  # 고주파 스케일 조정

            # 입력 에너지 기록 (L2 norm)
            buf["in"][block_idx] = float(
                xin_new.detach().pow(2).mean().sqrt().cpu()
            )
            return (xin_new, mx)
        return _pre

    def make_post_hook(block_idx, gamma, buf):
        def _post(mod, inputs, output):
            # output: [T, B, N, D]
            out = output
            buf["out"][block_idx] = float(
                out.detach().pow(2).mean().sqrt().cpu()
            )
            return output
        return _post

    # --- 여러 batch에 대해 sweep ---
    for bidx, batch in enumerate(loader):
        if bidx >= max_batches:
            break

        x, y, _, _ = batch
        x = x.float().to(args.device)  # [B, L, C]
        y = y.float().to(args.device)

        for gamma in gammas:
            g = float(gamma)

            # Block별 in/out norm 임시 버퍼 (이번 batch + 이번 gamma)
            buf = {
                "in":  [0.0 for _ in range(num_blocks)],
                "out": [0.0 for _ in range(num_blocks)],
            }

            # hook 등록
            pre_hooks = []
            post_hooks = []
            for bi, blk in enumerate(spikformer.data_block):
                pre_hooks.append(
                    blk.register_forward_pre_hook(
                        make_pre_hook(bi, gamma, buf)
                    )
                )
                post_hooks.append(
                    blk.register_forward_hook(
                        make_post_hook(bi, gamma, buf)
                    )
                )

            # forward
            functional.reset_net(spikformer)
            out = spikformer(x)

            # hook 해제
            for h in pre_hooks + post_hooks:
                h.remove()

            # 출력 정리
            if isinstance(out, (list, tuple)):
                z = out[0]
            else:
                z = out
            if z.dim() == 4:   # [T,B,L,C] 형태면 time 평균
                z = z.mean(0)

            mse = F.mse_loss(
                z[:, -args.pred_len:, :],
                y[:, -args.pred_len:, :]
            ).item()

            # 통계 누적
            agg[g]["mse"]     += mse
            agg[g]["in_norm"] += np.array(buf["in"], dtype=np.float64)
            agg[g]["out_norm"] += np.array(buf["out"], dtype=np.float64)
            agg[g]["count"]   += 1

    # --- 평균 계산 및 JSON 저장 ---
    result = {
        "keep_ratio": keep_ratio,
        "gammas": [float(g) for g in gammas],
        "per_gamma": {}
    }

    for gamma in gammas:
        g = float(gamma)
        entry = agg[g]
        c = max(entry["count"], 1)  # 0 division 방지
        mean_in  = (entry["in_norm"]  / c).tolist()
        mean_out = (entry["out_norm"] / c).tolist()
        mean_mse = entry["mse"] / c

        result["per_gamma"][str(g)] = {
            "mse": mean_mse,
            "in_norm": mean_in,
            "out_norm": mean_out
        }

    json_path = os.path.join(save_dir, "ssa_hf_sweep.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[SSA-HF-sweep] JSON saved to {json_path}")

    # --- 곡선 플롯: 평균 block out_norm vs gamma, MSE vs gamma ---
    gam_arr = np.array([float(g) for g in gammas], dtype=np.float64)
    mse_arr = np.array(
        [result["per_gamma"][str(float(g))]["mse"] for g in gammas],
        dtype=np.float64
    )
    # [G, num_blocks]
    out_block = np.stack(
        [result["per_gamma"][str(float(g))]["out_norm"] for g in gammas],
        axis=0
    )
    out_mean = out_block.mean(axis=1)  # gamma별 Block 평균 출력 norm

    # (1) mean out_norm vs gamma
    plt.figure()
    plt.plot(gam_arr, out_mean, "-o")
    plt.xlabel("HP scaling γ (x_hp coefficient)")
    plt.ylabel("mean Block out_norm")
    plt.title("SSA output energy vs high-frequency scaling")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ssa_hf_outnorm_vs_gamma.png"))
    plt.close()

    # (2) MSE vs gamma
    plt.figure()
    plt.plot(gam_arr, mse_arr, "-o")
    plt.xlabel("HP scaling γ (x_hp coefficient)")
    plt.ylabel("MSE")
    plt.title("Forecasting performance vs high-frequency scaling")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ssa_hf_mse_vs_gamma.png"))
    plt.close()

    print(
        f"[SSA-HF-sweep] Done over {min(bidx+1, max_batches)} batches, "
        f"gammas={list(gammas)}. Results in {save_dir}"
    )