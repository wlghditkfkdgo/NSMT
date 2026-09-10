#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reloaded protocol tests for CLS-style SNN dual-path model

변경점
- patch_size=16으로 T 증가(해상도 ↑)
- 네오 경로는 일관되게 pre-spike(encoding_neo.linear 출력)로 파워 측정
- BN 안전 모드: model.train() + no_grad()
- 주파수 인덱싱은 항상 토큰 길이 T 기준
- 진단 로그 강화
"""

import os, math, json, types
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---- import your model ----
from model import spikformer
from spikingjelly.clock_driven import functional  # reset_net

# ---------- utils ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True) if not os.path.exists(p) else None

def dct_matrix(T, device=None, dtype=torch.float32):
    device = device or 'cpu'
    n = torch.arange(T, device=device, dtype=dtype).reshape(1, T)
    k = torch.arange(T, device=device, dtype=dtype).reshape(T, 1)
    C = torch.cos(math.pi * (n + 0.5) * k / T) * math.sqrt(2.0 / T)
    C[0, :] *= 1 / math.sqrt(2.0)
    return C

def dct_time(x, C, time_dim=1):
    if time_dim != 1:
        perm = list(range(x.dim())); perm[1], perm[time_dim] = perm[time_dim], perm[1]
        x = x.permute(*perm).contiguous()
        inv_need = (perm != list(range(len(perm))))
    else:
        inv_need = False
    Y = torch.matmul(x.transpose(1,2).contiguous(), C.t()).transpose(1,2)
    if inv_need:
        inv = list(range(len(perm))); inv[1], inv[time_dim] = inv[time_dim], inv[1]
        Y = Y.permute(*inv).contiguous()
    return Y

def idct_time(xc, C, time_dim=1):
    if time_dim != 1:
        perm = list(range(xc.dim())); perm[1], perm[time_dim] = perm[time_dim], perm[1]
        xc = xc.permute(*perm).contiguous()
        inv_need = (perm != list(range(len(perm))))
    else:
        inv_need = False
    Y = torch.matmul(xc.transpose(1,2).contiguous(), C).transpose(1,2)
    if inv_need:
        inv = list(range(len(perm))); inv[1], inv[time_dim] = inv[time_dim], inv[1]
        Y = Y.permute(*inv).contiguous()
    return Y

def make_sine(batch, length, channels, freq, fs=1.0, amp=1.0, phase=0.0, device='cpu'):
    t = torch.arange(length, device=device).float() / fs
    sig = amp * torch.sin(2*math.pi*freq*t + phase)
    return sig.view(1, length, 1).repeat(batch, 1, channels)

def make_sweep(batch, length, channels, f_start, f_end, num, device='cpu'):
    freqs = torch.logspace(math.log10(f_start), math.log10(f_end), steps=num, device=device).tolist()
    waves = [make_sine(batch, length, channels, f, amp=1.0, device=device) for f in freqs]
    return freqs, waves

def smooth_1d(x_1d, kernel=5):
    T = x_1d.shape[0]
    if T <= 2: return x_1d
    k = min(kernel, T | 1)
    pad = k // 2
    filt = torch.ones(1,1,k, device=x_1d.device, dtype=x_1d.dtype) / k
    y = F.conv1d(x_1d.view(1,1,T), filt, padding=pad)
    return y.view(T)

# ---------- model wrappers ----------
def build_model(seq_len=512, patch=16, pred_len=1, embed_dim=128, num_heads=2, depths=2, device='cpu'):
    model = spikformer(
        train_mode='training',
        gating='original',
        patch_size=patch,            # ✅ 작은 패치 → T 증가
        pred_len=pred_len,
        seq_len=seq_len,
        time_num_layers=2,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depths=depths,
        bias=False, tau=2.0, spk_encoding=False,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0
    ).to(device)
    functional.reset_net(model)
    return model

def run_model(model, x):
    functional.reset_net(model)
    out = model(x)
    if isinstance(out, (tuple, list)):
        z, hippo_avg, x_neo, org_x, rec_x = out
        return z, hippo_avg, x_neo, org_x, rec_x
    return out, None, None, None, None

# hook: capture pre-spike at encoding_neo.linear
def register_neo_prespike_hook(model):
    box = {'pre': None}
    def hook_fn(module, inp, out):
        box['pre'] = inp[0].detach()  # [T, B, ..., D]
    h = None
    if hasattr(model, 'encoding_neo') and hasattr(model.encoding_neo, 'linear'):
        h = model.encoding_neo.linear.register_forward_hook(hook_fn)
    elif hasattr(model, 'encoding_neo'):
        h = model.encoding_neo.register_forward_hook(hook_fn)
    return box, h

# ---------- Protocol A ----------
def protocol_A_bode(model, device='cpu', outdir='protocol_out', seq_len=512, channels=1):
    ensure_dir(outdir)
    model.train()
    freqs, waves = make_sweep(batch=8, length=seq_len, channels=channels,
                              f_start=1/seq_len, f_end=0.45, num=24, device=device)
    hip_curve, neo_curve = [], []

    with torch.no_grad():
        for f, x in zip(freqs, waves):
            # capture neo pre-spike
            neo_box, h = register_neo_prespike_hook(model)
            z, hippo_avg, x_neo, *_ = run_model(model, x)
            if h is not None: h.remove()

            # reduce to [T]
            H = hippo_avg.float()
            dims = [d for d in range(H.dim()) if d != 0]
            H_power = (H**2).mean(dim=dims)               # [T]

            # pre-spike for neo (fallback)
            neo_pre = neo_box['pre']
            if neo_pre is None:
                # fallback to x_neo
                Nsrc = x_neo.float()
            else:
                Nsrc = neo_pre.float()
            N_power = (Nsrc.reshape(Nsrc.shape[0], -1).norm(dim=-1)**2)  # [T]

            # smooth
            Hs = smooth_1d(H_power, 5)
            Ns = smooth_1d(N_power, 5)

            Ph = torch.fft.rfft(Hs, norm='ortho').abs()**2  # [T//2+1]
            Pn = torch.fft.rfft(Ns, norm='ortho').abs()**2

            Ttok = Hs.shape[0]
            k = int(round(float(f) * (Ttok/2)))
            k = max(0, min(Ph.shape[0]-1, k))

            hip_curve.append(Ph[k].item())
            neo_curve.append(Pn[k].item())

    Hn = np.array(hip_curve); Nn = np.array(neo_curve)
    if Hn.max() > 0: Hn /= Hn.max()
    if Nn.max() > 0: Nn /= Nn.max()

    np.save(os.path.join(outdir, 'bode_freqs.npy'), np.array(freqs))
    np.save(os.path.join(outdir, 'bode_hip.npy'), Hn)
    np.save(os.path.join(outdir, 'bode_neo.npy'), Nn)

    plt.figure()
    plt.loglog(freqs, Hn + 1e-12, label='hippocampus (norm)')
    plt.loglog(freqs, Nn + 1e-12, label='neocortex (norm)')
    plt.xlabel('frequency'); plt.ylabel('normalized power')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'bode_curve.png')); plt.close()

    print(f"[A] T={int(H_power.numel())} | hip pow mean={H_power.mean().item():.4e} std={H_power.std().item():.4e} | neo pow mean={N_power.mean().item():.4e} std={N_power.std().item():.4e}")
    print("[A] Saved: protocol_out/bode_curve.png")

# ---------- Protocol B ----------
def protocol_B_filter_invariance(model, device='cpu', outdir='protocol_out', seq_len=512, channels=1):
    ensure_dir(outdir)
    model.train()
    B=8
    t = torch.linspace(0, 10, steps=seq_len, device=device)
    x_lf = torch.sin(2*math.pi*0.03*t)[None,:,None].repeat(B,1,channels)
    x_hf = 0.6*torch.sin(2*math.pi*0.25*t + 0.3)[None,:,None].repeat(B,1,channels)
    x_mix = x_lf + x_hf

    with torch.no_grad():
        z0, *_ = run_model(model, x_mix)

        # LP/HP 분해
        Cmt = dct_matrix(seq_len, device=device, dtype=x_mix.dtype)
        Xc = dct_time(x_mix, Cmt, time_dim=1)
        k = int(0.15*seq_len)
        m_lp = torch.zeros_like(Xc); m_lp[:, :k, :] = 1.
        m_hp = torch.zeros_like(Xc); m_hp[:, k:, :] = 1.
        x_lp = idct_time(Xc*m_lp, Cmt, time_dim=1)
        x_hp = idct_time(Xc*m_hp, Cmt, time_dim=1)

        # neo pre-spike hook
        def forward_and_capture(inp):
            box, h = register_neo_prespike_hook(model)
            z, hip, neo, *_ = run_model(model, inp)
            pre = box['pre'];  h.remove() if h is not None else None
            # power
            hip_pow = float((hip**2).mean().cpu()) if hip is not None else 0.0
            if pre is not None:
                neo_pow = float((pre**2).mean().cpu())
            else:
                neo_pow = float((neo**2).mean().cpu()) if neo is not None else 0.0
            return hip_pow, neo_pow

        hip_lp, neo_lp = forward_and_capture(x_lp)
        hip_hp, neo_hp = forward_and_capture(x_hp)

    res = dict(
        baseline=float((z0**2).mean().cpu()),
        lp_input=dict(hip=hip_lp, neo=neo_lp),
        hp_input=dict(hip=hip_hp, neo=neo_hp)
    )
    with open(os.path.join(outdir, 'filter_invariance.json'), 'w') as f:
        json.dump(res, f, indent=2)

    ratio_hip = (hip_hp + 1e-12) / (hip_lp + 1e-12)
    ratio_neo = (neo_lp + 1e-12) / (neo_hp + 1e-12)
    print(f"[B] hip HP/LP={ratio_hip:.3f}  | neo LP/HP={ratio_neo:.3f}")
    print("[B] Saved: protocol_out/filter_invariance.json")

# ---------- Protocol C ----------
def protocol_C_spectral_attr(model, device='cpu', outdir='protocol_out', seq_len=512, channels=1):
    ensure_dir(outdir)
    model.train()  # keep grads

    B = 2
    x = torch.randn(B, seq_len, channels, device=device, requires_grad=True)
    Cmt = dct_matrix(seq_len, device=device, dtype=x.dtype)
    Xc = dct_time(x, Cmt, time_dim=1); Xc.requires_grad_(True)
    x_time = idct_time(Xc, Cmt, time_dim=1)

    # --- 전역 스칼라: z.mean() (헤드 출력) ---
    z, hip, neo, *_ = run_model(model, x_time)
    scalar = z.mean()

    # z w.r.t. DCT 계수의 스펙트럴 민감도
    g = torch.autograd.grad(scalar, Xc, retain_graph=False, create_graph=False, allow_unused=False)[0]
    g = g.abs().mean(dim=(0, 2)).detach().cpu().numpy()

    # 저장/시각화
    np.save(os.path.join(outdir, "spectral_attr.npy"), g)
    import matplotlib.pyplot as plt
    plt.figure(); plt.plot(np.arange(len(g)), g)
    plt.xlabel("DCT bin (low→high)"); plt.ylabel("|∂ z̄ / ∂ Xc_k|")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "spectral_attr.png")); plt.close()
    print("[C] Saved spectral attribution -> protocol_out/spectral_attr.png")

@torch.no_grad()
def protocol_C_numerical_occlusion(model, device='cpu', outdir='protocol_out', seq_len=512, channels=1, bins=64, eps=0.01):
    """
    경로별(hippo, neo-pre) 주파수 민감도: DCT 구간을 조금씩 증감시키며 출력 변화량을 측정.
    detach로 막힌 경로도 전방 계산으로 안전하게 측정 가능.
    """
    ensure_dir(outdir)
    model.train()

    B = 2
    x = torch.randn(B, seq_len, channels, device=device)
    Cmt = dct_matrix(seq_len, device=device, dtype=x.dtype)

    # 구간 분할
    edges = torch.linspace(0, seq_len-1, steps=bins+1, device=device).long()
    hip_sens = []; neo_sens = []

    for i in range(bins):
        lo, hi = edges[i].item(), edges[i+1].item()

        # 기준 값
        box, h = register_neo_prespike_hook(model)
        z0, hip0, neo0, *_ = run_model(model, x)
        pre0 = box['pre'];  h.remove() if h is not None else None
        hip_base = float((hip0**2).mean().cpu()) if hip0 is not None else 0.0
        neo_base = float((pre0**2).mean().cpu()) if pre0 is not None else (float((neo0**2).mean().cpu()) if neo0 is not None else 0.0)

        # DCT에서 i번째 구간만 살짝 증폭(+eps)
        Xc = dct_time(x, Cmt, time_dim=1)
        mask = torch.zeros_like(Xc); mask[:, lo:hi, :] = 1.0
        x_pert = idct_time(Xc * (1.0 + eps * mask), Cmt, time_dim=1)

        box, h = register_neo_prespike_hook(model)
        z1, hip1, neo1, *_ = run_model(model, x_pert)
        pre1 = box['pre'];  h.remove() if h is not None else None

        hip_new = float((hip1**2).mean().cpu()) if hip1 is not None else 0.0
        neo_new = float((pre1**2).mean().cpu()) if pre1 is not None else (float((neo1**2).mean().cpu()) if neo1 is not None else 0.0)

        hip_sens.append((hip_new - hip_base) / (eps + 1e-12))
        neo_sens.append((neo_new - neo_base) / (eps + 1e-12))

    hip_sens = np.array(hip_sens); neo_sens = np.array(neo_sens)
    np.save(os.path.join(outdir, "spectral_attr_num_hip.npy"), hip_sens)
    np.save(os.path.join(outdir, "spectral_attr_num_neo.npy"), neo_sens)

    # 그림
    xs = 0.5 * (edges[:-1].cpu().numpy() + edges[1:].cpu().numpy())
    import matplotlib.pyplot as plt
    plt.figure(); plt.plot(xs, hip_sens, label='hip (num)')
    plt.plot(xs, neo_sens, label='neo-pre (num)')
    plt.xlabel("DCT bin center"); plt.ylabel("Δpower / ε")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "spectral_attr_num_both.png")); plt.close()
    print("[C-num] Saved numerical occlusion -> protocol_out/spectral_attr_num_both.png")

# ---------- Protocol D ----------
def add_narrowband_noise(x, center_f_idx, width=5):
    B,L,C = x.shape; device=x.device
    Cmt = dct_matrix(L, device=device, dtype=x.dtype)
    Xc = dct_time(x, Cmt, time_dim=1)
    noise = torch.randn_like(Xc) * 0.2
    lo = max(0, center_f_idx - width//2); hi = min(L, center_f_idx + width//2 + 1)
    m = torch.zeros_like(Xc); m[:, lo:hi, :] = 1.0
    Xc_noisy = Xc + noise * m
    return idct_time(Xc_noisy, Cmt, time_dim=1)

def protocol_D_noise_sensitivity(model, device='cpu', outdir='protocol_out', seq_len=512, channels=1):
    ensure_dir(outdir)
    model.train()
    B=8
    base = torch.sin(2*math.pi*0.08*torch.linspace(0,10,steps=seq_len, device=device))[None,:,None].repeat(B,1,channels)

    with torch.no_grad():
        z_base, *_ = run_model(model, base)
        base_pow = float((z_base**2).mean().cpu())
        drops = []
        K = seq_len//2
        for k in range(4, K, 8):
            noisy = add_narrowband_noise(base.clone(), center_f_idx=k, width=7)
            z_noisy, *_ = run_model(model, noisy)
            drops.append(base_pow - float((z_noisy**2).mean().cpu()))

    np.save(os.path.join(outdir, 'noise_sensitivity.npy'), np.array(drops))
    plt.figure(); plt.plot(np.arange(4, K, 8), drops)
    plt.xlabel("DCT center bin"); plt.ylabel("Power drop vs clean")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'noise_sensitivity.png')); plt.close()
    print("[D] Saved: protocol_out/noise_sensitivity.png")

# ---------- main ----------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = 512
    model = build_model(seq_len=seq_len, patch=64, pred_len=1, embed_dim=128, num_heads=2, depths=2, device=device)

    protocol_A_bode(model, device=device, seq_len=seq_len)
    protocol_B_filter_invariance(model, device=device, seq_len=seq_len)
    protocol_C_spectral_attr(model, device=device, seq_len=seq_len)
    protocol_D_noise_sensitivity(model, device=device, seq_len=seq_len)
    protocol_C_numerical_occlusion(model, device=device, seq_len=seq_len)
    print("[done] Check ./protocol_out/")

if __name__ == "__main__":
    import torch
    main()
