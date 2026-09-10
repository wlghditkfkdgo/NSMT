#!/usr/bin/env python3
"""Tier-1 mechanistic analysis: prove the (ii) active Neocortex implements
*active storage* (Zipser 1993), independent of end-task accuracy.

Given a trained ACTIVE checkpoint (RecurrentNeoLayer weights), measure:
  (1) memory half-life  : perturbation-persistence of the re-entrant loop
                          (full W_rec vs W_rec:=0 ablation), t_1/2 in steps.
  (2) spectral radius    : rho(W_rec) and effective loop-gain lambda_eff.
  (3) fixed-point attractor: zero-input iteration -> does activity converge to
                          a sustained fixed point (active storage) or die (passive)?

We reconstruct the single-step LIF recurrence in plain torch from the trained
weights (spikingjelly LIFNode, decay_input=True, v_th=1, hard reset):
    pre(t) = W_in in(t) + W_rec s(t-1);  H = (1-1/tau)V + (1/tau) BN(pre)
    s = 1[H>=v_th];  V = H*(1-s)
"""
import argparse, glob, os, re, json
import torch


def find_ckpt(log_dir):
    c = glob.glob(os.path.join(log_dir, "**", "best+model.pt"), recursive=True)
    return c[0] if c else None


def load_recurrent_weights(sd):
    """Locate RecurrentNeoLayer sub-weights in a state_dict. Returns list of
    dicts (one per Mem layer) with w_in, w_rec, bn params, or [] if none."""
    # keys look like: ...time_block.Mem.<l>.w_rec.weight
    layers = {}
    for k, v in sd.items():
        m = re.search(r"Mem\.(\d+)\.(w_in|w_rec)\.weight$", k)
        if m:
            l = int(m.group(1)); layers.setdefault(l, {})[m.group(2)] = v
        mb = re.search(r"Mem\.(\d+)\.bn\.(weight|bias|running_mean|running_var)$", k)
        if mb:
            l = int(mb.group(1)); layers.setdefault(l, {})["bn_" + mb.group(2)] = v
    return [layers[l] for l in sorted(layers) if "w_rec" in layers[l]]


def bn_apply(pre, L, eps=1e-5):
    if "bn_running_mean" not in L:
        return pre
    m = L["bn_running_mean"]; var = L["bn_running_var"]
    g = L.get("bn_weight"); b = L.get("bn_bias")
    out = (pre - m) / torch.sqrt(var + eps)
    if g is not None: out = out * g + b
    return out


def simulate(L, tau, T, in_seq, use_rec=True, v_th=1.0):
    """Run the single-step LIF recurrence. in_seq: [T, D]. Returns H_traj [T,D],
    spike_traj [T,D]."""
    Win = L["w_in"]; Wrec = L["w_rec"]; D = Win.shape[0]
    decay = 1.0 - 1.0 / tau
    V = torch.zeros(D); s = torch.zeros(D)
    Hs, Ss = [], []
    for t in range(T):
        pre = in_seq[t] @ Win.T + (s @ Wrec.T if use_rec else 0.0)
        pre = bn_apply(pre.unsqueeze(0), L).squeeze(0)
        H = decay * V + (1.0 / tau) * pre
        s = (H >= v_th).float()
        V = H * (1.0 - s)
        Hs.append(H.clone()); Ss.append(s.clone())
    return torch.stack(Hs), torch.stack(Ss)


def memory_capacity(L, tau, K=40, T=1600, washout=300, seed=0, use_rec=True, ridge=1e-2):
    """Reservoir-computing Memory Capacity (Jaeger 2001), monotonic in recurrent
    memory. Drive the layer with scalar i.i.d. input u(t) along a fixed random
    direction; for each delay k, r^2 of the best LINEAR reconstruction of u(t-k)
    from the current membrane state. MC = sum_k MC_k = #past inputs retained.
    Returns (total_MC, [MC_0, MC_1, ...])."""
    torch.manual_seed(seed)
    D = L["w_in"].shape[0]
    u = torch.rand(T) * 2 - 1                       # i.i.d. U(-1,1)
    e = torch.randn(D); e = e / e.norm()            # fixed input direction
    in_seq = u.unsqueeze(1) * e.unsqueeze(0)        # [T, D]
    H, _ = simulate(L, tau, T, in_seq, use_rec=use_rec)
    X = H[washout:]                                 # [T', D] states (membrane)
    uu = u[washout:]
    Tp = X.shape[0]
    Xb = torch.cat([X, torch.ones(Tp, 1)], dim=1)   # + bias
    mc_k = []
    I = ridge * torch.eye(Xb.shape[1])
    for k in range(K + 1):
        if k >= Tp - 5: break
        Xk = Xb[k:]                                 # state at t
        y = uu[:Tp - k]                             # target u(t-k)
        w = torch.linalg.solve(Xk.T @ Xk + I, Xk.T @ y)
        pred = Xk @ w
        ss_res = ((y - pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum() + 1e-12
        mc_k.append(float((1 - ss_res / ss_tot).clamp(0, 1)))
    return round(sum(mc_k), 3), [round(x, 4) for x in mc_k]


def half_life(diff_norm):
    """First step at which the perturbation norm drops below half its peak."""
    peak = diff_norm.max().item()
    if peak <= 0: return 0.0
    half = peak / 2.0
    below = (diff_norm < half).nonzero().flatten()
    peak_t = int(diff_norm.argmax())
    after = below[below > peak_t]
    return float(after[0].item() - peak_t) if len(after) else float(len(diff_norm))


def analyze(log_dir, tau, T=64, seed=0, label=""):
    torch.manual_seed(seed)
    ckpt = find_ckpt(log_dir)
    if ckpt is None:
        return {"error": f"no checkpoint under {log_dir}"}
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) and "state_dict" in sd else sd
    Ls = load_recurrent_weights(sd)
    if not Ls:
        return {"ckpt": ckpt, "has_recurrent": False,
                "note": "no RecurrentNeoLayer (passive/feedforward Neo) — active storage N/A"}
    L = Ls[-1]; D = L["w_in"].shape[0]
    res = {"ckpt": ckpt, "has_recurrent": True, "D": D, "tau": tau, "n_layers": len(Ls)}

    # (2) spectral radius of W_rec + effective loop gain
    ev = torch.linalg.eigvals(L["w_rec"].float())
    rho = ev.abs().max().item()
    lam_leak = 1.0 - 1.0 / tau
    res["spectral_radius_Wrec"] = round(rho, 4)
    res["lambda_leak(passive)"] = round(lam_leak, 4)
    res["lambda_eff(active~)"] = round(lam_leak + (1.0 / tau) * rho, 4)

    # (1) PRIMARY: Memory Capacity (monotonic), full W_rec vs recurrence removed
    mc_a, mck_a = memory_capacity(L, tau, use_rec=True)
    mc_n, mck_n = memory_capacity(L, tau, use_rec=False)
    res["MC_active"] = mc_a
    res["MC_norec"] = mc_n
    res["MC_gain_from_recurrence"] = round(mc_a - mc_n, 3)
    res["MC_curve_active"] = mck_a          # MC_k vs delay k
    res["MC_curve_norec"] = mck_n

    # (3) fixed-point attractor: zero external input, random init spikes, iterate
    torch.manual_seed(seed + 1)
    s = (torch.rand(D) < 0.1).float(); V = torch.zeros(D)
    decay = 1.0 - 1.0 / tau; states = []
    for t in range(200):
        pre = bn_apply((s @ L["w_rec"].T).unsqueeze(0), L).squeeze(0)
        H = decay * V + (1.0 / tau) * pre
        s2 = (H >= 1.0).float(); V = H * (1.0 - s2)
        states.append(s2)
        s = s2
    tail = torch.stack(states[-50:])
    sustained_rate = tail.mean().item()                    # >0 => activity sustained
    # settle: variance of firing pattern across last 50 steps (low => fixed point)
    settle = tail.std(dim=0).mean().item()
    res["attractor_sustained_rate"] = round(sustained_rate, 4)
    res["attractor_settle_std"] = round(settle, 4)
    res["attractor_verdict"] = ("sustained-fixed-point (active storage)"
                                if sustained_rate > 0.01 and settle < 0.15
                                else ("sustained-dynamic" if sustained_rate > 0.01
                                      else "decays-to-silence (no active storage)"))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--active_dir", required=True)
    ap.add_argument("--passive_dir", default=None)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--out", default="tier1_result.json")
    a = ap.parse_args()
    out = {"active": analyze(a.active_dir, a.tau, a.T, label="active")}
    if a.passive_dir:
        out["passive"] = analyze(a.passive_dir, a.tau, a.T, label="passive")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    # concise console summary
    A = out["active"]
    print("=== Tier-1 mechanistic analysis (active Neocortex) ===")
    if A.get("has_recurrent"):
        print(f"  rho(W_rec)={A['spectral_radius_Wrec']}  lambda_leak={A['lambda_leak(passive)']}"
              f"  lambda_eff={A['lambda_eff(active~)']}")
        print(f"  Memory Capacity: active={A['MC_active']}  no-recurrence={A['MC_norec']}"
              f"  GAIN={A['MC_gain_from_recurrence']}")
        print(f"  attractor: sustained_rate={A['attractor_sustained_rate']} "
              f"settle_std={A['attractor_settle_std']} -> {A['attractor_verdict']}")
    else:
        print("  ", A.get("note", A.get("error")))
    print(f"saved -> {a.out}")
