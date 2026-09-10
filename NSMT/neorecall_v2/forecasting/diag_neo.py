"""What does the Neocortex actually learn?

Runs on a TRAINED checkpoint -- no retraining. Three families of measurement:

  1. causal ablation   -- replace x_neo with zeros / batch-shuffled / patch-shuffled and re-measure
                          test MSE. This is the only direct test of whether the Neocortex is USED.
  2. linear probes     -- ridge from x_neo (and from the Hippo representation, as the reference)
                          onto {future target, raw patches, low-frequency trend, Hippo itself}.
                          Fit on train, R^2 on test. Answers "is the information there at all".
  3. representation    -- firing rate, patch-axis autocorrelation, DCT low-frequency fraction,
                          participation ratio.

usage: diag_neo.py <ckpt_dir_glob> --data ETTh1 --pred_len 96 --seed 7 [--gpu 0]
"""
import argparse, glob, os, sys, json
import numpy as np
import torch

from config import Config, set_random_seed
from data_provider.data_factory import data_provider
import model as M
import ours


def build_args(data, pred_len, seed, gpu, root):
    a = Config.__new__(Config)
    for k, v in dict(
        model='myModel', seed=seed, data=data, data_path=f'{data}.csv',
        root_path=os.path.join(root, 'forecasting/dataset/ETT-small'),
        freq='h' if data.startswith('ETTh') else 't', features='M', target='OT',
        seq_len=96, pred_len=pred_len, label_len=0, patch_size=8, c_in=7,
        embed_dim=64, num_heads=8, num_layers=1, time_num_layers=2, mlp_ratios=1.0,
        max_ratio=2, keep_ratio=0.25, gating='attn', bias=False, tau=2.0, alpha=0.0,
        batch_size=64, num_workers=4, perm=False, spk_encoding=False, scale=True,
        embed='timeF', augmentation_ratio=0, seasonal_patterns=None, task_name='long_term_forecast',
        device=torch.device(f'cuda:{gpu}'), num_device=gpu,
        neo_recall=True, neo_recall_tgt='raw', neo_recall_gate='alpha',
        neo_recall_grad='full', neo_full_grad=True, init_order_fix=True,
        neo_backend='reservoir', neo_xattn=True, refine_steps=1,
        res_size=256, res_in_scale=30.0, res_wrec='full', res_seed=seed,
        train_mode='training', inverse=False,
    ).items():
        setattr(a, k, v)
    return a


@torch.no_grad()
def collect(model, loader, device, cap=20000):
    """-> H (Hippo repr), G (Neocortex), Y (normalised future), P (raw patches), all [n, d]."""
    from spikingjelly.clock_driven import functional
    blk, neo = [], []
    h1 = model.data_block[0].register_forward_hook(lambda m, i, o: blk.append(o))
    h2 = model.reservoir.register_forward_hook(lambda m, i, o: neo.append(o))
    H, G, Y, P = [], [], [], []
    n = 0
    for batch in loader:
        x, y, _, _ = batch
        x = x.float().to(device); y = y.float().to(device)[:, -model.pred_len:, :]
        blk.clear(); neo.clear()
        functional.reset_net(model)
        model(x)
        B, C = x.shape[0], x.shape[-1]
        # final Block output -> Hippo representation the head reads (T pooled away)
        hb = blk[-1].transpose(0, 2).mean(2)                      # [N, BC, D]
        gb = neo[-1].squeeze(2)                                   # [N, BC, D]
        H.append(hb.permute(1, 0, 2).reshape(B * C, -1).cpu())
        G.append(gb.permute(1, 0, 2).reshape(B * C, -1).cpu())
        yn = ((y - model.means) / model.stdev)                    # [B, pred_len, C]
        Y.append(yn.permute(0, 2, 1).reshape(B * C, -1).cpu())
        # _training leaves org_x as [B, C, 1, N, P]; dims 0,1 are already (batch, channel)
        P.append(model.org_x.reshape(B * C, -1).cpu())
        n += B * C
        if n >= cap:
            break
    h1.remove(); h2.remove()
    return [torch.cat(v).double().numpy() for v in (H, G, Y, P)]


def ridge_r2(Xtr, Ytr, Xte, Yte, lam=1.0):
    """closed-form ridge, R^2 on the held-out split (variance-weighted over outputs)."""
    mx, my = Xtr.mean(0, keepdims=True), Ytr.mean(0, keepdims=True)
    Xc, Yc = Xtr - mx, Ytr - my
    A = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
    W = np.linalg.solve(A, Xc.T @ Yc)
    pred = (Xte - mx) @ W + my
    ss_res = ((Yte - pred) ** 2).sum()
    ss_tot = ((Yte - Yte.mean(0, keepdims=True)) ** 2).sum()
    return 1.0 - ss_res / ss_tot


def dct_lowfreq_fraction(sig, keep=0.25):
    """fraction of DCT energy in the lowest `keep` of the patch axis. sig: [n, N, D]."""
    N = sig.shape[1]
    k = np.arange(N)[:, None]; t = np.arange(N)[None, :]
    Cm = np.cos(np.pi * (t + 0.5) * k / N) * np.sqrt(2.0 / N); Cm[0] /= np.sqrt(2)
    X = np.einsum('kt,ntd->nkd', Cm, sig)
    p = (X ** 2).mean(axis=(0, 2))
    return float(p[:max(1, int(N * keep))].sum() / (p.sum() + 1e-12))


@torch.no_grad()
def ablate(model, loader, device, mode):
    """test MSE with x_neo replaced. mode: none|zero|shuffle_batch|shuffle_patch"""
    from spikingjelly.clock_driven import functional
    orig = ours.SpikingReservoir.forward

    def patched(self, e, org_seq=None):
        out = orig(self, e, org_seq)                              # [N, BC, 1, D]
        if mode == 'zero':
            return torch.zeros_like(out)
        if mode == 'shuffle_batch':
            return out[:, torch.randperm(out.shape[1], device=out.device)]
        if mode == 'shuffle_patch':
            # NOTE: attention here is softmax-free `(q@k^T)@v`, summed over the key index, and the
            # keys carry no positional encoding -- so permuting the Neocortex patch axis is an
            # EXACT identity. Kept as a wiring check, not as evidence about what Neo encodes.
            return out[torch.randperm(out.shape[0], device=out.device)]
        return out

    ours.SpikingReservoir.forward = patched
    se, cnt = 0.0, 0
    model.train_mode = 'testing'
    try:
        for batch in loader:
            x, y, _, _ = batch
            x = x.float().to(device); y = y.float().to(device)[:, -model.pred_len:, :]
            functional.reset_net(model)
            z = model(x)
            z = z.mean(0) if z.dim() == 4 else z                  # T-mean, as test.py does
            se += ((z[:, -model.pred_len:, :] - y) ** 2).sum().item()
            cnt += y.numel()
    finally:
        ours.SpikingReservoir.forward = orig
        model.train_mode = 'training'
    return se / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--pred_len', type=int, default=96)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    ap.add_argument('--lam', type=float, default=10.0)
    o = ap.parse_args()

    set_random_seed(o.seed)
    torch.use_deterministic_algorithms(False)          # hooks + shuffle ops
    a = build_args(o.data, o.pred_len, o.seed, o.gpu, o.root)
    dev = a.device
    model = M.load_mymodel(a, True).to(dev)
    sd = torch.load(o.ckpt, map_location='cpu')
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    model.eval(); model.train_mode = 'training'

    _, tr = data_provider(a, flag='train')
    _, te = data_provider(a, flag='test')

    Htr, Gtr, Ytr, Ptr = collect(model, tr, dev)
    Hte, Gte, Yte, Pte = collect(model, te, dev)
    n_tr, n_te = Htr.shape[0], Hte.shape[0]
    print(f"\n=== {o.data}_p{o.pred_len} seed{o.seed}   train {n_tr}  test {n_te} ===")

    # ---- representation statistics -------------------------------------------------
    N = model.num_patches; D = Gte.shape[1] // N
    g3 = Gte.reshape(-1, N, D)
    fr = float(g3.mean())
    ac = np.corrcoef(g3[:, :-1].ravel(), g3[:, 1:].ravel())[0, 1]
    pr_ev = np.linalg.eigvalsh(np.cov(Gte.T) + 1e-12 * np.eye(Gte.shape[1]))
    pr = float(pr_ev.sum() ** 2 / (pr_ev ** 2).sum())
    print(f"[repr]  발화율 {fr:.4f}   패치축 autocorr {ac:+.4f}   "
          f"DCT 저주파비중 {dct_lowfreq_fraction(g3):.4f}   participation ratio {pr:.1f}/{Gte.shape[1]}")

    # ---- linear probes --------------------------------------------------------------
    # lambda is swept per probe: a fixed lambda would penalise the concatenated feature set
    # (2x the columns) purely for being wider, which is exactly the comparison we care about.
    lams = [0.1, 1, 10, 100, 1000, 1e4]
    print(f"[probe] ridge, lambda 그리드 최적 (train 적합 / test R^2)")
    for name, Xtr, Xte_, Ytr_, Yte_ in [
        ("H̄      -> 미래 y  ", Htr, Hte, Ytr, Yte),
        ("Neo     -> 미래 y  ", Gtr, Gte, Ytr, Yte),
        ("[H̄,Neo] -> 미래 y  ", np.hstack([Htr, Gtr]), np.hstack([Hte, Gte]), Ytr, Yte),
        ("Neo     -> 원 패치 ", Gtr, Gte, Ptr, Pte),
        ("H̄      -> 원 패치 ", Htr, Hte, Ptr, Pte),
        ("Neo     -> H̄      ", Gtr, Gte, Htr, Hte),
    ]:
        rs = [(ridge_r2(Xtr, Ytr_, Xte_, Yte_, l), l) for l in lams]
        best, bl = max(rs)
        print(f"   {name} R^2 = {best:+.4f}  (lam={bl:g})")

    # ---- causal ablation -------------------------------------------------------------
    print("[ablate] test MSE (정규화 공간 아님, 원 스케일)")
    base = ablate(model, te, dev, 'none')
    for m in ('none', 'zero', 'shuffle_batch', 'shuffle_patch'):
        v = base if m == 'none' else ablate(model, te, dev, m)
        print(f"   x_neo={m:14s} MSE {v:.6f}   Δ {100*(v-base)/base:+.2f}%")


if __name__ == '__main__':
    main()
