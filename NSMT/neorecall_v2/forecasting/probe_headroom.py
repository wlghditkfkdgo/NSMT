"""Is there ANY headroom for a second pathway? A linear-probe upper bound.

Every Neocortex variant tried so far reads a function of the Hippo representation, which caps what
it can add (docs 8.14.2, 8.17). Variant (1) would instead feed the Neocortex the RAW window. Before
spending a 48-run sweep on it, this asks the cheap version of the question on a TRAINED `h0`
checkpoint -- no retraining:

    H        -> y        the Hippo representation the head actually reads (reference)
    R        -> y        the raw normalised window itself
    [H, R]   -> y        DECISIVE: does the raw window add anything over Hippo?
    G        -> y        a frozen reservoir's view of the raw window
    [H, G]   -> y        does that low-pass view add anything over Hippo?

`[H, R]` is the strict bound for the linear part: every candidate Neocortex feature is a function of
the window, so if the window itself adds nothing to a linear readout, no such feature can either.
It bounds the LINEAR contribution -- a Neocortex entering through cross-attention acts nonlinearly,
so this is evidence, not proof.

usage: probe_headroom.py --ckpt <h0 best+model.pt> --data ETTh1 --pred_len 96 --seed 7
"""
import argparse, os
import numpy as np
import torch

from config import set_random_seed
from data_provider.data_factory import data_provider
from spikingjelly.clock_driven import functional
import model as M
import ours
from diag_neo import build_args, ridge_r2


def h0_args(data, pred_len, seed, gpu, root, seq_len=96, patch_size=8):
    a = build_args(data, pred_len, seed, gpu, root)
    for k, v in dict(neo_xattn=False, serial_mode='off', hippo_mixer='attn',
                     neo_backend='neolayer', neo_recall=True, neo_recall_gate='none',
                     neo_recall_grad='stop', neo_full_grad=False, no_aux=True,
                     seq_len=seq_len, patch_size=patch_size).items():
        setattr(a, k, v)
    return a


def lowfrac(a, keep=0.25):
    """DCT energy fraction in the lowest `keep` of the patch axis. a: [n, N, D]. white = keep."""
    N = a.shape[1]
    k = np.arange(N)[:, None]; t = np.arange(N)[None, :]
    C = np.cos(np.pi * (t + 0.5) * k / N) * np.sqrt(2.0 / N); C[0] /= np.sqrt(2)
    X = np.einsum('kt,ntd->nkd', C, a); p = (X ** 2).mean(axis=(0, 2))
    return float(p[:max(1, int(N * keep))].sum() / (p.sum() + 1e-12))


@torch.no_grad()
def collect(model, loader, dev, res, cap_n=20000):
    """-> H [n,N*D], R [n,N*P], G [n,N*D_res], Y [n,pred_len]"""
    blk = []
    h = model.data_block[0].register_forward_hook(lambda m, i, o: blk.append(o))
    H, R, G, Y = [], [], [], []
    n = 0
    for batch in loader:
        x, y, _, _ = batch
        x = x.float().to(dev); y = y.float().to(dev)[:, -model.pred_len:, :]
        blk.clear(); functional.reset_net(model); model(x)
        B, C = x.shape[0], x.shape[-1]
        hb = blk[-1].transpose(0, 2).mean(2)                      # [N, BC, D]
        H.append(hb.permute(1, 0, 2).reshape(B * C, -1).cpu())
        # org_x layout depends on the path taken: the --no_aux early return in _training skips the
        # rearrange, leaving [N, BC, 1, P]; the normal path leaves [B, C, 1, N, P]. Reshaping the
        # 4-D form as if dims 0,1 were (B,C) interleaves the patch axis with the batch.
        ox = model.org_x
        raw = (ox.squeeze(2).permute(1, 0, 2).contiguous() if ox.dim() == 4
               else ox.reshape(B * C, model.num_patches, -1))    # [BC, N, P] normalised patches
        R.append(raw.reshape(B * C, -1).cpu())
        if res is not None:
            functional.reset_net(res)
            g = res(raw.permute(1, 0, 2).unsqueeze(2).contiguous())   # [N, BC, 1, P] -> [N, BC, 1, D]
            G.append(g.squeeze(2).permute(1, 0, 2).reshape(B * C, -1).cpu())
        Y.append(((y - model.means) / model.stdev).permute(0, 2, 1).reshape(B * C, -1).cpu())
        n += B * C
        if n >= cap_n:
            break
    h.remove()
    out = [torch.cat(v).double().numpy() for v in (H, R, Y)]
    return out[0], out[1], (torch.cat(G).double().numpy() if G else None), out[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--pred_len', type=int, default=96)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--res_size', type=int, default=256)
    ap.add_argument('--seq_len', type=int, default=96)
    ap.add_argument('--patch_size', type=int, default=8)
    ap.add_argument('--root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    o = ap.parse_args()

    set_random_seed(o.seed); torch.use_deterministic_algorithms(False)
    a = h0_args(o.data, o.pred_len, o.seed, o.gpu, o.root, o.seq_len, o.patch_size); dev = a.device
    model = M.load_mymodel(a, True).to(dev)
    miss, unexp = model.load_state_dict(torch.load(o.ckpt, map_location='cpu'), strict=False)
    print(f"[load] missing={len(miss)} unexpected={len(unexp)}")
    model.eval(); model.train_mode = 'training'

    _, tr = data_provider(a, flag='train')
    _, te = data_provider(a, flag='test')

    # ---- calibrate the reservoir for a RAW-patch input ---------------------------------------
    # in_scale=30 was calibrated for Hippo spike rates (mean 0.07); normalised raw patches are
    # zero-mean unit-variance, so the drive is completely different and must be re-tuned or the
    # liquid silences/saturates and the probe is vacuous (docs 8.15.4).
    Htr, Rtr, _, Ytr = collect(model, tr, dev, None)
    probe_in = torch.from_numpy(Rtr[:2048]).float().reshape(-1, model.num_patches, Rtr.shape[1] // model.num_patches)
    probe_in = probe_in.permute(1, 0, 2).unsqueeze(2).to(dev)                     # [N, n, 1, P]
    P = probe_in.shape[-1]
    best = None
    print("[calib] raw-patch 입력에 대한 발화율 (목표 0.05~0.30, 침묵<0.30)")
    for sc in (0.3, 1, 3, 10, 30):
        r = ours.SpikingReservoir(in_dim=P, out_dim=64, size=o.res_size, in_scale=sc,
                                  seed=o.seed, readout='analog').to(dev)
        with torch.no_grad():
            functional.reset_net(r); r(probe_in)
        s = r.last_stats
        ok = 0.05 <= s['rate'] <= 0.30 and s['silent'] < 0.30
        print(f"   in_scale={sc:>5g}  rate={s['rate']:.4f}  silent={s['silent']:.3f}  sat={s['saturated']:.3f}"
              + ("   <- 채택" if ok and best is None else ""))
        if ok and best is None:
            best = (sc, r)
    if best is None:
        print("[calib] 건전 대역 없음 -> 리저버 probe 생략"); res = None
    else:
        res = best[1]

    Htr, Rtr, Gtr, Ytr = collect(model, tr, dev, res)
    Hte, Rte, Gte, Yte = collect(model, te, dev, res)
    print(f"\n=== {o.data}_p{o.pred_len} seed{o.seed}  train {Htr.shape[0]}  test {Hte.shape[0]} ===")
    print(f"    H {Htr.shape[1]}차원  R {Rtr.shape[1]}차원" + (f"  G {Gtr.shape[1]}차원" if Gtr is not None else ""))

    # Fitting on train and scoring on test conflates INFORMATION CONTENT with distribution shift:
    # on ETTh2/ETTm1/ETTm2 the train-fit linear map does not transfer at all (R^2 ~ 0), which makes
    # the deltas uninterpretable. The question here is "is the information present", so score with
    # 2-fold CV inside the test split -- same distribution on both sides of the fit.
    def cv2(Xte_, Yte_, lams_):
        n = Xte_.shape[0] // 2
        out = []
        for tr_s, te_s in ((slice(0, n), slice(n, None)), (slice(n, None), slice(0, n))):
            out.append(max(ridge_r2(Xte_[tr_s], Yte_[tr_s], Xte_[te_s], Yte_[te_s], l) for l in lams_))
        return float(np.mean(out))

    Nq = model.num_patches
    print(f"    [스펙트럼] DCT 저주파 비중 (백색=0.25)   H̄ {lowfrac(Hte.reshape(-1, Nq, Hte.shape[1]//Nq)):.4f}"
          f"   원 윈도우 {lowfrac(Rte.reshape(-1, Nq, Rte.shape[1]//Nq)):.4f}")

    lams = [1, 10, 100, 1000, 1e4, 1e5]
    rows = [("H        -> y  (Hippo, 기준)", Htr, Hte),
            ("R        -> y  (원 윈도우)  ", Rtr, Rte),
            ("[H, R]   -> y  ★ 상한       ", np.hstack([Htr, Rtr]), np.hstack([Hte, Rte]))]
    if Gtr is not None:
        rows += [("G        -> y  (리저버/원신호)", Gtr, Gte),
                 ("[H, G]   -> y  ★           ", np.hstack([Htr, Gtr]), np.hstack([Hte, Gte]))]
    base = base2 = None
    print(f"   {'probe':30s} {'R^2 (train적합/test)':>20s} {'R^2 (test 2-fold CV)':>22s}")
    for name, Xtr, Xte in rows:
        r2, lam = max((ridge_r2(Xtr, Ytr, Xte, Yte, l), l) for l in lams)
        c2 = cv2(Xte, Yte, lams)
        if base is None:
            base, base2 = r2, c2
        print(f"   {name} {r2:+.4f} ({r2-base:+.4f}) {c2:+16.4f} ({c2-base2:+.4f})")


if __name__ == '__main__':
    main()
