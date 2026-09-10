"""Per-horizon-step accumulators and helpers used by ``test.py``.

Kept as a standalone module so the canonical eval flow only adds a single
import line. All shapes follow the convention of ``test.py``:
    pred / true : ``[B, pred_len, C]``
    output      : either tensor ``[B, L, C]`` or ``[T, B, L, C]`` (spike-time
                  dim T collapsed by the helper).
"""

import os

import numpy as np
import torch


def extract_pred_true_for_metrics(output, y, pred_len, long_pred_len=None,
                                  reduce_T="mean"):
    """Pull ``(pred_h, true_h, pred_long_h, true_long_h)`` from a model output.

    output : Tensor or tuple. If tuple, the first element is the prediction.
             Shape is either ``[B, L, C]`` or ``[T, B, L, C]``.
    y      : ``[B, L_y, C]`` ground truth.
    """
    out = output[0] if isinstance(output, tuple) else output
    if out.dim() == 4:
        if reduce_T == "mean":
            out = out.mean(0)
        elif reduce_T == "sum":
            out = out.sum(0)
        else:
            out = out[-1]
    assert out.dim() == 3, f"expected pred shape [B,L,C], got {tuple(out.shape)}"
    pred_h = out[:, -pred_len:, :].detach().cpu()
    true_h = y[:, -pred_len:, :].detach().cpu()
    long_pred_len = long_pred_len or pred_len
    L = min(long_pred_len, out.shape[1], y.shape[1])
    pred_long_h = out[:, -L:, :].detach().cpu()
    true_long_h = y[:, -L:, :].detach().cpu()
    return pred_h, true_h, pred_long_h, true_long_h


class HorizonStats:
    """Per-horizon-step accumulator for MAE / MSE.

    update : pred, true of shape ``[B, pred_len, C]``
    compute: returns ``mae_h, mse_h`` each of shape ``[pred_len]``
    """

    def __init__(self, pred_len, device="cpu"):
        self.pred_len = int(pred_len)
        self.device = device
        self.mae_sum = torch.zeros(self.pred_len, device=device)
        self.mse_sum = torch.zeros(self.pred_len, device=device)
        self.count = torch.zeros(self.pred_len, device=device)

    def update(self, pred: torch.Tensor, true: torch.Tensor):
        # pred,true: [B, pred_len, C]
        L = min(pred.shape[-2], true.shape[-2], self.pred_len)
        diff = (pred[..., -L:, :] - true[..., -L:, :]).to(self.device)
        # diff : [B, L, C]
        n = diff.shape[0] * diff.shape[-1]
        self.mae_sum[:L] += diff.abs().sum(dim=(0, -1))
        self.mse_sum[:L] += (diff ** 2).sum(dim=(0, -1))
        self.count[:L] += n

    def compute(self):
        c = torch.clamp(self.count, min=1)
        return self.mae_sum / c, self.mse_sum / c


def summarize_bins_from_horizon(mae_h, mse_h, pred_len, near_k=24, far_k=192):
    near = min(near_k, pred_len)
    far = min(far_k, pred_len)
    return {
        "mae_overall": float(mae_h.mean()),
        "mse_overall": float(mse_h.mean()),
        "mae_near": float(mae_h[:near].mean()),
        "mse_near": float(mse_h[:near].mean()),
        "mae_far": float(mae_h[-far:].mean()),
        "mse_far": float(mse_h[-far:].mean()),
    }


def save_horizon_artifacts(mae_h, mse_h, save_dir, prefix="h", make_plot=True):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{prefix}_horizon_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("step,mae,mse\n")
        for i, (a, b) in enumerate(zip(mae_h.tolist(), mse_h.tolist())):
            f.write(f"{i},{a:.6f},{b:.6f}\n")
    if make_plot:
        try:
            import matplotlib.pyplot as plt
            steps = list(range(len(mae_h)))
            fig, ax1 = plt.subplots(figsize=(7, 3))
            ax1.plot(steps, mae_h.tolist(), label="MAE", color="tab:blue")
            ax2 = ax1.twinx()
            ax2.plot(steps, mse_h.tolist(), label="MSE",
                     color="tab:red", linestyle="--")
            ax1.set_xlabel("horizon step")
            ax1.set_ylabel("MAE", color="tab:blue")
            ax2.set_ylabel("MSE", color="tab:red")
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, f"{prefix}_horizon_curves.png"),
                        dpi=200)
            plt.close(fig)
        except Exception:
            pass
