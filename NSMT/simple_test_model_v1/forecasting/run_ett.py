"""Small reproducible ETT trainer; all windows, train-only scaling, val selection.

Uses the same 12/4/4-month borders as model_v1's Dataset_ETT_hour/minute.
Small ETT arrays reside on the selected GPU; indexing replaces loader workers.
Each process owns one independent model. Test is evaluated only after restoring
the minimum-validation-MSE checkpoint. Metrics cover every horizon/channel and
include the final partial batch. Local checkpoints live in log/, JSON in results/.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

TASK = Path(__file__).resolve().parent
NSMT = TASK.parents[1]
sys.path.insert(0, str(NSMT))
from forecasting.simple_test_model_v1 import SimpleTestModelV1

VARIANTS = ("population", "temporal", "temporal_embedding", "no_attention", "linear")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class ETTWindows:
    def __init__(self, path, dataset, seq_len, pred_len, device):
        frame = pd.read_csv(path)
        columns = [c for c in frame.columns if c != "date"]
        values = frame[columns].to_numpy(dtype=np.float64)
        if values.shape[1] != 7 or not np.isfinite(values).all():
            raise ValueError("Expected seven finite ETT features")
        factor = 4 if dataset.startswith("ETTm") else 1
        ends = [12 * 30 * 24 * factor, 16 * 30 * 24 * factor, 20 * 30 * 24 * factor]
        if len(values) < ends[-1]:
            raise ValueError("Dataset is shorter than the canonical 20-month split")
        scaler = StandardScaler().fit(values[:ends[0]])
        scaled = scaler.transform(values).astype(np.float32)
        starts = [0, ends[0] - seq_len, ends[1] - seq_len]
        self.seq_len, self.pred_len = seq_len, pred_len
        self.arrays, self.counts = {}, {}
        for name, start, end in zip(("train", "val", "test"), starts, ends):
            self.arrays[name] = torch.from_numpy(scaled[start:end].copy()).to(device)
            self.counts[name] = end - start - seq_len - pred_len + 1
            if self.counts[name] < 1:
                raise ValueError("No complete forecasting windows in split")
        self.x_offsets = torch.arange(seq_len, device=device)
        self.y_offsets = torch.arange(seq_len, seq_len + pred_len, device=device)
        self.metadata = {
            "path": str(Path(path).resolve()), "sha256": sha256(path), "features": columns,
            "total_rows": len(values), "used_rows": ends[-1], "seq_len": seq_len,
            "pred_len": pred_len, "scale_fit_rows": [0, ends[0]],
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "splits": {name: {"context_start": start, "end_exclusive": end,
                               "first_target_start": start + seq_len, "windows": self.counts[name]}
                       for name, start, end in zip(("train", "val", "test"), starts, ends)},
        }

    def batches(self, split, batch_size, generator=None, max_batches=0):
        order = torch.randperm(self.counts[split], generator=generator) if generator is not None else torch.arange(self.counts[split])
        order = order.to(self.arrays[split].device)
        for batch_index, start in enumerate(range(0, len(order), batch_size)):
            if max_batches and batch_index >= max_batches:
                break
            indices = order[start:start + batch_size, None]
            yield self.arrays[split][indices + self.x_offsets], self.arrays[split][indices + self.y_offsets]


class LinearForecast(nn.Module):
    """Shared per-channel linear forecaster with the same input-window norm."""
    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.head = nn.Linear(seq_len, pred_len, bias=False)

    def forward(self, x):
        mean = x.mean(1, keepdim=True).detach()
        std = (x.var(1, unbiased=False, keepdim=True) + 1e-5).sqrt().detach()
        return self.head(((x - mean) / std).transpose(1, 2)).transpose(1, 2) * std + mean


def make_model(args):
    if args.variant == "linear":
        return LinearForecast(args.seq_len, args.pred_len), {"seq_len": args.seq_len, "pred_len": args.pred_len}
    config = dict(
        seq_len=args.seq_len, pred_len=args.pred_len, num_population=args.population,
        d_model=args.d_model, num_heads=args.heads, depth=args.depth, mlp_ratio=2.0,
        input_mode="patch", patch_size=args.patch_size, stride=args.patch_size,
        encoding="direct", readout="two_stage", head_dim=args.head_dim,
        tau=2.0, threshold=1.0, bias=False, normalize=True, backend=args.backend,
        attn_scale=args.attn_scale, attention_axis={"population": "population", "no_attention": "none"}.get(args.variant, "temporal"),
        learn_population_embedding=args.variant == "temporal_embedding",
        population_low=-3.0, population_high=3.0, population_width=1.0,
    )
    return SimpleTestModelV1(**config), config


class SpikeMonitor:
    def __init__(self, model):
        self.enabled = False
        self.values = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, MultiStepLIFNode):
                self.handles.append(module.register_forward_hook(self.hook(name)))

    def hook(self, name):
        def collect(module, inputs, output):
            if self.enabled:
                self.values[name] = output.detach().float().mean()
        return collect

    def summary(self):
        return {name: float(value) for name, value in self.values.items()}


@torch.no_grad()
def evaluate(model, data, split, batch_size, monitor, max_batches=0, baselines=False):
    model.eval()
    totals = torch.zeros(6, device=data.arrays[split].device, dtype=torch.float64)
    count = 0
    for index, (x, y) in enumerate(data.batches(split, batch_size, max_batches=max_batches)):
        monitor.enabled = index == 0
        output = model(x)
        monitor.enabled = False
        difference = (output - y).double()
        totals[0] += difference.square().sum()
        totals[1] += difference.abs().sum()
        if baselines:
            for offset, prediction in ((2, x[:, -1:]), (4, x.mean(1, keepdim=True))):
                error = (prediction - y).double()
                totals[offset] += error.square().sum()
                totals[offset + 1] += error.abs().sum()
        count += y.numel()
    metrics = {"mse": float(totals[0] / count), "mae": float(totals[1] / count), "elements": count}
    if baselines:
        metrics["persistence"] = {"mse": float(totals[2] / count), "mae": float(totals[3] / count)}
        metrics["window_mean"] = {"mse": float(totals[4] / count), "mae": float(totals[5] / count)}
    if not all(np.isfinite(metrics[k]) for k in ("mse", "mae")):
        raise FloatingPointError("Nonfinite evaluation metric")
    return metrics, monitor.summary()


def run(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    torch.cuda.set_device(device) if device.type == "cuda" else None
    generator = torch.Generator().manual_seed(args.seed + 1000)
    run_id = f"{args.dataset}_p{args.pred_len}_{args.variant}_seed{args.seed}"
    log_dir = TASK / "log" / args.suite / run_id
    result_path = TASK / "results" / args.suite / f"{run_id}.json"
    if log_dir.exists() or result_path.exists():
        raise FileExistsError(f"Run already exists; choose another --suite: {run_id}")
    log_dir.mkdir(parents=True)
    start_time = time.time()
    data = ETTWindows(Path(args.data_root) / f"{args.dataset}.csv", args.dataset, args.seq_len, args.pred_len, device)
    model, model_config = make_model(args)
    model = model.to(device)
    monitor = SpikeMonitor(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    metadata = {
        "run_id": run_id, "command": shlex.join([sys.executable, *sys.argv]), "cwd": str(Path.cwd()),
        "config": vars(args), "model_config": model_config, "data": data.metadata,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=NSMT, text=True).strip(),
        "source_sha256": {str(path.relative_to(NSMT)): sha256(path) for path in
                          (Path(__file__).resolve(), NSMT / "forecasting/simple_test_model_v1.py")},
        "environment": {"python": sys.version, "torch": torch.__version__,
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "packages": {p: importlib.metadata.version(p) for p in ("spikingjelly", "cupy-cuda11x", "numpy", "pandas", "scikit-learn")}},
        "parameters": sum(p.numel() for p in model.parameters()),
        "protocol": {"loss": "MSE", "metrics": "train-standardized scale, all channels and horizon points",
                     "selection": "minimum validation MSE", "test_selection": False, "drop_last": False,
                     "gradient_clip": 1.0, "scheduler": "ReduceLROnPlateau(val MSE, factor=.5, patience=1)",
                     "spike_diagnostics": "first batch of each split, not full-split firing rates",
                     "quick_smoke_only": bool(args.max_train_batches or args.max_eval_batches)},
    }
    write_json(log_dir / "config.json", metadata)
    print(json.dumps({"event": "start", "run": run_id, "parameters": metadata["parameters"], "windows": data.counts}), flush=True)
    best, best_epoch, bad_epochs, history = float("inf"), 0, 0, []
    checkpoint = log_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_total = torch.zeros((), device=device, dtype=torch.float64)
        train_count = 0
        train_rates, gradient_norms = {}, {}
        for index, (x, y) in enumerate(data.batches("train", args.batch_size, generator, args.max_train_batches)):
            optimizer.zero_grad(set_to_none=True)
            monitor.enabled = index == 0
            prediction = model(x)
            monitor.enabled = False
            loss = (prediction - y).square().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if index == 0:
                train_rates = monitor.summary()
                gradient_norms = {name: float(parameter.grad.detach().norm()) for name, parameter in model.named_parameters()
                                  if parameter.grad is not None and (name.endswith("linear.weight") or name in
                                                                    ("population_embedding", "head.weight", "head_compress.weight"))}
            optimizer.step()
            train_total += loss.detach().double() * y.numel()
            train_count += y.numel()
        validation, val_rates = evaluate(model, data, "val", args.batch_size, monitor, args.max_eval_batches)
        improved = validation["mse"] < best
        if improved:
            best, best_epoch, bad_epochs = validation["mse"], epoch, 0
            # Save state_dict only; membrane states are reset per forward, never persisted.
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "validation": validation, "model_config": model_config}, checkpoint)
        else:
            bad_epochs += 1
        record = {"epoch": epoch, "train_mse": float(train_total / train_count), "validation": validation,
                  "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - epoch_start,
                  "train_first_batch_spikes": train_rates, "val_first_batch_spikes": val_rates,
                  "first_batch_gradient_norms": gradient_norms, "best": improved}
        history.append(record)
        write_json(log_dir / "history.json", history)
        print(json.dumps({"event": "epoch", "run": run_id, "epoch": epoch,
                          "train_mse": record["train_mse"], "val_mse": validation["mse"],
                          "best_epoch": best_epoch, "seconds": record["seconds"]}), flush=True)
        scheduler.step(validation["mse"])
        if bad_epochs >= args.patience:
            break
    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["state_dict"])
    restored_val, restored_rates = evaluate(model, data, "val", args.batch_size, monitor, args.max_eval_batches)
    if abs(restored_val["mse"] - best) > 1e-7 * max(1.0, abs(best)):
        raise AssertionError("Restored checkpoint does not reproduce selected validation score")
    test, test_rates = evaluate(model, data, "test", args.batch_size, monitor, args.max_eval_batches, baselines=True)
    result = {**metadata, "status": "complete", "best_epoch": best_epoch, "epochs_run": len(history),
              "validation": restored_val, "test": test, "history": history,
              "selected_val_first_batch_spikes": restored_rates, "test_first_batch_spikes": test_rates,
              "seconds": time.time() - start_time, "checkpoint": str(checkpoint),
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
    write_json(result_path, result)
    print(json.dumps({"event": "complete", "run": run_id, "best_epoch": best_epoch,
                      "test": test, "seconds": result["seconds"], "result": str(result_path)}), flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("ETTh1", "ETTh2", "ETTm1", "ETTm2"), required=True)
    p.add_argument("--pred-len", type=int, choices=(96, 720), required=True)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--data-root", default=str(NSMT / "forecasting/dataset/ETT-small"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--backend", choices=("torch", "cupy"), default="cupy")
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--population", type=int, default=16)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--attn-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-train-batches", type=int, default=0, help="smoke only; zero uses full split")
    p.add_argument("--max-eval-batches", type=int, default=0, help="smoke only; zero uses full split")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1 or min(args.max_train_batches, args.max_eval_batches) < 0:
        raise ValueError("Invalid training budget")
    run(args)
