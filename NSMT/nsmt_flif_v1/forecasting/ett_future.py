"""Second-stage evaluation on the ETT "future interval" (nsmt_flif_v1).

The v3 line (prereg 2O, 2026-09-27) opened ETTh rows [14400, 17420) once; the user allowed this line
to re-use them as a second-stage check (decision 10, 2026-10-14). ETTm rows [57600, 69680) are the
same months at 15-min resolution. Targets lie inside the interval; the preceding seq_len rows are
context. Scaling uses the train split of the standard ETT split (rows [0, 12 months)), exactly as
data_provider/data_loader.py does. Predictions are the T-mean of the head output, as test.py.

usage:
    python ett_future.py --run <run_dir> [--run ...] [--split future|test] [-nd 0] [--out DIR]
                         [--max_windows N] [--force] [--purpose TEXT]
--split test is a dry run on the standard test split that must reproduce the run's final+result.csv.
Records keep `target_rows` and `evaluated_rows` in separate fields. A registry file per
<data>-future-H<pl> refuses a second look at the same checkpoint unless --force.
"""
import os
import sys
import json
import hashlib
import argparse
from argparse import Namespace
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from spikingjelly.clock_driven import functional

from config import Config, set_random_seed
from model import LOAD_MODEL
from data_provider.data_factory import data_provider
from analysis import extract_pred_true_for_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
FUTURE_ROWS = {'ETTh1': (14400, 17420), 'ETTh2': (14400, 17420), 'ETTm1': (57600, 69680), 'ETTm2': (57600, 69680)}
SOURCES = ('ours.py', 'layers.py', 'neurons.py', 'model.py', 'ett_future.py', 'data_provider/data_loader.py')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def cond_name(cfg):
    if getattr(cfg, 'hippo_only', False):
        return 'hippo_only'
    nn_ = getattr(cfg, 'neo_neuron', 'sj_lif')
    name = 'base' if nn_ == 'sj_lif' else f"neo_{nn_}_{getattr(cfg, 'neo_where', 'both')}"
    lead, ng = getattr(cfg, 'aux_lead', 0), getattr(cfg, 'neo_grad', 0.05)
    return name if (lead == 0 and ng == 0.05) else f'{name}@L{lead}g{ng}'      # stage-2 regime suffix


class FutureWindows(Dataset):
    """Windows whose targets lie in [R0, R1); context = the seq_len rows before the target."""
    def __init__(self, root_path, data_path, data, seq_len, pred_len):
        df = pd.read_csv(os.path.join(root_path, data_path))
        self.csv_sha = sha256(os.path.join(root_path, data_path))
        f = 4 if data.startswith('ETTm') else 1
        self.border = 12 * 30 * 24 * f                              # train split of the standard ETT split
        self.R0, self.R1 = FUTURE_ROWS[data]
        if len(df) != self.R1:
            raise ValueError(f'{data}: expected {self.R1} rows, got {len(df)}')
        values = df[df.columns[1:]].values                           # features M
        scaler = StandardScaler().fit(values[0:self.border])
        self.values = scaler.transform(values).astype(np.float32)
        self.seq_len, self.pred_len = seq_len, pred_len
        self.n = self.R1 - self.R0 - pred_len + 1

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = self.R0 + i
        return (torch.from_numpy(self.values[s - self.seq_len:s]), torch.from_numpy(self.values[s:s + self.pred_len]),
                torch.zeros(1), torch.zeros(1))


WEIGHTS = {'last': 'best+model.pt', 'best': 'checkpoint.pth'}      # paper naming: best+model.pt holds the LAST epoch


def load_run(run_dir, gpu, weights='last'):
    cfg = Config()
    cfg.load_args(run_dir, Namespace(saved_epoch=[1], num_device=gpu))
    set_random_seed(cfg.seed)
    model = LOAD_MODEL['myModel'](cfg, train=False, ckpt=WEIGHTS[weights])
    model.eval()
    return cfg, model


@torch.no_grad()
def evaluate(model, loader, cfg, max_windows=None):
    mse, mae, crit, loss, n = MeanSquaredError(), MeanAbsoluteError(), nn.L1Loss(), 0., 0
    for x, y, _, _ in loader:
        x, y = x.float().to(cfg.device), y.float().to(cfg.device)
        functional.reset_net(model)
        out = model(x)
        p, t, _, _ = extract_pred_true_for_metrics(out, y, cfg.pred_len, long_pred_len=cfg.pred_len, reduce_T='mean')
        loss += crit(p, t).item() * x.shape[0]; n += x.shape[0]
        mse.update(p.cpu().contiguous(), t.cpu().contiguous()); mae.update(p.cpu().contiguous(), t.cpu().contiguous())
        if max_windows and n >= max_windows:
            break
    return {'mse': mse.compute().item(), 'mae': mae.compute().item(), 'loss': loss / max(1, n), 'windows': n}


def registry_check(reg_path, ckpt_sha, record_path, purpose, force):
    """One look per trained model. Creates the registry on first use."""
    now = datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
    if not os.path.exists(reg_path):
        os.makedirs(os.path.dirname(reg_path), exist_ok=True)
        with open(reg_path, 'x') as f:
            json.dump({'split': os.path.basename(reg_path)[:-5], 'purpose': purpose, 'opened_at': now,
                       'user': os.environ.get('USER', '?'), 'looks': []}, f, indent=2)
    reg = json.load(open(reg_path))
    seen = [l for l in reg['looks'] if l['checkpoint_sha256'] == ckpt_sha]
    if seen and not force:
        raise SystemExit(f'REFUSED: checkpoint {ckpt_sha[:12]} already evaluated on {reg["split"]} '
                         f'({seen[0]["record"]}); pass --force to log an override')
    reg['looks'].append({'checkpoint_sha256': ckpt_sha, 'record': record_path, 'opened_at': now, 'forced': bool(seen)})
    json.dump(reg, open(reg_path, 'w'), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True)
    ap.add_argument('--split', choices=['future', 'test'], default='future')
    ap.add_argument('-nd', type=int, default=0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--max_windows', type=int, default=None)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--weights', choices=['last', 'best'], default='last', help='last-epoch or best-validation weights')
    ap.add_argument('--purpose', default='nsmt_flif_v1 second-stage check (decision 10, 2026-10-14)')
    ap.add_argument('--registry_dir', default=os.path.join(HERE, 'results', 'split_openings'), help='tests may point this at a scratch dir')
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, 'results', f"future-{datetime.now().strftime('%Y%m%d')}")
    os.makedirs(out, exist_ok=True)
    for run in a.run:
        run = os.path.abspath(run)
        cfg, model = load_run(run, a.nd, a.weights)
        if a.split == 'test':
            _, loader = data_provider(cfg, flag='test')
            ds = None
        else:
            ds = FutureWindows(cfg.root_path, cfg.data_path, cfg.data, cfg.seq_len, cfg.pred_len)
            loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
        res = evaluate(model, loader, cfg, a.max_windows)
        ckpt = os.path.join(run, 'model_state', WEIGHTS[a.weights])
        name = f"{cfg.data}_H{cfg.pred_len}_{cond_name(cfg)}_seed{cfg.seed}_{a.split}" + ('_bestval' if a.weights == 'best' else '')
        rec_path = os.path.join(out, name + '.json')
        rec = {'split': a.split, 'data': cfg.data, 'pred_len': cfg.pred_len, 'seq_len': cfg.seq_len,
               'cond': cond_name(cfg), 'seed': cfg.seed, 'run_dir': run, 'weights': a.weights, 'checkpoint_file': WEIGHTS[a.weights], **res,
               'target_rows': None if ds is None else [ds.R0, ds.R1],
               'evaluated_rows': None if ds is None else [ds.R0 - cfg.seq_len, ds.R1],
               'scaler_rows': None if ds is None else [0, ds.border],
               'checkpoint_sha256': sha256(ckpt), 'config_sha256': sha256(os.path.join(run, 'model_state', 'config.pt')),
               'data_sha256': ds.csv_sha if ds is not None else None,
               'source_sha256': {s: sha256(os.path.join(HERE, s)) for s in SOURCES},
               'torch': torch.__version__, 'device': str(cfg.device),
               'neo': {k: getattr(cfg, k, None) for k in ('neo_neuron', 'neo_where', 'neo_tau', 'neo_alpha', 'fde_method', 'fde_step', 'fde_memory', 'hippo_only', 'aux_lead', 'neo_grad')},
               'timestamp': datetime.now(ZoneInfo('Asia/Seoul')).isoformat()}
        if a.split == 'future' and not a.max_windows:
            registry_check(os.path.join(a.registry_dir, f'{cfg.data}-future-H{cfg.pred_len}.json'),
                           rec['checkpoint_sha256'], rec_path, a.purpose, a.force)
        json.dump(rec, open(rec_path, 'w'), indent=2)
        line = f"{a.split},{cfg.data},{cfg.pred_len},{cond_name(cfg)},{cfg.seed},{res['mse']:.6f},{res['mae']:.6f},{res['windows']},{rec['checkpoint_sha256'][:12]},{run}"
        with open(os.path.join(out, 'future.csv'), 'a') as f:
            if f.tell() == 0:
                f.write('split,data,pred_len,cond,seed,mse,mae,windows,ckpt,run_dir\n')
            f.write(line + '\n')
        print('FUTURE', line)


if __name__ == '__main__':
    main()
