"""SpikF energy under the syops library rule as published, input 96, all four ETT datasets x H 96 / 192 / 336 / 720 (2026-11-24,
exp/f-lif-pop-v3; PROJECT_LOG full-ETT-table pre-registration; user decision 11-24 "SpikF보다 energy가 낮아야해"). No training. The same
method as results/energy-common-20261123/count_spikf.py (read its docstring), extended from ETTh2 / ETTm2 x H720 / H96:
  H720            the trained checkpoint exp/run_ETT/SpikF_<ds>_ftM_sl96_pl720_lr0.0005_bs32/checkpoint.pth, loaded strictly; recomputed
                  test MSE / MAE and syops energy must reproduce the logged final+result.csv (fail closed: 1e-5, 0.01 uJ)
  H96 / 192 / 336 no trained SpikF at that horizon exists locally: SpikF built with that pred_len, the H720 checkpoint loaded except
                  dense2 (Linear(720 -> pred_len), random, seed 0). dense2 is counted as MAC whatever its weights, so the MAC count is
                  exact; the firing rates (AC count) come from the H720-trained encoder. Accuracy is not reported.
The SpikF clone (/home/yschoi/SpikF) is only read (python -B). One dataset per call so that the four can run side by side.
usage (from cinc_mixer_v1/): python -B results/ett-full16-20261124/count_spikf16.py --ds ETTh1
      -> results/ett-full16-20261124/spikf16/{spikf_syops_<ds>.csv, spikf_<ds>_H<pl>_model_info.txt}
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spikf16')
os.makedirs(OUT, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument('--spikf', default='/home/yschoi/SpikF')
ap.add_argument('--root', default='/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-'
                                  'learning/forecasting/dataset/ETT-small')
ap.add_argument('--ds', required=True, choices=('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'))
a = ap.parse_args()
sys.path.insert(0, a.spikf)
from models.SpikF import SpikF                                                      # noqa: E402
from data_provider.ETT_data_loader import Dataset_ETT_hour, Dataset_ETT_minute      # noqa: E402
from spikingjelly.clock_driven import functional                                    # noqa: E402
from syops import get_model_complexity_info                                         # noqa: E402

E_AC, E_MAC = 0.9, 4.6                                                              # pJ (SpikF utils.tools.get_energy_consumption)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def stop(m):
    print('STOP:', m); sys.exit(2)


def loader(ds, pl):
    D = Dataset_ETT_hour if ds.startswith('ETTh') else Dataset_ETT_minute
    s = D(root_path=a.root, data_path=f'{ds}.csv', flag='test', size=[96, 48, pl], features='M', target='OT', cols=None)
    return torch.utils.data.DataLoader(s, batch_size=1, shuffle=False, num_workers=1, drop_last=False)


def logged(ds):
    p = os.path.join(a.spikf, 'exp', 'ETT_results', f'SpikF_{ds}_ftM_sl96_pl720_lr0.0005_bs32', 'final+result.csv')
    v = [x.strip() for x in open(p).read().splitlines()[1].split(',')]                # header has 9 names, the row 8 values (no loss)
    return dict(mse=float(v[0]), mae=float(v[1]), AC_M=float(v[3].split()[0]), MAC_M=float(v[4].split()[0]),
                energy=float(v[7].split()[0]), path=p)


rows = []
for ds in (a.ds,):
    ck = os.path.join(a.spikf, 'exp', 'run_ETT', f'SpikF_{ds}_ftM_sl96_pl720_lr0.0005_bs32', 'checkpoint.pth')
    sd = torch.load(ck, map_location='cpu')
    for pl in (720, 96, 192, 336):
        torch.manual_seed(0)
        m = SpikF(96, 48, 32, 16, 1, 7, pl, 2.0, 2.0, 720)
        if pl == 720:
            m.load_state_dict(sd, strict=True)
        else:
            keep = {k: v for k, v in sd.items() if not k.startswith('dense2.')}
            miss = m.load_state_dict(keep, strict=False)
            if sorted(miss.missing_keys) != ['dense2.bias', 'dense2.weight'] or miss.unexpected_keys:
                stop(f'{ds} H{pl} partial load: {miss}')
        m = m.to(dev).eval()
        dl = loader(ds, pl)
        mse = mae = None
        if pl == 720:                                                               # SpikF test(): mean over the T outputs
            se = ae = n = 0.
            with torch.no_grad():
                for x, y in dl:
                    p = m(x.float().to(dev)).mean(dim=0).cpu().double()
                    t = y[:, -pl:, :].double()
                    se += float(((p - t) ** 2).sum()); ae += float((p - t).abs().sum()); n += t.numel()
                    functional.reset_net(m)
            mse, mae = se / n, ae / n
        functional.reset_net(m)
        x0 = next(iter(dl))[0]
        with open(os.path.join(OUT, f'spikf_{ds}_H{pl}_model_info.txt'), 'w', encoding='utf-8') as fo:   # syops needs the per-layer print
            ops, params, fr = get_model_complexity_info(model=m, input_res=(tuple(x0[0].shape),), dataloader=dl, as_strings=False,
                                                        print_per_layer_stat=True, verbose=False, ost=fo)
        e = (E_AC * ops[1] + E_MAC * ops[2]) * 1e-6
        r = dict(ds=ds, pl=pl, weights='trained H720 checkpoint' if pl == 720 else 'H720 checkpoint, dense2 random (shape only)',
                 mse='' if mse is None else f'{mse:.6f}', mae='' if mae is None else f'{mae:.6f}', AC_M=f'{ops[1] / 1e6:.4f}',
                 MAC_M=f'{ops[2] / 1e6:.4f}', E_AC_uJ=f'{E_AC * ops[1] * 1e-6:.4f}', E_MAC_uJ=f'{E_MAC * ops[2] * 1e-6:.4f}',
                 energy_uJ=f'{e:.4f}', params=int(params))
        if pl == 720:
            g = logged(ds)
            r.update(logged_mse=g['mse'], logged_mae=g['mae'], logged_energy_uJ=g['energy'])
            if abs(mse - g['mse']) > 1e-5 or abs(mae - g['mae']) > 1e-5 or abs(e - g['energy']) > 0.01 or not math.isfinite(e):
                stop(f'{ds} H720 does not reproduce {g["path"]}: mse {mse} vs {g["mse"]}, mae {mae} vs {g["mae"]}, energy {e} vs {g["energy"]}')
        rows.append(r)
        print(r, flush=True)
keys = list(rows[0].keys()) + [k for k in ('logged_mse', 'logged_mae', 'logged_energy_uJ') if k not in rows[0]]
with open(os.path.join(OUT, f'spikf_syops_{a.ds}.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
print(f'ALL SPIKF COUNTS DONE {a.ds}')
