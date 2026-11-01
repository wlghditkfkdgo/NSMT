# MODI 2026-11-01: --mitbih_val split check (sizes, stratification, determinism, no overlap between the two parts)
import sys, numpy as np, torch
sys.path.insert(0, '.')
from dataloader import create_loader
from utils import get_class_weights
D = '../../nsmt_base/classification/ECG/data/5class_mitbih_train_7_187.csv'
out = []
for rep in range(2):
    (tr, va), cns = create_loader(train=True, batch_size=512, num_classes=5, data_root=D, train_val_ratio=[4, 1], data_window_size=187, sampling='None', num_workers=0)
    ti, vi = np.array(tr.dataset.indices), np.array(va.dataset.indices)
    y = np.asarray(tr.dataset.dataset.y).reshape(-1)
    out.append((ti, vi))
    print('rep', rep, 'train', len(ti), 'val', len(vi), 'overlap', len(np.intersect1d(ti, vi)), 'union', len(np.union1d(ti, vi)))
    print('  train class counts', np.bincount(y[ti].astype(int), minlength=5), 'val', np.bincount(y[vi].astype(int), minlength=5))
    print('  sampler class_num_samples', cns.tolist(), 'CAL weights', [round(float(w), 4) for w in get_class_weights(cns)])
print('deterministic split:', np.array_equal(out[0][0], out[1][0]) and np.array_equal(out[0][1], out[1][1]))
full, cnf = create_loader(train=True, batch_size=512, num_classes=5, data_root=D, train_val_ratio=None, data_window_size=187, sampling='None', num_workers=0)
print('full training file', len(full.dataset), 'class_num_samples', cnf.tolist())
