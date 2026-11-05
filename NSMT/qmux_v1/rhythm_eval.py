"""Stage 9 evaluation (MODI_PROJECT_LOG 2026-10-19).

predict()      runs the model in testing mode over an evaluation Dataset and returns per-unit probabilities: CinC 2017 windows
               are averaged per record (softmax mean); SHD samples are units themselves.
metrics()      CinC: challenge score = mean F1 over N, AF, Other (Noisy excluded), 4-class macro F1, accuracy, AF sensitivity /
               specificity, confusion; SHD: accuracy, macro F1, confusion.
final_test()   predict on the test set with the neuron firing-rate recorder, syops synaptic operations, and write
               final+result.csv / confusion_matrix.csv / neuron_energy.csv into out_dir.
"""
import csv
import os

import numpy as np
import torch
from spikingjelly.clock_driven import functional
from syops import get_model_complexity_info
from syops.ops import LIF_syops_counter_hook

from energy import FiringRateRecorder
from neurons import LIFNeuron, FLIFNeuron
from rhythm_data import loader
from utils import get_energy_consumption


def _logits(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def predict(model, ds, device, bs=32, recorder=None):
    model.eval()
    model.train_mode = 'testing'
    probs, units, labels = [], [], []
    functional.reset_net(model)
    for batch in loader(ds, bs, shuffle=False):
        x, y, u = batch
        x = x.float().to(device)
        z = _logits(model, x)
        if recorder is not None:
            recorder.add_samples(x.shape[0])
        functional.reset_net(model)
        probs.append(torch.softmax(z.float(), dim=1).cpu()); units.append(u); labels.append(y)
    p, u, y = torch.cat(probs).numpy(), torch.cat(units).numpy(), torch.cat(labels).numpy()
    uniq, inv = np.unique(u, return_inverse=True)
    P = np.zeros((len(uniq), p.shape[1])); np.add.at(P, inv, p)
    cnt = np.bincount(inv).astype(float)
    P /= cnt[:, None]
    Y = np.zeros(len(uniq), dtype=int); Y[inv] = y
    return P, Y, uniq


def confusion(y, pred, k):
    c = np.zeros((k, k), dtype=int)
    np.add.at(c, (y, pred), 1)
    return c


def f1s(c):
    out = []
    for i in range(c.shape[0]):
        tp, fp, fn = c[i, i], c[:, i].sum() - c[i, i], c[i, :].sum() - c[i, i]
        out.append(0. if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return np.array(out)


def metrics(P, Y, data):
    k = P.shape[1]
    pred = P.argmax(1)
    c = confusion(Y, pred, k)
    f = f1s(c)
    m = dict(acc=float((pred == Y).mean()), macro_f1=float(f.mean()), n=int(len(Y)))
    m.update({f'f1_{i}': float(f[i]) for i in range(k)})
    if str(data).upper() == 'CINC2017':
        m['score'] = float(f[:3].mean())                                # challenge score: N, AF, Other
        tp = c[1, 1]; fn = c[1].sum() - tp; fp = c[:, 1].sum() - tp; tn = c.sum() - tp - fn - fp
        m['af_sens'] = float(tp / max(1, tp + fn)); m['af_spec'] = float(tn / max(1, tn + fp))
    else:
        m['score'] = m['acc']
    return m, c


def final_test(args, model, ds, out_dir, tag='best'):
    os.makedirs(out_dir, exist_ok=True)
    rec = FiringRateRecorder(model)
    P, Y, units = predict(model, ds, args.device, bs=int(getattr(args, 'eval_bs', 32)), recorder=rec)
    rec.detach()
    m, c = metrics(P, Y, args.data)
    xb = ds[0][0]
    model.train_mode = 'testing'
    functional.reset_net(model)
    with open(os.path.join(out_dir, 'model+info+per+layer.txt'), 'w', encoding='utf-8') as fo:
        ops, params, fr = get_model_complexity_info(model=model, input_res=(tuple(xb.shape),), dataloader=loader(ds, int(getattr(args, 'eval_bs', 32)), False),
                                                    as_strings=False, print_per_layer_stat=True, verbose=False, ost=fo,
                                                    custom_modules_hooks={LIFNeuron: LIF_syops_counter_hook, FLIFNeuron: LIF_syops_counter_hook})
    syn_uJ = get_energy_consumption(O_ac=ops[1], O_mac=ops[2], unit='u')
    en = rec.summary(syn_energy_J=get_energy_consumption(O_ac=ops[1], O_mac=ops[2]) * 1e-12)
    rec.write_layers(os.path.join(out_dir, 'neuron_energy.csv'))
    with open(os.path.join(out_dir, 'confusion_matrix.csv'), 'w') as f:
        f.write('true\\pred,' + ','.join(str(i) for i in range(c.shape[0])) + '\n')
        for i in range(c.shape[0]):
            f.write(f'{i},' + ','.join(str(int(v)) for v in c[i]) + '\n')
    np.savez(os.path.join(out_dir, 'test_probs.npz'), P=P, Y=Y, units=units)
    keys = ['score', 'acc', 'macro_f1'] + [f'f1_{i}' for i in range(c.shape[0])] + (['af_sens', 'af_spec'] if 'af_sens' in m else []) + ['n']
    head = ['weights'] + keys + ['total_op_M', 'AC_op_M', 'MAC_op_M', 'params_M', 'syops_firing_rate', 'syn_energy_uJ'] + en['columns']
    row = [tag] + [f'{m[k]:.6f}' if isinstance(m[k], float) else str(m[k]) for k in keys] + \
          [f'{ops[0] / 1e6:.4f}', f'{ops[1] / 1e6:.4f}', f'{ops[2] / 1e6:.4f}', f'{params / 1e6:.6f}', f'{fr:.6f}', f'{syn_uJ:.6f}'] + en['values']
    with open(os.path.join(out_dir, 'final+result.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(head); w.writerow(row)
    print(f'[{tag}] test ' + ' '.join(f'{k}={m[k]:.4f}' for k in keys if isinstance(m[k], float)) + f' syn_uJ={syn_uJ:.4f}')
    return m
