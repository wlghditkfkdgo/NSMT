"""Shared pieces of prereg 2T: the 2J-2Q protocols rerun with the spikeDE-based neurons (neurons.py).

The earlier evaluators (ett_test.py, ett_future.py, ett_alpha.py, ett_readout.py, analysis/hard_*.py)
stay as they were run. They read layers.PopulationNeuron internals (neuron.selector, reseed_hard,
hard_mask) and legacy run names, so the 2T evaluators (ett_rerun.py, analysis/recall_rerun.py) share
this module instead:

  conditions      the new-neuron settings of every rerun condition and the registered manifest
  find_cell       one run by its exact run_id: 'done', a recorded G11 failure, or fatal
  check_manifest  every registered field of a finished run, fail closed (empty list = OK)
  one_pass        the whole split once; error, per-channel error, safety, kept mass, firing rate,
                  M_eff (recall) and top-k ties all come from ONE forward per batch
  legacy          the earlier records (fixed SHA256), their per-seed values and their statistics
  paired/category the paired t statistics and the verdict category under each original rule

    python rerun_common.py freeze-calibration <suite> [<calibration dir>]   fix the 2T calibration files (name + SHA256)
"""
import sys
import glob
import json
import math
import hashlib
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from config import Config, parse_defaults, TASK, check_neuron_args, neuron_variant, block_tag
from model import LOAD_MODEL
from test import EVENT_KINDS, kind_mask
import layers
import neurons

SEEDS = (7, 13, 21, 42, 123, 256, 512, 1024)
TORCH = '1.12.0+cu113'
MDE = .005
REPRO_TOL = 1e-6                    # re-measured validation MSE vs the trainer's best_val_loss (2P D-BU)
TWICE_TOL = 1e-12                   # recall: two passes over the same split must agree (2L D-AU)
SOURCES = ('neurons.py', 'spikede.py', 'layers.py', 'ours.py', 'model.py', 'train.py', 'test.py', 'config.py',
           'calibrate.py', 'rerun_common.py', 'ett_rerun.py', 'data_provider/data_loader.py',
           'data_provider/synthetic.py', 'data_provider/data_factory.py')
DATA_ROOT = (TASK.parents[1] / 'forecasting' / 'dataset' / 'ETT-small').resolve()
DATA_SHA256 = {'ETTh1': 'f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066',
               'ETTh2': 'a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b'}
TARGET_ROWS = {'val': [8640, 11520], 'test': [11520, 14400], 'future': [14400, 17420]}   # A34: own field

# ---- conditions ---------------------------------------------------------------------------------------
NEURON_FIXED = {'num_population': 4, 'tau': [4., 8., 16., 32.], 'heterogeneous': True, 'tau_s': 2.,
                'threshold': 1., 'surrogate_scale': 5., 'fde_method': 'pred', 'fde_step': 1., 'fde_memory': 0,
                'input_norm': 'frozen', 'readout': 'spike', 'model': 'myModel', 'soma': 'lif'}
Q1 = {'neuron': 'pop_flif', 'mode': 'full', 'hard_q': 1.}
SEL = {'neuron': 'sel_pop_flif', 'mode': 'hard', 'hard_axis': 'shared', 'select_part': 'all'}
PEARSON = {**SEL, 'hard_stat': 'pearson', 'hard_q': .5}
ETT_CONDITIONS = {
    'q1':            {**Q1, 'revin': False, 'alpha': .7, 'head_mode': 'flatten'},
    'pearson':       {**PEARSON, 'revin': False, 'alpha': .7, 'head_mode': 'flatten'},
    'q1_R':          {**Q1, 'revin': True, 'alpha': .7, 'head_mode': 'flatten'},
    'pearson_R':     {**PEARSON, 'revin': True, 'alpha': .7, 'head_mode': 'flatten'},
    'q1_R_a1':       {**Q1, 'revin': True, 'alpha': 1., 'head_mode': 'flatten'},
    'pearson_R_a1':  {**PEARSON, 'revin': True, 'alpha': 1., 'head_mode': 'flatten'},
    'q1_R_lin':      {**Q1, 'revin': True, 'alpha': .7, 'head_mode': 'linear'},
    'pearson_R_lin': {**PEARSON, 'revin': True, 'alpha': .7, 'head_mode': 'linear'},
}
# prereg 2U: population axis kept (members fire, no soma), Linear + per-step BN (gamma from D11) before the
# population f-LIF, a spiking hidden block (Linear + BN + SpikingJelly LIF, 32 units), one output Linear.
U_BLOCK = {'soma': 'none', 'input_norm': 'bn', 'hidden_dim': 32, 'hidden_tau': 2., 'head_mode': 'linear'}
ETT_CONDITIONS_2U = {c: {**ETT_CONDITIONS[c], **U_BLOCK}
                     for c in ('q1', 'pearson', 'q1_R', 'pearson_R', 'q1_R_a1', 'pearson_R_a1')}
CONDITIONS = {'2T': ETT_CONDITIONS, '2U': ETT_CONDITIONS_2U}


# prereg 2V: the 2T -> 2U changes taken apart, 2 x 2 x 2 on validation: soma (lif | none) x input norm
# (frozen standardisation + fixed scale | per-step BN with gamma from D11) x head (flatten: Linear(D*,32) then
# Linear(1344,96), as 2T | hidden: Linear + BN + SpikingJelly LIF then one Linear, as 2U). Cell names
# '<base>-<soma>-<norm>-<head>', base q1_R or q1. lif-frozen-flatten is 2T's run, none-bn-hidden is 2U's.
def cell_2v(name):
    base, soma, norm, head = name.split('-')
    spec = {**ETT_CONDITIONS[base], 'soma': soma, 'input_norm': norm}
    if head == 'hidden':
        spec.update(hidden_dim=32, hidden_tau=2., head_mode='linear')
    elif head == 'flatten':
        spec.update(hidden_dim=0, head_mode='flatten')
    else:
        raise ValueError(f'head {head!r}')
    return spec


def cell_2w(name):
    """Prereg 2W: '<base>-<arch>-<lr>-<wd>' on the 2U base (population axis kept, per-step BN with gamma from D11).
    arch direct (no hidden block; one Linear on [B*C, T*128]) | hmem (hidden block read through its membrane);
    lr 0.0005 | 0.001 | 0.002; wd all (AdamW decays everything, as before) | xnb (no decay on BN and biases)."""
    base, arch, lr, wd = name.split('-', 3)
    if arch not in ('direct', 'hmem') or wd not in ('all', 'xnb'):
        raise ValueError(f'2W cell {name!r}')
    spec = {**ETT_CONDITIONS[base], 'soma': 'none', 'input_norm': 'bn', 'lr': float(lr),
            'wd_exclude_norm_bias': wd == 'xnb', 'head_mode': 'linear'}
    if arch == 'hmem':
        spec.update(hidden_dim=32, hidden_tau=2., hidden_readout='membrane')
    else:
        spec.update(hidden_dim=0)
    return spec


GRID_2W = [f'{a}-{lr}-{wd}' for a in ('direct', 'hmem') for lr in ('0.0005', '0.001', '0.002') for wd in ('all', 'xnb')]


def conditions_2w(cfg):
    """The ETT conditions of the selected 2W setting, for ett_rerun.py --appendix 2W."""
    return {base: cell_2w(f'{base}-{cfg}') for base in ('q1', 'pearson', 'q1_R', 'pearson_R')}


def cell_2x(name):
    """Prereg 2X: '<base>-p<P>s<S>' -- the 2W pick (direct-0.0005-all) with patch length P and stride S.
    p8s8 is the 2W pick itself (patch_stride 0 = the default it was trained with)."""
    base, patch = name.split('-')
    size, stride = (int(v) for v in patch[1:].split('s'))
    spec = cell_2w(f'{base}-direct-0.0005-all')
    if (size, stride) != (8, 8):
        spec.update(patch_size=size, patch_stride=stride)
    return spec


CELLS_2V = [f'{b}-{s}-{n}-{h}' for b in ('q1_R', 'q1') for s in ('lif', 'none') for n in ('frozen', 'bn')
            for h in ('flatten', 'hidden')]
SOURCE_2V = {   # (soma, norm, head) -> (training suite, stdout condition label of base, calibration suite)
    ('lif', 'frozen', 'flatten'): ('etthard-spikede-20261021', '{base}', 'rerun-cal-20261021'),
    ('none', 'bn', 'hidden'): ('etthard-2u-20261021', '{base}', 'rerun2u-cal-20261021'),
    ('lif', 'bn', 'flatten'): ('etthard-2v-20261022', '{cell}', 'rerun2v-cal-lifbn-20261022'),
    ('lif', 'bn', 'hidden'): ('etthard-2v-20261022', '{cell}', 'rerun2v-cal-lifbn-20261022'),
    ('lif', 'frozen', 'hidden'): ('etthard-2v-20261022', '{cell}', 'rerun-cal-20261021'),
    ('none', 'frozen', 'flatten'): ('etthard-2v-20261022', '{cell}', 'rerun2v-cal-nonefrozen-20261022'),
    ('none', 'frozen', 'hidden'): ('etthard-2v-20261022', '{cell}', 'rerun2v-cal-nonefrozen-20261022'),
    ('none', 'bn', 'flatten'): ('etthard-2v-20261022', '{cell}', 'rerun2u-cal-20261021'),
}
# run_ett.sh / ett_test.REGISTERED: the O8 budget and the shape; alpha and head_mode come from the condition
ETT_REGISTERED = {'task': 'ett', 'seq_len': 336, 'patch_size': 8, 'embed_dim': 32, 'head_dim': 32, 'epoch': 50,
                  'batch_size': 128, 'lr': 1e-3, 'weight_decay': .01, 'patience': 10, 'scheduler_patience': 5,
                  'num_workers': 0, 'max_train_batches': 0, 'max_eval_batches': 0, 'pred_len': 96, **NEURON_FIXED}
# hard_*_train.sh and the 2L manifest: the recall budget, revision r2
RECALL_REGISTERED = {'task': 'recall', 'data': 'recall', 'n_train': 2048, 'n_val': 256, 'n_test': 256,
                     'n_confirm': 1000, 'n_keys': 3, 'run_range': [2, 5], 'min_gap': 1, 'cue_mode': 'onehot',
                     'cue_noise': 0., 'data_seed': 20260921, 'epoch': 12, 'batch_size': 64, 'lr': 1e-3,
                     'weight_decay': .01, 'seq_len': 336, 'patch_size': 8, 'embed_dim': 32, 'head_dim': 32,
                     'head_mode': 'flatten', 'alpha': .7, 'revin': False, 'num_workers': 0,
                     'max_train_batches': 0, 'max_eval_batches': 0, 'cpu': True, **NEURON_FIXED}


def recall_spec(stat, q):
    """q = 1 is the plain population f-LIF (no Selective); q < 1 adds the selection with `stat`."""
    return dict(Q1) if float(q) >= 1. else {**SEL, 'hard_stat': stat, 'hard_q': float(q)}


def recall_label(stat, q):
    return 'q1' if float(q) >= 1. else f'{stat}_q{float(q):g}'


def calibration_label(task, cond=None, data=None, stat=None, q=None):
    """The rerun_calibrate.sh label of the file a condition trains with (2Q B shares 2O's)."""
    if task == 'recall':
        return 'recall_q1' if float(q) >= 1. else f'recall_{stat}_q{float(q):g}'
    spec = ETT_CONDITIONS[cond]
    regime = 'Roff' if not spec['revin'] else ('R_a1' if spec['alpha'] == 1. else 'R')
    return f"{data}_{regime}_{'q1' if spec['neuron'] == 'pop_flif' else 'pearson'}"


def variant_of(task, spec):
    """config.Config's variant for this condition (the run directory and run_id carry it)."""
    ns = parse_defaults()
    for key, value in {**NEURON_FIXED, **spec}.items():
        setattr(ns, key, value)
    ns.task = task
    check_neuron_args(ns)
    return (neuron_variant(ns) + block_tag(ns) + ('_revin' if spec.get('revin') else '')
            + ('_k3' if task == 'recall' else ''))


def run_id(task, dataset, spec, seed, pred_len=96):
    return (f"myModel_{task}_{dataset}_p{pred_len}_{spec['head_mode']}_a{float(spec['alpha'])}_spike_"
            f"{variant_of(task, spec)}_seed{seed}")


def run_dirs(suite, task, dataset, spec, seed):
    return glob.glob(str(TASK / 'log' / suite / dataset / '*' / f"*+model+myModel+*+alpha+{float(spec['alpha'])}+*"
                         / f"seed{seed}_{spec['head_mode']}_spike_{variant_of(task, spec)}"))


def refused_cell(suite, task, dataset, spec, seed, calibration):
    """A configuration whose calibration was refused is not trained (2T D-CN): the cell is blocked,
    and a run directory for it would be an error (fail closed)."""
    stray = run_dirs(suite, task, dataset, spec, seed)
    bad = [f'calibration refused but {len(stray)} run directories exist'] if stray else []
    return bad, {'status': 'calibration_refused', 'calibration': calibration['file'], 'reason': calibration['reason']}


def find_cell(suite, task, dataset, spec, seed, pred_len=96):
    """('done', run dir, result json) or ('g11', run dir, None); anything else is fatal (2P D-BX).
    A G11 stop can come after an epoch that saved a checkpoint, so only the G11 record and the
    missing result decide 'g11'."""
    variant = variant_of(task, spec)
    runs = run_dirs(suite, task, dataset, spec, seed)
    result = TASK / 'results' / suite / f'{run_id(task, dataset, spec, seed, pred_len)}.json'
    if len(runs) != 1:
        raise SystemExit(f'[rerun] {suite} {dataset} {variant} seed {seed}: {len(runs)} run directories')
    run = Path(runs[0])
    has_g11 = (run / 'G11_violation.json').exists()
    if result.exists() and not has_g11 and (run / 'model_state' / 'best+model.pt').exists():
        return 'done', str(run), str(result)
    if has_g11 and not result.exists():
        return 'g11', str(run), None
    raise SystemExit(f'[rerun] {suite} {dataset} {variant} seed {seed}: result {result.exists()}, '
                     f'G11 record {has_g11} -- neither a finished run nor a recorded G11 failure')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(run, device):
    base = parse_defaults()
    base.cpu = device.type == 'cpu'
    base.num_device = 0 if device.type == 'cpu' else device.index
    args = Config()
    args.load_args(run, base)
    args.device = device
    model = LOAD_MODEL[args.model](args, train=False)
    return model, args


# ---- calibration ---------------------------------------------------------------------------------------
def freeze_calibration(suite, caldir='calibration'):
    """results/<suite>/calibration_manifest.json from rerun_calibrate.sh's CALIB lines: each label's
    file, its SHA256 and what it fixed. Refused configurations are entered as refused. Created with 'x'."""
    raw = TASK / 'results' / suite / 'calibration_raw.txt'
    entries = {}
    for line in raw.read_text().splitlines():
        if not line.startswith('CALIB '):
            continue
        fields = dict(f.split('=', 1) for f in line.split()[1:])
        label, name = fields['label'], fields['file']
        if label in entries:
            raise SystemExit(f'[calibration] {label} appears twice in {raw}')
        path = TASK / 'results' / caldir / name
        if name == 'none' or not path.exists():
            raise SystemExit(f'[calibration] {label}: no calibration file written ({line})')
        payload = json.loads(path.read_text())
        picked = payload.get('picked')
        entries[label] = {'file': name, 'sha256': sha256(path), 'rc': int(fields['rc']),
                          'refused': picked is None, 'reason': payload.get('reason'),
                          'input_scale': None if picked is None else picked['input_scale'],
                          'declared_bound': None if picked is None else picked['declared_bound'],
                          'firing_rate': None if picked is None else picked['firing_rate'],
                          'requested_at_init': payload.get('requested_at_init'),
                          'requested_neuron': payload.get('requested_neuron'),
                          'neuron': payload.get('neuron')}
        if (int(fields['rc']) == 0) == entries[label]['refused']:
            raise SystemExit(f'[calibration] {label}: rc {fields["rc"]} disagrees with picked {picked is not None}')
    out = TASK / 'results' / suite / 'calibration_manifest.json'
    with open(out, 'x') as handle:
        json.dump({'prereg': '2T', 'suite': suite, 'calibration_dir': caldir, 'entries': entries}, handle, indent=2,
                  allow_nan=False)
    for label, e in entries.items():
        state = 'REFUSED' if e['refused'] else f"input_scale {e['input_scale']}, bound {e['declared_bound']:.3f}, " \
                                               f"rate {e['firing_rate']:.4f}"
        print(f'[calibration] {label:<24} {e["file"]}  {state}')
    print(f'[calibration] fixed {len(entries)} files in {out}')
    return out


def calibration_manifest(suite):
    """label -> entry, after re-hashing every file (fail closed)."""
    manifest = json.loads((TASK / 'results' / suite / 'calibration_manifest.json').read_text())
    entries, caldir = manifest['entries'], manifest['calibration_dir']
    for label, e in entries.items():
        if sha256(TASK / 'results' / caldir / e['file']) != e['sha256']:
            raise SystemExit(f'[calibration] {label}: {e["file"]} changed since it was fixed')
    return entries


# ---- manifest ------------------------------------------------------------------------------------------
def check_manifest(args, result, task, spec, seed, calibration, data=None, cond=None):
    """Every registered field of a finished run; returns the mismatches (empty = OK)."""
    registered = ETT_REGISTERED if task == 'ett' else RECALL_REGISTERED
    want = {**registered, **spec, 'seed': seed, **({'data': data} if task == 'ett' else {})}
    bad = [f'{k}: {getattr(args, k, None)!r} != {v!r}' for k, v in want.items() if getattr(args, k, None) != v]
    if calibration['refused']:
        bad.append(f"its calibration {calibration['file']} was refused; the condition should not have trained")
    if getattr(args, 'g11_bound', None) != calibration['declared_bound']:
        bad.append(f"g11_bound {getattr(args, 'g11_bound', None)!r} != {calibration['declared_bound']!r}")
    if getattr(args, 'input_scale', None) != calibration['input_scale']:
        bad.append(f"input_scale {getattr(args, 'input_scale', None)!r} != {calibration['input_scale']!r}")
    if getattr(args, 'calibration_file', None) != calibration['file']:
        bad.append(f"calibration_file {getattr(args, 'calibration_file', None)!r} != {calibration['file']!r}")
    if task == 'ett':
        if getattr(args, 'data_path', None) != f'{data}.csv':
            bad.append(f"data_path {getattr(args, 'data_path', None)!r} != '{data}.csv'")
        if Path(getattr(args, 'root_path', '')).resolve() != DATA_ROOT:
            bad.append(f"root_path {getattr(args, 'root_path', None)!r} != {str(DATA_ROOT)!r}")
    payload = json.loads(Path(result).read_text())
    train, provenance = payload.get('train', {}), payload.get('provenance', {})
    epochs = train.get('epochs_run') or 0
    if task == 'recall' and epochs != registered['epoch']:
        bad.append(f'epochs_run {epochs!r} != {registered["epoch"]}')
    if task == 'ett' and not 1 <= epochs <= registered['epoch']:
        bad.append(f'epochs_run {epochs!r} outside 1..{registered["epoch"]}')
    if payload.get('test') is not None or payload.get('test_skipped') is not True:
        bad.append('test was not skipped in training')
    if provenance.get('checkpoint_sha256') != sha256(Path(args.save_model_state_path) / 'best+model.pt'):
        bad.append('checkpoint sha256 differs from the one the training run recorded')
    device = str(provenance.get('device', ''))
    if task == 'ett' and not device.startswith('cuda'):
        bad.append(f'trained on {device!r}, not CUDA (the declared tie rule is CUDA topk)')
    if task == 'recall' and device != 'cpu':
        bad.append(f'trained on {device!r}, not CPU (the recall protocol)')
    if provenance.get('torch') != TORCH:
        bad.append(f"trained with torch {provenance.get('torch')!r} != {TORCH}")
    if (provenance.get('calibration') or {}).get('file') != calibration['file']:
        bad.append(f"provenance calibration {(provenance.get('calibration') or {}).get('file')!r} != {calibration['file']!r}")
    if task == 'ett':
        from ett_test import stopping_evidence                     # unchanged 2N helper (CSV, stdout, recount)
        bad += stopping_evidence(args, train, cond, data, args.pred_len, seed)
    return bad


def check_failure(run, task, spec, seed, calibration):
    """A recorded G11 stop: the record and the saved config must belong to this cell."""
    saved = torch.load(Path(run) / 'model_state' / 'config.pt', map_location='cpu')
    record = json.loads((Path(run) / 'G11_violation.json').read_text())
    bad = [f'config {k}: {saved.get(k)!r} != {v!r}' for k, v in {**spec, 'seed': seed}.items() if saved.get(k) != v]
    if record.get('run_id') != saved.get('run_id'):
        bad.append(f"G11 record run_id {record.get('run_id')!r} != config {saved.get('run_id')!r}")
    if record.get('bound') != calibration['declared_bound'] or record.get('calibration_file') != calibration['file']:
        bad.append(f"G11 record bound/calibration {record.get('bound')!r}/{record.get('calibration_file')!r}")
    if record.get('neuron') != spec['neuron']:
        bad.append(f"G11 record neuron {record.get('neuron')!r} != {spec['neuron']!r}")
    if not record.get('max_abs_state', 0.) >= record.get('bound', float('inf')):
        bad.append('G11 record does not show the bound reached')
    return bad, record


# ---- one pass ------------------------------------------------------------------------------------------
def selection_of(neuron):
    """(q, stat, axis, mask function) for the new Selective or the legacy Selector; None = no selection."""
    sel = getattr(neuron, 'selective', None)
    if sel is not None:
        return sel.q, sel.stat, sel.axis, lambda xi, hist: neurons.select_mask(xi, hist, sel.q, sel.stat, sel.axis)
    legacy = getattr(neuron, 'selector', None)
    if legacy is not None and getattr(legacy, 'hard_q', 1.) < 1.:
        return legacy.hard_q, legacy.hard_stat, legacy.hard_axis, legacy.hard_mask
    return None


@torch.no_grad()
def one_pass(model, loader, args, ties=True):
    """The whole split, one forward per batch. Error and every state come from the same forward, so
    the safety numbers belong to the trajectory the error was measured on (2L D-AU). The random
    control's generator must be reseeded by the caller before the pass (D-AP)."""
    from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
    from ours import window_norm
    model.eval()
    device = args.device
    recall = args.task == 'recall'
    spiking = args.model == 'myModel'
    neuron = model.embedding.neuron if spiking else None
    selection = selection_of(neuron) if spiking else None
    count_ties = ties and selection is not None and selection[1] == 'pearson'
    mse, mae = MeanSquaredError().to(device), MeanAbsoluteError().to(device)
    sums = {name: torch.zeros(3, dtype=torch.float64, device=device) for name in EVENT_KINDS}
    sse, per_n, windows = None, 0, 0
    peak, finite, nonfinite = 0., True, 0
    kept, kept_n, spikes, spike_n = 0., 0, 0., 0
    hidden, hidden_n = 0., 0                                    # 2U: the spiking hidden block
    meff_seq, n_ties, tie_n, mismatch = [], 0, 0, 0
    for x, y, truth, kind in loader:
        x, y = x.float().to(device), y.float().to(device)
        output, aux = model(x, mode=args.mode, return_aux=True)
        windows += x.shape[0]
        if recall:
            error = (output - y).double()                       # test.evaluate, the same sums
            kind = kind.to(device)
            for name, want in EVENT_KINDS.items():
                m = kind_mask(kind, want)
                sums[name][0] += (error * m).square().sum()
                sums[name][1] += (error.abs() * m).sum()
                sums[name][2] += m.sum()
        else:
            mse.update(output.contiguous(), y.contiguous())     # ett_test / ett_alpha, the same metric
            mae.update(output.contiguous(), y.contiguous())
            err = (output.double() - y.double()).square().sum(dim=(0, 1))
            sse = err if sse is None else sse + err
            per_n += output.shape[0] * output.shape[1]
        if not spiking:
            if not torch.isfinite(output).all():
                finite, nonfinite = False, nonfinite + 1
            continue
        state = aux['state']                                    # [T, B*C, D, K]; state[t] = u_{t+1}
        if not (torch.isfinite(state).all() and torch.isfinite(output).all()):
            finite, nonfinite = False, nonfinite + 1            # never folded into the maximum
        else:
            peak = max(peak, state.abs().max().item())
        spikes += aux['spikes'].double().mean(dim=(0, 2)).sum().item()
        spike_n += aux['spikes'].shape[1]
        if 'hidden_spikes' in aux:
            hidden += aux['hidden_spikes'].double().mean(dim=(0, 2)).sum().item()
            hidden_n += aux['hidden_spikes'].shape[1]
        b = neuron.b
        rows = state.shape[1]
        share = torch.zeros(rows, dtype=torch.float64, device=device)
        tot, cnt = torch.zeros(rows, dtype=torch.float64, device=device), torch.zeros(rows, dtype=torch.float64, device=device)
        for n in range(1, len(aux['coeff'])):
            c = aux['coeff'][n].double()                        # [B*C, D, n] = m * b
            bh = b[1:n + 1].flip(0).double()
            share += (c.sum(-1) / bh.sum()).mean(-1)
            if recall:                                          # D-Q: the answer slots' share of the kept kernel
                answer = truth[:, n, :n].to(device)
                valid = (kind[:, n] > 0) & answer.any(-1)
                part = ((c * answer.unsqueeze(1)).sum(-1) / c.sum(-1).clamp_min(1e-12)).mean(-1)
                tot += torch.where(valid, part, torch.zeros_like(part))
                cnt += valid.double()
        kept += (share / (len(aux['coeff']) - 1)).sum().item()
        kept_n += rows
        if recall:
            meff_seq.extend((tot[cnt > 0] / cnt[cnt > 0]).tolist())
        if count_ties:                                          # the trained model's top-k boundary ties
            q, _, _, mask_fn = selection
            xin = window_norm(x)[0] if getattr(args, 'revin', False) else x
            current = model.embedding.current(layers.to_patches(xin, args.patch_size, getattr(args, 'patch_stride', 0)))
            zero = torch.zeros_like(state[0])
            for n in range(1, current.shape[0]):
                xi = torch.cat([state[n - 1], current[n].unsqueeze(-1)], dim=-1)
                hist = torch.stack([torch.cat([state[j - 1] if j else zero, current[j].unsqueeze(-1)], -1)
                                    for j in range(n)], dim=-2)
                m, sim = mask_fn(xi, hist)
                mismatch += int(not torch.equal(m, (aux['coeff'][n] > 0).to(m.dtype)))   # rebuilt == the model's
                k = max(1, int(round(q * n)))
                if k < n:
                    ordered = sim[:, 0, :].sort(dim=-1, descending=True).values
                    n_ties += int((ordered[:, k - 1] == ordered[:, k]).sum())
                tie_n += xi.shape[0]
    bound = getattr(args, 'g11_bound', None)
    out = {'windows': windows, 'finite': finite, 'nonfinite_batches': nonfinite,
           'max_abs_state': peak if spiking else None, 'bound': bound if spiking else None,
           'safe': finite and (not spiking or (bound is not None and peak < bound)),
           'kept_mass_frac': kept / kept_n if kept_n else None,
           'firing_rate': spikes / spike_n if spike_n else None,
           'hidden_firing_rate': hidden / hidden_n if hidden_n else None,
           'ties': [n_ties, tie_n] if count_ties else None, 'mask_mismatch': mismatch if count_ties else None}
    if recall:
        res = {name: (t[0] / t[2]).item() for name, t in sums.items() if t[2] > 0}
        out.update(recall_mse=res['recall'], copy_mse=res['copy'], recall_first_mse=res['recall_first'],
                   all_mse=res['all'], m_eff=float(np.mean(meff_seq)) if meff_seq else None)
        finite_err = all(math.isfinite(v) for v in res.values())
    else:
        out.update(mse=mse.compute().item(), mae=mae.compute().item(), mse_per_channel=(sse / per_n).tolist())
        finite_err = math.isfinite(out['mse']) and math.isfinite(out['mae'])
    if not finite_err:
        out['finite'], out['safe'] = False, False
    return out


# ---- statistics and verdict categories -------------------------------------------------------------------
def paired(a, b, level=.95, relative=True):
    """Paired t statistics of a - b over the seeds; refuses NaN / missing values (fail closed)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.size < 2 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError(f'paired: need two finite vectors of the same length >= 2, got {a.tolist()} / {b.tolist()}')
    d = a - b
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    half = stats.t.ppf(1 - (1 - level) / 2, n - 1) * sd / math.sqrt(n)
    return {'deltas': d.tolist(), 'mean': float(mean), 'sd': float(sd), 'level': level,
            'ci': [float(mean - half), float(mean + half)],
            'relative': float(mean / np.mean(b)) if relative else None, 'a_lower': int((d < 0).sum()), 'n': n}


def category(rule, t):
    """The verdict category of one contrast under its original rule; 'withheld' when blocked."""
    if t is None or t.get('status') == 'withheld':
        return 'withheld'
    lo, hi = t['ci']
    mean = t['mean']
    if rule in ('dbg95', 'dir95'):                    # 2N D-BG wording / the direction of a descriptive CI
        return 'a_lower' if hi < 0 else 'a_higher' if lo > 0 else 'no_difference'
    if rule == 'dbn975':                              # 2O D-BN: 97.5% upper < 0 and mean <= -0.005
        return 'candidate' if hi < 0 and mean <= -MDE else 'not_candidate'
    if rule == 'dbv':                                 # 2P D-BV / D-BY labels
        return 'a_lower' if hi < 0 and mean <= -MDE else 'a_higher' if lo > 0 and mean >= MDE else 'no_difference'
    if rule == 'dcb':                                 # 2Q D-CB: the linear readout clearly worse
        return 'worse' if lo > 0 and mean >= MDE else 'not_worse'
    if rule == 'dan':                                 # 2J D-AN primary: the CI excludes 0, negative
        return 'pass' if hi < 0 else 'fail'
    if rule == 'dan20':                               # 2J secondary: primary and >= 20% improvement
        return 'pass' if hi < 0 and -t['relative'] >= .20 else 'fail'
    if rule == 'dar':                                 # 2K D-AR primary
        return 'pass' if hi < 0 else 'fail'
    if rule == 'o73':                                 # 2M O7-3: CI upper < 0 and <= -20%
        return 'pass' if hi < 0 and t['relative'] <= -.20 else 'fail'
    raise ValueError(f'unknown rule {rule}')


def compare(rule, new, old):
    """'same verdict' / 'different verdict' between the rerun and the earlier record (2T)."""
    a, b = category(rule, new), category(rule, old)
    return {'rule': rule, 'rerun': a, 'earlier': b, 'agreement': 'same verdict' if a == b else 'different verdict'}


# ---- the earlier records ---------------------------------------------------------------------------------
LEGACY = {   # git-tracked records of the legacy-neuron experiments, fixed by SHA256 before the 2T runs
    ('2N', 'ETTh1'): ('etthard-20260925-test-ETTh1/ett_test_record.json',
                      'a415ae1778d3aebe77e704646feb768ca8f435ce3935fa01a1cb8a1377e2d8e4'),
    ('2N', 'ETTh2'): ('etthard-20260925-test-ETTh2/ett_test_record.json',
                      '0ea877ff1e69f19d79cf53b4d79573665992b8d157920b45d6bc3718828c5f51'),
    ('2O', 'ETTh1'): ('etthard-revin-20260927-future-ETTh1/ett_future_record.json',
                      '187f376b2227d066ec77dfeea32a403501f4bcf5eb531d01c26955adb09fe236'),
    ('2O', 'ETTh2'): ('etthard-revin-20260927-future-ETTh2/ett_future_record.json',
                      '972f6ee0f03be00056615bd5de4712fb152310f1a7860135aef6a45e15430f8a'),
    ('2P', 'ETTh1'): ('etthard-alpha1-20260927-val-ETTh1/ett_alpha_record.json',
                      '0f53c5c42c917fc60679c36e179fc8714e5cc454704a2b5522f64fe9572ed54c'),
    ('2P', 'ETTh2'): ('etthard-alpha1-20260927-val-ETTh2/ett_alpha_record.json',
                      '13d0896f28548168f4750198d015fd69e9d7bcfdf4381a70e63d77d974f27d27'),
    ('2Q', None):    ('etthard-readout-20260928-val/ett_readout_record.json',
                      '02dc9fdd1f5a1a2d9356f56a492b0a83130ccb15a579609d20dd90aa33fd5db3'),
    ('2J', None):    ('hardsel-014614/hard_selection_record.json',
                      '6b19471ef020b7bde76d85e6e20e91204a575a3d4f3377de1a3b3221a2248de3'),
    ('2K', None):    ('hardctrl-152631/hard_control_record.json',
                      'ddd21119cc26e728c7f21927cf4e3dd5730bc877035c7e208a478ddd72aa7301'),
    ('2L', None):    ('hardsafety-170640/hard_safety_record.json',
                      '7abfa7aae53fbf47ac6b786394bdf20f127c73a180d990d598e5212f3c3daf9d'),
    ('2M', None):    ('hardbench-172518-eval/hard_benchmark_record.json',
                      '91912981ebf0bd8d23c8f44cc820a3d852c5241a86bc2b734876645f83d6aeea'),
}


SPIKEDE_2T = {   # the 2T records (spikeDE neurons, soma, two-Linear head), fixed by SHA256 before the 2U runs
    ('2N', 'ETTh1'): ('etthard-spikede-20261021-2N-test-ETTh1/ett_rerun_record.json',
                      'df773bfafc21a9a962aaf92bec2b5e704f8f04ca6fae34ef2e5e6d3260f2cca8'),
    ('2N', 'ETTh2'): ('etthard-spikede-20261021-2N-test-ETTh2/ett_rerun_record.json',
                      '821062ea061eb7a51916ebf43b68108c739b7b1647d21d1e13543462c8028a98'),
    ('2O', 'ETTh1'): ('etthard-spikede-20261021-2O-future-ETTh1/ett_rerun_record.json',
                      '4c1df3ec84672811fd6e18035723917b111fd72f9ceeaf68519f78f7a71a7446'),
    ('2O', 'ETTh2'): ('etthard-spikede-20261021-2O-future-ETTh2/ett_rerun_record.json',
                      '4833fcef3f6de6ac8423bd84be8225ed8d8366105ba1804d9b7dfd88a3520010'),
    ('2P', None):    ('etthard-spikede-20261021-2P-val-both/ett_rerun_record.json',
                      '793585bab4f864ff936c72abf86723bcfc05243418f9989cf32a1edbb60299bc'),
}


def record_2t(exp, data):
    """(record, name, sha) of the 2T record of an ETT experiment; fatal if it changed."""
    name, digest = SPIKEDE_2T[(exp, None if exp == '2P' else data)]
    path = TASK / 'results' / name
    if sha256(path) != digest:
        raise SystemExit(f'[2T] {name} is not the fixed record (sha256 differs)')
    return json.loads(path.read_text()), name, digest


def values_2t(exp, data):
    """(cond, seed) -> error of the 2T rerun, the withheld cells, and its stored contrasts."""
    record, _, _ = record_2t(exp, data)
    values = {(c['cond'], c['seed']): c['mse'] for c in record['cells']
              if c.get('data') == data and c.get('status') == 'evaluated'}
    withheld = {(b[0], b[1]) for b in record['per_data'][data]['blocked']}
    return values, withheld, record['per_data'][data]['contrasts']


def legacy_record(exp, data=None):
    name, digest = LEGACY[(exp, data)]
    path = TASK / 'results' / name
    if sha256(path) != digest:
        raise SystemExit(f'[legacy] {name} is not the fixed record (sha256 differs)')
    return json.loads(path.read_text()), name, digest


def legacy_values(exp, data=None):
    """(cond, seed) -> error of the earlier record, and the set of withheld cells."""
    record, _, _ = legacy_record(exp, None if exp == '2Q' else data)   # 2Q: one record for both datasets
    values, withheld = {}, set()
    if exp in ('2N', '2O', '2P', '2Q'):
        for r in record['rows']:
            if exp == '2Q' and r['data'] != data:
                continue
            if r.get('status', 'evaluated') != 'evaluated':
                withheld.add((r['cond'], r['seed']))
                continue
            values[(r['cond'], r['seed'])] = r['mse']
            if not r.get('safe', True) or not r.get('repro_ok', True):
                withheld.add((r['cond'], r['seed']))
    elif exp == '2J':
        for r in record['confirm']['rows']:
            values[('q1' if r['label'] == 'q=1' else 'pearson', r['seed'])] = r['recall_mse']
    elif exp in ('2K', '2M'):
        for r in record['rows']:
            values[(r['condition'], r['seed'])] = r['recall_mse']
            if exp == '2M' and not r['safe']:
                withheld.add((r['condition'], r['seed']))
    return values, withheld


def per_seed(values, cond):
    return [values[(cond, s)] for s in SEEDS]


if __name__ == '__main__':
    if len(sys.argv) in (3, 4) and sys.argv[1] == 'freeze-calibration':
        freeze_calibration(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)
