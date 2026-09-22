import os
import sys
import time
import json
import glob
import hashlib
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn

from config import Config, parse_arguments, set_random_seed, TASK
from model import LOAD_MODEL
from data_provider.data_factory import data_provider
from utils import EpochLog, EarlyStopping, write_json, sha256, parameter_hash
from test import test, evaluate, selection_diagnostics

SOURCE_FILES = ['config.py', 'model.py', 'ours.py', 'layers.py', 'train.py', 'test.py',
                'utils.py', 'calibrate.py', 'data_provider/data_loader.py',
                'data_provider/data_factory.py', 'data_provider/synthetic.py']


def load_calibration(args):
    """Read back the frozen G11 bound and input_scale that Phase B fixed.

    The bound has to come from calibration and stay put. Recomputing it each step from that
    step's own maximum would make the check vacuous: the state could grow without limit and
    the bound would follow it (audit follow-up 9.3).
    """
    stem = f"{args.dataset}_k{args.n_keys}_r2" if args.task == 'recall' else args.dataset
    pattern = str(TASK / 'results' / 'calibration'
                  / f'{stem}_a{args.alpha}_norm-{args.input_norm}_seed*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1]) as handle:
        payload = json.load(handle)
    picked = payload.get('picked')
    if picked is None:
        return None

    return {'file': os.path.basename(files[-1]), 'input_scale': picked['input_scale'],
            'theta': picked.get('theta'),                     # D-X: 보정에서 정해 그대로 쓴다
            'g11_bound': picked['declared_bound'], 'firing_rate': picked['firing_rate']}


def fit_input_norm(model, data_loader, args):
    """Freeze the per-unit input standardisation on the train split, exactly as calibrate did.

    Without this the buffers stay at mean 0 / std 1 and the model runs at a different
    operating point from the one Phase B fixed: on the recall smoke run the firing rate came
    out at 0.077 instead of the calibrated 0.185. The buffers are registered, so they travel
    with the checkpoint and a reloaded model reproduces the same point.
    """
    from layers import to_patches
    if args.model != 'myModel' or args.input_norm != 'frozen':
        return None
    sample = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= 8:
                break
            sample.append(to_patches(batch[0].float().to(args.device), args.patch_size))
        model.embedding.fit_norm(torch.cat(sample, dim=1))
    print(f"[train] frozen input norm fitted on {sum(t.shape[1] for t in sample)} windows: "
          f"mean in [{model.embedding.norm_mean.min():.3f}, {model.embedding.norm_mean.max():.3f}], "
          f"std in [{model.embedding.norm_std.min():.3f}, {model.embedding.norm_std.max():.3f}]")

    return True


def train_one_epoch(model, data_loader, optimizer, args):
    model.train()
    criterion = nn.MSELoss()
    total = torch.zeros(3, dtype=torch.float64, device=args.device)
    peak, rate = 0., []
    watched = {k: [] for k in ('eta', 'kappa', 'would_cap_rate', 'support_p', 'score_std')}
    selector_grad = {'grad_WQ': [], 'grad_WK': [], 'grad_eta_hat': [], 'singleton_frac': []}
    for i, batch in enumerate(data_loader):
        if args.max_train_batches and i >= args.max_train_batches:
            break
        x, y, truth, _ = batch
        x, y = x.float().to(args.device), y.float().to(args.device)
        truth = truth.to(args.device) if args.mode == 'oracle' else None
        watch = args.model == 'myModel' and i % max(args.g11_every, 1) == 0
        result = model(x, mode=args.mode, truth=truth, return_aux=watch)
        output, aux = result if watch else (result, None)
        loss = criterion(output, y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if watch and args.model == 'myModel':                     # clipping 전 원본 크기를 본다
            sel = model.embedding.neuron.selector
            for name, tensor in (('grad_WQ', sel.query.weight), ('grad_WK', sel.key.weight),
                                 ('grad_eta_hat', sel.eta_hat)):
                selector_grad[name].append(0. if tensor.grad is None
                                           else tensor.grad.norm().item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if watch:
            # G11: 학습 중에도 보정에서 고정한 상한을 본다. 넘으면 즉시 중단한다.
            peak = max(peak, aux['state'].abs().max().item())
            rate.append(aux['spikes'].mean().item())
            live = aux['has_history']
            for key in watched:                                  # eta가 움직이는지 epoch마다 본다
                watched[key].append(aux[key][live].mean().item())
            # sparsemax가 한 칸만 고르면 score gradient가 0이다 (audit 5.2). 비율을 본다.
            selector_grad['singleton_frac'].append((aux['support_size'][live] == 1).double().mean().item())
            if args.g11_bound and peak >= args.g11_bound:
                raise FloatingPointError(
                    f'G11 violated: max|u| = {peak:.3f} reached the bound {args.g11_bound:.3f} '
                    f'frozen at calibration')
        error = (output.detach() - y).double()
        total[0] += error.square().sum()
        total[1] += error.abs().sum()
        total[2] += y.numel()

    result = {'loss': (total[0] / total[2]).item(), 'mse': (total[0] / total[2]).item(),
              'mae': (total[1] / total[2]).item()}
    if rate:                                                  # GRU 등 비스파이킹 모델은 비운다
        result['max_abs_state'] = peak
        result['firing_rate'] = float(np.mean(rate))
        result.update({k: float(np.mean(v)) for k, v in watched.items() if v})
        result.update({k: float(np.mean(v)) for k, v in selector_grad.items() if v})

    return result


def val_one_epoch(model, data_loader, args):
    result = evaluate(model, data_loader, args)[0]

    return {key: result[key] for key in ('loss', 'mse', 'mae')}


def train(args: Config):
    set_random_seed(args.seed)
    started = time.time()
    logger = EpochLog(args.save_log_path, kids=0)
    stopper = EarlyStopping(verbose=True, patience=args.patience)
    train_set, train_loader = data_provider(args, flag='train')
    val_set, val_loader = data_provider(args, flag='val')
    model = LOAD_MODEL[args.model](args, True)
    fit_input_norm(model, train_loader, args)

    active = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"[train] {args.run_id}")
    print(f"[train] parameters: {sum(p.numel() for p in model.parameters())} total, "
          f"{sum(p.numel() for n, p in model.named_parameters() if n in active)} trainable")
    if args.g11_bound:
        print(f"[train] G11 frozen bound {args.g11_bound:.1f} from {args.calibration_file}")
    else:
        print("[train] G11 bound NOT set: no calibration artifact matched. Run calibrate.py first.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                           patience=args.scheduler_patience)
    args.save_arg()
    args.print_info()

    for epoch in range(args.epoch):
        train_result = train_one_epoch(model, train_loader, optimizer, args)
        val_result = val_one_epoch(model, val_loader, args)
        logger.logging(epoch=epoch, train_result=train_result, val_result=val_result)
        logger.verbose(epoch=epoch, lr=optimizer.param_groups[0]['lr'],
                       train_result=train_result, val_result=val_result)
        scheduler.step(val_result['loss'])
        stopper(val_result['loss'], model, args.save_model_state_path)
        if stopper.early_stop:
            print(f"[train] early stop at epoch {epoch}")
            break
    logger.close()

    payload = test(args)
    payload['train'] = {'epochs_run': epoch + 1, 'seconds': time.time() - started,
                        'best_val_loss': float(stopper.val_loss_min),
                        **{k: train_result[k] for k in
                           ('firing_rate', 'max_abs_state', 'eta', 'kappa', 'would_cap_rate',
                            'support_p', 'score_std', 'grad_WQ', 'grad_WK', 'grad_eta_hat',
                            'singleton_frac') if k in train_result}}
    payload['provenance'] = {
        'run_uuid': args.run_uuid, 'config_hash': args.config_hash,
        'calibration': {'file': args.calibration_file, 'input_scale': args.input_scale,
                        'g11_bound': args.g11_bound},
        'source_sha256': {name: sha256(str(TASK / name)) for name in SOURCE_FILES},
        'parameter_hash': parameter_hash(model), 'trainable_parameters': active,
        'data': {'train': len(train_set), 'val': len(val_set),
                 'revision': 'r2' if args.task == 'recall' else None,
                 'data_seed': args.data_seed},
        'python': sys.version.split()[0], 'torch': torch.__version__,
        'device': str(args.device), 'finished_utc': datetime.now(timezone.utc).isoformat()}
    write_json(args.result_path, payload)

    # fresh reload: 저장한 것이 그대로 돌아오는지 확인한다. frozen 통계 포함.
    reloaded = LOAD_MODEL[args.model](args, train=False)
    _, test_loader = data_provider(args, flag="test")
    check = evaluate(reloaded, test_loader, args)[0]
    same = abs(check['mse'] - payload['test']['mse']) < 1e-9
    print(f"[train] fresh reload mse {check['mse']:.9f} vs saved {payload['test']['mse']:.9f} "
          f"-> {'identical' if same else 'MISMATCH'}")
    payload['provenance']['fresh_reload_matches'] = bool(same)
    write_json(args.result_path, payload)

    return payload


def main():
    args = parse_arguments()
    config = Config()
    config.set_args(args)
    calibration = load_calibration(config)
    config.calibration_file = calibration['file'] if calibration else None
    config.g11_bound = calibration['g11_bound'] if calibration else None
    # 보정이 정한 값은 CLI 기본값을 덮어쓴다. 하나라도 빠지면 이름만 보정된 실험이 된다.
    for field in ('input_scale', 'theta'):
        value = calibration.get(field) if calibration else None
        if value is not None and getattr(config, field) != value:
            print(f"[train] {field} {getattr(config, field)} overridden by calibration {value}")
            setattr(config, field, value)
    config.calibrated_fields = [f for f in ('input_scale', 'theta')
                                if calibration and calibration.get(f) is not None]
    blob = json.dumps({k: str(v) for k, v in sorted(vars(config).items())
                       if k not in ('device',)}, sort_keys=True)
    config.config_hash = hashlib.sha256(blob.encode()).hexdigest()[:16]
    config.run_uuid = hashlib.sha256((blob + str(time.time_ns())).encode()).hexdigest()[:16]

    return train(config)


if __name__ == '__main__':
    main()
