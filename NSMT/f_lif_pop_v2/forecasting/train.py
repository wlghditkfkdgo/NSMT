import os
import sys
import time
import shlex
import subprocess
import importlib.metadata
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn

from config import Config, parse_arguments, set_random_seed, TASK, NSMT
from model import LOAD_MODEL
from data_provider.data_factory import data_provider
from utils import EpochLog, EarlyStopping, write_json, sha256, parameter_hash
from test import test, evaluate


def train_one_epoch(model, data_loader, optimizer, args):
    model.train()
    criterion = nn.MSELoss()
    total_mse, total_mae, elements = 0., 0., 0
    for i, batch in enumerate(data_loader):
        if args.max_train_batches and i >= args.max_train_batches:
            break
        x, y, _, _ = batch
        x, y = x.float().to(args.device), y.float().to(args.device)
        output = model(x)
        loss = criterion(output, y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        error = (output.detach() - y).double()
        total_mse += error.square().sum().item()
        total_mae += error.abs().sum().item()
        elements += y.numel()
    return {'loss': total_mse / elements, 'mse': total_mse / elements, 'mae': total_mae / elements}


def val_one_epoch(model, data_loader, args):
    result = evaluate(model, data_loader, args)[0]
    return {k: result[k] for k in ['loss', 'mse', 'mae']}


def train(args: Config):
    set_random_seed(args.seed)
    started = time.time()
    logger = EpochLog(args.save_log_path, kids=0)
    early_stopping = EarlyStopping(verbose=True, patience=args.patience)
    train_set, train_loader = data_provider(args, flag='train')
    val_set, val_loader = data_provider(args, flag='val')
    model = LOAD_MODEL[args.model](args, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=args.scheduler_patience)
    source_files = ['config.py', 'model.py', 'ours.py', 'layers.py', 'backbones.py', 'train.py', 'test.py', 'utils.py',
                    'data_provider/data_loader.py', 'data_provider/data_factory.py']
    result = {'run_id': args.run_id, 'config': {k: str(v) if isinstance(v, torch.device) else v for k, v in vars(args).items()},
              'command': shlex.join([sys.executable, *sys.argv]), 'cwd': os.getcwd(),
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=NSMT, text=True).strip(),
              'source_sha256': {p: sha256(TASK / p) for p in source_files},
              'initial_parameter_sha256': parameter_hash(model),
              'parameters': sum(p.numel() for p in model.parameters()),
              'beta': model.embedding.lif.beta.cpu().tolist(),
              'architecture': args.architecture, 'neuron_version': 'population_selective_v2',
              'data': {'path': os.path.realpath(train_set.path), 'sha256': sha256(train_set.path),
                       'columns': train_set.columns, 'scaler_mean': train_set.scaler.mean_.tolist(),
                       'scaler_scale': train_set.scaler.scale_.tolist(), 'scale_fit_rows': [0, 8640],
                       'splits': {'train': [0, 8640], 'val': [8640, 11520], 'test': [11520, 14400]},
                       'context': 'validation/test include preceding seq_len observations',
                       'windows': {'train': len(train_set), 'val': len(val_set), 'test': 2880 - args.pred_len + 1}},
              'environment': {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                              'gpu': torch.cuda.get_device_name(args.device) if args.device.type == 'cuda' else None,
                              'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'spikingjelly', 'tensorboard']}},
              'protocol': {'loss': 'MSE', 'scale': 'train-standardized, all horizon/channel elements',
                           'state_reset': 'each independent input window', 'drop_last': False,
                           'selection': 'strict minimum validation MSE', 'gradient_clip': 1.,
                           'max_epochs': args.epoch, 'early_stopping_patience': args.patience,
                           'scheduler': {'name': 'ReduceLROnPlateau', 'factor': .5, 'patience': args.scheduler_patience},
                           'read_mode': args.read_mode if args.retrieval else 'off',
                           'score': 'negative mean squared learned Q/K distance / temperature',
                           'gate': 'real probability mass * sigmoid([charge,memory,best-score-minus-null,real-mass])',
                           'memory_gradient': 'full BPTT; detached subtractive reset',
                           'smoke_only': bool(args.max_train_batches or args.max_eval_batches)},
              'history': []}
    write_json(os.path.join(args.save_log_path, 'provenance.json'), result)
    best = float('inf')
    args.best_epoch = -1
    try:
        for epoch in range(args.epoch):
            tick = time.time()
            lr = optimizer.param_groups[0]['lr']
            train_result = train_one_epoch(model, train_loader, optimizer, args)
            val_result = val_one_epoch(model, val_loader, args)
            scheduler.step(val_result['mse'])
            logger.write(epoch=epoch, lr=lr, train_result=train_result, val_result=val_result)
            if val_result['mse'] < best:
                best, args.best_epoch = val_result['mse'], epoch
            early_stopping(val_result['mse'], model, args.save_model_state_path)
            result['history'].append({'epoch': epoch, 'lr': lr, 'train': train_result, 'val': val_result,
                                      'seconds': time.time() - tick})
            write_json(os.path.join(args.save_log_path, 'history.json'), result['history'])
            if early_stopping.early_stop:
                break
    finally:
        logger.close()
    model.load_state_dict(torch.load(os.path.join(args.save_model_state_path, 'best+model.pt'), map_location=args.device))
    restored_val = val_one_epoch(model, val_loader, args)
    if not np.isclose(restored_val['mse'], best, rtol=1e-7, atol=1e-9):
        raise AssertionError('Restored checkpoint does not reproduce selected validation MSE')
    args.save_arg()
    result['best_epoch'], result['best_val_mse'] = args.best_epoch, best
    result['restored_val'] = restored_val
    if args.test:
        result['test'] = test(args, model=model)
    result['wall_seconds'] = time.time() - started
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    result['max_cuda_memory_bytes'] = torch.cuda.max_memory_allocated(args.device) if args.device.type == 'cuda' else 0
    result['checkpoint_sha256'] = sha256(os.path.join(args.save_model_state_path, 'best+model.pt'))
    write_json(args.result_path, result)
    print(f'Completed {args.run_id}: {result["wall_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    config = parse_arguments()
    args = Config()
    args.set_args(config)
    args.print_info()
    train(args)
