"""Forecasting logs following neorecall_v1 Config/EpochLog conventions.

One isolated date/config/seed+variant directory contains logargs.txt,
log/best_log_0.csv, log/final+result.csv, log/train_0 and log/val_0
TensorBoard events, and model_state/{config.pt,best+model.pt}.
Only measured forecasting metrics are written; no invented energy/ops values.
Full precision metadata remains available in results/<suite>/<run_id>.json.
"""
import csv
from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def run_name(args):
    return f"{args.dataset}_p{args.pred_len}_{args.variant}_code{args.population_code}_seed{args.seed}"


def run_directory(task, args):
    date = datetime.now().strftime('%y%m%d')
    tag = (f'{date}+model+simple_test_model_v1+seq_len+{args.seq_len}'
           f'+pred_len+{args.pred_len}+patch_size+{args.patch_size}'
           f'+K+{args.population}+embed_dim+{args.d_model}+num_heads+{args.heads}'
           f'+depth+{args.depth}+head_dim+{args.head_dim}+lr+{args.lr}')
    isolate = f'seed{args.seed}_{args.variant}_code{args.population_code}'
    return Path(task)/'log'/args.suite/args.dataset/date/tag/isolate


class EpochLog:
    """Same CSV naming, metric prefixes, scalar tags and six-digit CSV format."""

    def __init__(self, save_log_dir, kids=0):
        self.path = Path(save_log_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.kids = kids
        self.csv_path = self.path/f'best_log_{kids}.csv'
        self.train_writer = SummaryWriter(log_dir=str(self.path/f'train_{kids}'))
        self.val_writer = SummaryWriter(log_dir=str(self.path/f'val_{kids}'))
        self.columns = None

    def write(self, epoch, lr, train_result, val_result):
        row = {'epoch': int(epoch)}
        for split, metrics, writer in (('train', train_result, self.train_writer),
                                       ('val', val_result, self.val_writer)):
            for key, value in metrics.items():
                row[f'{split}_{key}'] = f'{value:.6f}'
                writer.add_scalar(f'{split}_{self.kids}/{key}', value, epoch)
        if self.columns is None:
            self.columns = list(row)
        assert self.columns == list(row), 'Metric columns changed during training'
        with self.csv_path.open('a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns, lineterminator='\n')
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        print(f'epoch {epoch} (lr={lr}): train_loss={train_result["loss"]:.6f}, '
              f'val_mse={val_result["mse"]:.6f}\nCurrent log saved to `{self.path}`', flush=True)

    def final(self, result):
        row = {'loss': result['test']['mse'], 'mse': result['test']['mse'],
               'mae': result['test']['mae'], 'parameters': result['parameters'],
               'best_epoch': result['best_epoch'], 'seed': result['config']['seed']}
        with (self.path/'final+result.csv').open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=row, lineterminator='\n')
            writer.writeheader()
            writer.writerow({k: f'{v:.6f}' if isinstance(v, float) else v for k,v in row.items()})
        print(f'Final result saved to `{self.path}`', flush=True)

    def close(self):
        self.train_writer.close()
        self.val_writer.close()


def write_args(run_dir, values):
    # Match Config.print_info's existing logargs.txt location and formatting.
    with (Path(run_dir)/'logargs.txt').open('x') as handle:
        for key, value in values.items():
            handle.write(f'{key:-<30s}{str(value):->70s}\n')
