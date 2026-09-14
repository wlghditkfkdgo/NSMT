"""EpochLog/EarlyStopping adapted directly from neorecall_v1/forecasting/utils.py."""
import os
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class EpochLog:
    def __init__(self, save_log_dir, kids=0, filename=None):

        self.kids = kids
        os.makedirs(save_log_dir, exist_ok=True)
        if filename is None:
            # 공백/특수문자 최소화한 파일명 권장
            filename = f"best_log_{kids}.csv"
        self.save_log_path = save_log_dir +'/'+ filename
        self._header_written = os.path.exists(self.save_log_path)

        self._columns = None
        self.train_writer = SummaryWriter(log_dir=save_log_dir + f'/train_{kids}')
        self.val_writer = SummaryWriter(log_dir=save_log_dir + f'/val_{kids}')

    def _get_args(self, **kwargs):

        epoch = kwargs.get("epoch")
        train_result = kwargs.get("train_result")
        val_result = kwargs.get("val_result")
        lr = kwargs.get("lr")

        return epoch, train_result, val_result, lr

    def write(self, **kwargs):

        epoch, train_result, val_result, lr = self._get_args(**kwargs)

        self._logging(epoch, train_result=train_result, val_result=val_result)
        self._verbose(epoch, lr, train_result=train_result, val_result=val_result)

        train_dict = {f"train_{k}": v for k, v in train_result.items()}
        val_dict   = {f"val_{k}": v for k, v in val_result.items()}

        row = {"epoch": int(epoch), **train_dict, **val_dict}
        df = pd.DataFrame([row])

        if self._columns is None:
            self._columns = list(df.columns)
        else:
            df = df.reindex(columns=self._columns, fill_value=pd.NA)

        df.to_csv(
            self.save_log_path,
            mode="a",
            header=(not self._header_written),
            index=False,
            float_format="%.6f"
        )
        self._header_written = True

    def _logging(self, epoch, train_result, val_result):

        for k, v in train_result.items():
            self.train_writer.add_scalar(f'train_{self.kids}/{k}', v, epoch)

        for k, v in val_result.items():
            self.val_writer.add_scalar(f'val_{self.kids}/{k}', v, epoch)

    def logging(self, **kwargs):

        epoch, train_result, val_result, _ = self._get_args(**kwargs)
        self._logging(epoch, train_result, val_result)

    def _verbose(self, epoch, lr, train_result, val_result):

        bar = "-" * 30

        print(bar)
        print(f"{'epoch':15s}{epoch:>15d} (lr={lr})")
        for k, v in train_result.items():
            value = v if isinstance(v, float) else v.mean()
            print(f"{'train_' + k:15s}{value:>15.5f}")
        if val_result is not None:
            for k, v in val_result.items():
                value = v if isinstance(v, float) else v.mean()
                print(f"{'val_' + k:15s}{value:>15.5f}")
            print(bar)
        # if num_classes > 0:
        #     for i in range(num_classes): print(f"{'val_' + 'acc' + '_' + str(i):15s}{val_result['acc'][i]:>15.5f}")
        # print(bar)
    def verbose(self, **kwargs):

        epoch, train_result, val_result, lr = self._get_args(**kwargs)
        self._verbose(epoch, lr, train_result, val_result)

    def close(self):

        try :
            self.train_writer.flush()
            self.val_writer.flush()
        except Exception:
            pass

        self.train_writer.close()
        self.val_writer.close()


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if path is not None: self.save_checkpoint(val_loss, model, path)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if path is not None: self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, 'best+model.pt'))
        self.val_loss_min = val_loss
