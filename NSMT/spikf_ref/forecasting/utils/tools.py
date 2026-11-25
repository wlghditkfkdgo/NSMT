import numpy as np
import torch
import matplotlib.pyplot as plt
import os

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
   
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def plot_predicted_time_series(x, pred, true, path,
                               sample_idx=0,
                               channel_idx=0,
                               filename=None):
    """
    x      : 입력 시계열 (Tensor)  [B, seq_len, C] 가정
    pred   : 예측 시계열 (Tensor)  [B, pred_len, C]
    true   : GT 시계열 (Tensor)    [B, pred_len, C]
    args   : Config (save_log_path, pred_len, seq_len 등 포함)
    sample_idx, channel_idx: 몇 번째 샘플 / 채널을 그릴지
    """


    # ---- 텐서 -> numpy ----
    # x: [B, L, C]
    x_np = x[sample_idx].detach().cpu().numpy()        # [L, C] (또는 [L])
    pred_np = pred[sample_idx].detach().cpu().numpy()  # [pred_len, C]
    true_np = true[sample_idx].detach().cpu().numpy()  # [pred_len, C]

    # 채널 선택
    if x_np.ndim == 2:
        hist = x_np[:, channel_idx]          # [L]
    else:
        hist = x_np                          # [L]

    if pred_np.ndim == 2:
        fut_pred = pred_np[:, channel_idx]   # [pred_len]
        fut_true = true_np[:, channel_idx]   # [pred_len]
    else:
        fut_pred = pred_np
        fut_true = true_np

    seq_len = hist.shape[0]
    pred_len = fut_pred.shape[0]

    # ---- 시간축 설정 ----
    t_hist = np.arange(seq_len)
    # 마지막 입력 타임스텝에서부터 예측/GT를 이어서 그림
    t_future = np.arange(seq_len - 1, seq_len - 1 + pred_len)

    # ---- Figure 생성: 가로로 길고 세로로 짧게 ----
    fig, ax = plt.subplots(figsize=(10, 2.5))

    # 입력 시그널
    ax.plot(t_hist, hist, label="Input", linewidth=1.0, color="black")

    # 경계선 (입력/예측 구분)
    ax.axvline(seq_len - 1, linestyle="--", linewidth=0.8, color="gray")

    # GT & 예측
    ax.plot(t_future, fut_true, label="Ground Truth", linewidth=0.8)
    ax.plot(t_future, fut_pred, label="Prediction", linewidth=1.0)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")

    ax.legend(loc="upper left", fontsize=5)
    ax.set_xlim(0, t_future[-1])

    # 여백 최소화 (논문용)
    plt.tight_layout()

    # ---- 저장 ----
    os.makedirs(path, exist_ok=True)
    if filename is None:
        filename = f"predicted_time_series_s{sample_idx}_c{channel_idx}"
    save_path = os.path.join(path, filename)

    plt.savefig(save_path+'.png', dpi=300, bbox_inches="tight")
    plt.savefig(save_path+'.pdf', dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Predicted time series figure saved to `{save_path}`")
    
    
def get_energy_consumption(O_ac, O_mac, E_ac=0.9, E_mac=4.6, unit=None) -> float:
    
    E_ac = E_ac * 1e-12
    E_mac = E_mac * 1e-12
    
    energy = E_ac * O_ac + E_mac * O_mac
    
    if unit is None or unit == 'p':
        return energy / 1e-12
    
    elif unit == 'm' :
        return energy / 1e-3
    
    elif unit == 'u' :
        return energy / 1e-6