from exp.exp_ETT import Exp_ETT
import argparse
import torch

# Argument parser setup
parser = argparse.ArgumentParser()

# Model settings
parser.add_argument('--model', type=str, default='SpikF', help='model of the experiment')
parser.add_argument('-s', '--save', dest='save', action='store_true', help='Save best model state')
# Dataset settings
parser.add_argument('--data', type=str, required=False, default='ETTh1', 
                    choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ECL', 'traffic', 'weather', 
                             'electricity', 'metr-la', 'pems-bay', 'solar-energy', 'exchange'], 
                    help='name of dataset')
parser.add_argument('--root_path', type=str, default='./datasets/long/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='location of the data file')
parser.add_argument('--features', type=str, default='M', choices=['S', 'M'], 
                    help='features S is univariate, M is multivariate')
parser.add_argument('--target', type=str, default='OT', help='target feature')
parser.add_argument('--checkpoints', type=str, default='exp/run_ETT/', help='location of model checkpoints')

# Model settings
parser.add_argument('--seq_len', type=int, default=96, help='look back window')
parser.add_argument('--pred_len', type=int, default=336, help='prediction sequence length, horizon')
parser.add_argument('--levels', type=int, default=2)
parser.add_argument('--T', type=int, default=16)
parser.add_argument('--patch_num', type=int, default=48)
parser.add_argument('--patch_dim', type=int, default=32)
parser.add_argument('--alpha', type=float, default=2.0)
parser.add_argument('--hidden_dim', type=int, default=720)
parser.add_argument('--random_seed', type=int, default=0)

# Training settings
parser.add_argument('--cols', type=str, nargs='+', help='file list')
parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
parser.add_argument('--itr', type=int, default=0, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=1, help='early stopping patience')
parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mae', help='loss function')
parser.add_argument('--model_name', type=str, default='SpikF')
parser.add_argument('--evaluate', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--tau', type=float, default=2.0)
parser.add_argument('--label_len', type=int, default=48)
# 2026-11-25 (exp/f-lif-pop-v3, PROJECT_LOG SpikF-LIF pre-registration): --activ gelu | lif (head activation, models/SpikF.py);
# --out <run dir>: checkpoints (<out>/checkpoints/<setting>/checkpoint.pth), test outputs (<out>/test_results/<setting>/) and
# <out>/result.json; without --out the paths are the released ones. The released scripts never pass --random_seed (seed 0).
parser.add_argument('--activ', type=str, default='gelu', choices=['gelu', 'lif'])
parser.add_argument('--out', type=str, default='')
# 2026-11-26: --fast_eval (speed only, PROJECT_LOG SpikF-LIF pre-registration, check-spikf-act-20261125/check_fast): validation and the
# final test (and the in-loop test pass) in batches of --eval_bs instead of 1 (losses weighted by batch size so they stay means over
# windows), autograd anomaly mode off, neurons reset before the syops count. Training steps, model selection and the metrics are
# unchanged up to floating-point rounding (check-spikf-act-20261125/check_fast).
parser.add_argument('--fast_eval', action='store_true')
parser.add_argument('--eval_bs', type=int, default=256)
parser.add_argument('--syops_bs', type=int, default=1)

args = parser.parse_args()
if args.fast_eval:
    torch.autograd.set_detect_anomaly(False)
if args.out:
    import os
    args.checkpoints = os.path.join(args.out, 'checkpoints')

def main(rank):
    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    # Initialize experiment
    Exp = Exp_ETT

    setting = '{}_{}_ft{}_sl{}_pl{}_lr{}_bs{}'.format(
        args.model, args.data, args.features, args.seq_len, args.pred_len, 
        args.lr, args.batch_size)
    
    args.rank = rank
    exp = Exp(args)
    torch.cuda.set_device(args.rank)                                    # 2026-11-25: was set_device(1); one visible GPU, rank 0

    # Start training
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    # Start testing
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mse, mae = exp.test(setting)
    print('Final mean normed mse:{:.4f}, mae:{:.4f}'.format(mse, mae))

if __name__ == '__main__':
    main(args.rank)