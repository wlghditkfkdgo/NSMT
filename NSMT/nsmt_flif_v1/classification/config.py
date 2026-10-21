import os

# os.environ['CUBLAS_WORKSPACE_CONFIG'] =':4096:8'

import torch
import pandas as pd
from scipy.io import arff

from matplotlib import pyplot as plt
import numpy as np

import time
import argparse
import random

import torch.backends


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.random.initial_seed()  
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-gpu
    torch.backends.cudnn.deterministic = True # reduce operation speed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # torch.cuda.set_rng_state(seed)
    # torch.set_rng_state(seed)
    torch.use_deterministic_algorithms(True)
    
def set_seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# def parse_arguments(args):
def parse_arguments():
    
    parser = argparse.ArgumentParser(description='the hyperparameters for training')
    
    parser.add_argument('--model', default='myModel')
    parser.add_argument('--seed', dest='seed', type=int, default=42)
    parser.add_argument('--config', dest='config', default=None, help='If you have config file, input the file path')
    parser.add_argument('--saved_epoch', dest='saved_epoch', nargs='+', type=int, default=[1, ], help='The epoch number saved as check point')
    
    save_arg = parser.add_argument_group("save information")
    save_arg.add_argument('--tag', dest='tag', nargs='+')
    save_arg.add_argument('-s', '--save', dest='best_save', action='store_true', help='Save best model state')
    save_arg.add_argument('--print_epoch', nargs='?', type=int, default=5, help='Epoch print infomation period')
    save_arg.add_argument('--log_dir', nargs='?', default='../results/ECG/result', help='The directory name to save log file \n\t default: %(default)s')
    save_arg.add_argument('--save_patience', dest='save_log_patience', nargs='?', type=int, default=2)
    
    save_arg.add_argument('--test', dest='test', action='store_true', help='If true, model is tested after training.')
    save_arg.add_argument('--tsne', dest='tsne', action='store_true', help='If true, the emb of model is visualized.')
    save_arg.add_argument('--only_path_test', dest='only_path_test', action='store_true', help='If true, the time path of model is tested.')
    
    train_arg = parser.add_argument_group("training parameters")
    train_arg.add_argument('-nd', '--num_device', dest='num_device', nargs='?', type=int, choices=[0, 1, 2, 3, 4, 5], default=0, help='Device number : [0, 1, 2, 3] (default: %(default)s)')
    train_arg.add_argument('-e', '--epoch', dest='epoch', nargs='?', type=int, default=50, help='# of total epoch')
    train_arg.add_argument('--warm_up_epoch', dest='warm_up_epoch', nargs='?', type=int, default=100, help='# of total epoch for pre-training')
    train_arg.add_argument('-bs', '--batch_size', dest='batch_size', nargs='?', type=int, default=128, help='Batch size')
    train_arg.add_argument('-lr', '--learning_rate', dest='lr', nargs='?', type=float, default=1e-4, help='Learning rate')
    train_arg.add_argument('--max_lr', dest='max_lr', nargs='?', type=float, default=0.001, help='maximum learning rate in cosine lr scheduler')
    train_arg.add_argument('--scheduler', dest='scheduler', nargs='?', type=str, choices=['step', 'lambda', 'exponential', 'cosine', 'reduce'], default='reduce', help='scheduler (default: %(default)s)')
    train_arg.add_argument('--weight_decay', dest='weight_decay', nargs='?', type=float, default=1e-4)   # stage 9: was type=int (a CLI value could not be parsed); default unchanged
    train_arg.add_argument('--num_workers', dest='num_workers', nargs='?', type=int, default=8)

    model_arg = parser.add_argument_group("Transformer parameters")
    model_arg.add_argument('--alpha', dest='alpha', nargs='?', type=float, default=0.1, help='hyperparameter for gating')
    model_arg.add_argument('-emb', '--embedding_dim', dest='embed_dim', nargs='?', type=int, default=256, help='d_model in transformer')
    model_arg.add_argument('-nh', '--num_heads', dest='num_heads', type=int, default=8, help='num of heads in self-attention layer')
    model_arg.add_argument('--layers', dest='num_layers', nargs='?', type=int, default=1, help='number of encoder layers')
    model_arg.add_argument('--time_layers', dest='time_num_layers', nargs='?', type=int, default=2, help='number of layers in neocortex')
    # model_arg.add_argument('--num_patches', dest='num_patches', type=int, default=64, help='number of patches in Data Stream (default: %(default)s)')
    model_arg.add_argument('--patch_size', dest='patch_size', type=int, default=16, help='patch size in data stream')
    model_arg.add_argument('--mlp_ratios', dest='mlp_ratios', type=float, default=4)
    model_arg.add_argument('--patch_stride', dest='stride', type=int, default=9)
    model_arg.add_argument('--connect_f', dest='connect_f', nargs='?', choices=['ADD', 'AND', 'IAND'], default='IAND', help='the types of residual connection in SNN')
    model_arg.add_argument('--gating', dest='gating', default='original', help='gating type')
    model_arg.add_argument('--attn', dest='attn', nargs='?', default='MSSA', choices=['SSA'],)
    model_arg.add_argument('--keep_ratio', nargs='?', type=float, default=0.15)
    
    snn_arg = parser.add_argument_group("snn parameters")
    snn_arg.add_argument('-T', '--time_steps', dest='time_steps', nargs='?', type=int, default=4, help='Total time steps for SNN simulation (default: %(default)s)')
    snn_arg.add_argument('-b', '--bias', dest='bias', action=argparse.BooleanOptionalAction, help='spiking neuron bias (option: `--no-bias`)')
    snn_arg.add_argument('-spk', '--spk_encoding', dest='spk_encoding', action='store_true', help='spike encoding')
    snn_arg.add_argument('--roll_mode', dest='roll_mode', choices=['none','sine','rand'], default='none', help='Hippocampus spike encoding: none(repeat) | sine(raised-cosine shift, APE-like) | rand(random past-token selection, time order preserved)')
    snn_arg.add_argument('--neo_recurrent', dest='neo_recurrent', action='store_true', default=False, help='P1: re-entrant recurrent Neocortex (Zipser attractor)')
    snn_arg.add_argument('--neo_rec_gate', dest='neo_rec_gate', action='store_true', default=False, help='learnable recurrence gate g (init 0): stable feedforward superset, fixes classification instability')
    snn_arg.add_argument('--neo_legacy', dest='neo_legacy', action='store_true', default=False, help='review 8.1: use OLD separate passive/active classes instead of exact-passive-superset NeoLayer (default: exact)')
    snn_arg.add_argument('--no_neo', dest='no_neo', action='store_true', default=False, help='ablation: Hippocampus only (no Neocortex / Memory Replay / low-pass aux)')
    snn_arg.add_argument('--grad_clip', dest='grad_clip', type=float, default=0.0, help='gradient clipping max_norm (0=off) for recurrent BPTT stability')
    snn_arg.add_argument('--neo_tau', dest='neo_tau', type=float, default=2.0, help='Neocortex memory time-constant (recurrent attractor leak)')
    # nsmt_flif_v1: neocortex-path neuron replacement (defaults = the ECG model)
    snn_arg.add_argument('--neo_neuron', choices=['sj_lif', 'lif', 'flif', 'pop_flif'], default='sj_lif',
                         help='neocortex neurons: sj_lif = spikingjelly LIF (ECG model); lif = spikeDE LIF (control); flif = spikeDE f-LIF')
    snn_arg.add_argument('--neo_where', choices=['both', 'mlp'], default='both',
                         help='both = encoding_neo LIF + time_block LIFs; mlp = time_block only (all on the token axis)')
    snn_arg.add_argument('--neo_alpha', type=float, default=0.5, help='f-LIF fractional order (spikeDE default)')
    snn_arg.add_argument('--fde_method', choices=['pred', 'gl'], default='pred')
    snn_arg.add_argument('--fde_step', type=float, default=1.0)
    snn_arg.add_argument('--fde_memory', type=int, default=0, help='0 = full history')
    # stage 6 (MODI 2026-10-19): population f-LIF branches and the bidirectional neocortex
    snn_arg.add_argument('--pop_taus', type=str, default='4,8,16,32', help='pop_flif: branch time constants (soma uses --neo_tau)')
    snn_arg.add_argument('--neo_bidir', action='store_true', default=False,
                         help='stage 6: second neocortex over the reversed token axis, fused with the forward one by OR')
    # stage 7: --model neoMixer (attention replaced by the neocortex network; --layers = blocks, --time_layers = mixer layers)
    snn_arg.add_argument('--mixer', choices=['none', 'lif', 'flif', 'pop_flif', 'lif_rec', 'sj_lif'], default='pop_flif')
    snn_arg.add_argument('--mixer_bidir', action='store_true', default=False)
    # stage 9 (MODI 2026-10-19): CinC 2017 / SHD, Spikformer-style Hippo-only, population calibration (see train_rhythm.py)
    snn_arg.add_argument('--hippo_ssa', action='store_true', default=False, help='Hippo-only with Spikformer-style token self-attention')
    snn_arg.add_argument('--hippo_T', type=int, default=0, help='hippocampus repetitions (0 = number of tokens; only hippo_ssa / hippo_xattn may differ)')
    # MODI 2026-10-21 (stage 9 cause-separation): 1-B position, 1-C hippocampus T_h with token cross-attention over neocortex tokens
    snn_arg.add_argument('--ssa_pe', type=int, choices=[0, 1], default=1, help='relative position bias in SSA / hippo_xattn (0 = none)')
    snn_arg.add_argument('--pos_emb', action='store_true', default=False, help='learned absolute position in the hippocampus / mixer embedding')
    snn_arg.add_argument('--hippo_xattn', action='store_true', default=False, help='hippocampus T_h repeats with token cross-attention (q = hippocampus, k/v = neocortex tokens)')
    snn_arg.add_argument('--pop_soma_init', type=float, default=None, help='pop_flif soma mixing weight init (None = 1/K)')
    snn_arg.add_argument('--fold', type=int, default=0, help='CinC 2017: test fold 0..4 of the record-wise stratified 5-fold split')
    snn_arg.add_argument('--val_frac', type=float, default=0.1, help='share of the training part held out for validation')
    snn_arg.add_argument('--win_s', type=float, default=30.0, help='CinC 2017 window length (s)')
    snn_arg.add_argument('--hop_s', type=float, default=15.0, help='CinC 2017 evaluation window hop (s)')
    snn_arg.add_argument('--fs_target', type=float, default=125.0, help='CinC 2017 resampling rate (Hz)')
    snn_arg.add_argument('--loss_kind', choices=['cal', 'ce'], default='cal', help='CinC 2017: CAL class-weighted CE or plain CE')
    snn_arg.add_argument('--crop_tokens', type=int, default=0, help='memory pre-check: input limited to k tokens (0 = full)')
    snn_arg.add_argument('--eval_bs', type=int, default=32)
    snn_arg.add_argument('--hippo_only', action='store_true', default=False,
                         help='MODI decision 7: no neocortex; the cross-attention key/value is the hippocampus itself')
    snn_arg.add_argument('--tau', dest='tau', type=float, default=2.0)
    
    data_arg = parser.add_argument_group("data arguments")
    # data_arg.add_argument('--train_data_root', nargs='?', type=str, default='../data/5class_mitbih_train_7_187.csv', help='The root of train data root saved \n\t default: %(default)s')
    # # data_arg.add_argument('--train_data_root', nargs='?', type=str, default='./data/aami_4cls_mitbih_train_denoise_180.csv', help='The root of train data root saved \n\t default: %(default)s')
    # data_arg.add_argument('--test_data_root', nargs='?', type=str, default='../data/5class_mitbih_test_3_187.csv', help='The root of test data root saved \n\t default: %(default)s')
    # data_arg.add_argument('--sampling', dest='sampling', choices=['avg', 'min', 'cut', 'smote', 'None'], default='None')
    # data_arg.add_argument('--ratio', dest='train_val_ratio', nargs='+', type=float, default=None, help='ratio for splitting training and validation(test) dataset (default:%(default)s)')
    # data_arg.add_argument('--n_folds', dest='n_folds', type=int, default=2)
    # data_arg.add_argument('--best_folds', dest='best_folds', type=int, default=0)
    data_arg.add_argument('--num_classes', dest='num_classes', type=int, default=5)
    data_arg.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    data_arg.add_argument('--num_channels', dest='num_channels', type=int, default=1)
    data_arg.add_argument('--segment_len', dest='segment_len', type=int, default=0)
    data_arg.add_argument('--data', help='The name of dataset')
    data_arg.add_argument('--task_name', default='classification')
    data_arg.add_argument('--root_path', type=str, default='dataset',
                        help='root path of the data file')
    
    data_arg.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    data_arg.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    data_arg.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    data_arg.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    data_arg.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    data_arg.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    data_arg.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    data_arg.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    data_arg.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    data_arg.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    data_arg.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    data_arg.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    data_arg.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    data_arg.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    data_arg.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    data_arg.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    data_arg.add_argument('--extra_tag', type=str, default="", help="Anything extra")
    
    data_arg.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    config = parser.parse_args()
    
    return config


class Config():
    def __init__(self):
        self.date = time.strftime("%y%m%d", time.localtime(time.time()))
    
    def set_args(self, args:argparse.ArgumentParser):
        
        tag = ""
        if args.tag is not None:
            for t in args.tag:
                tag = tag + f"+{t}"
                
        tag_args = ["model", "keep_ratio", "patch_size", "time_num_layers", "embed_dim", "num_heads", "alpha", "mlp_ratios", "epoch", "lr", "gating"]

        for ktag in tag_args:
            vtag = getattr(args, ktag, 'na')
            tag = tag + f"+{ktag}+{vtag}"

        # review 6.1: isolate seed + variant in a SEPARATE short path level (avoids 255-byte
        # per-component filename limit) so parallel multi-seed / multi-variant runs never share
        # one best+model.pt.
        _iso = "seed{}_rec{}_tau{}_wm{}".format(
            getattr(args, 'seed', 'na'), int(bool(getattr(args, 'neo_recurrent', False))),
            getattr(args, 'neo_tau', 'na'), getattr(args, 'wrec_mode', 'full'))
        # nsmt_flif_v1: non-default conditions get their own seed+variant directory; the default name is unchanged
        if getattr(args, 'neo_neuron', 'sj_lif') != 'sj_lif':
            _iso += f"_neo{args.neo_neuron}-{args.neo_where}-a{args.neo_alpha}"
            if args.fde_step != 1.0 or args.fde_method != 'pred' or args.fde_memory != 0:
                _iso += f"-h{args.fde_step}-{args.fde_method}-m{args.fde_memory}"
        if getattr(args, 'no_neo', False):
            _iso += "_noneo"
        if getattr(args, 'hippo_only', False):
            _iso += "_hippoonly"
        if getattr(args, 'neo_bidir', False):
            _iso += "_bidir"
        if getattr(args, 'model', 'myModel') == 'neoMixer':
            _iso += f"_mix-{args.mixer}" + ("-bidir" if getattr(args, 'mixer_bidir', False) else "") + f"-L{args.num_layers}-a{args.neo_alpha}"
        if (getattr(args, 'neo_neuron', 'sj_lif') == 'pop_flif' or getattr(args, 'mixer', '') == 'pop_flif') and getattr(args, 'pop_taus', '4,8,16,32') != '4,8,16,32':
            _iso += "_pt" + args.pop_taus.replace(',', '-')
        # stage 9 suffixes (absent for every earlier configuration)
        if getattr(args, 'hippo_ssa', False):
            _iso += f"_ssaT{args.hippo_T}"
        if getattr(args, 'hippo_xattn', False):
            _iso += f"_xaT{args.hippo_T}"
        if getattr(args, 'pos_emb', False):
            _iso += "_pe"
        if int(getattr(args, 'ssa_pe', 1)) == 0:
            _iso += "_nope"
        if getattr(args, 'pop_soma_init', None) is not None:
            _iso += f"_psi{args.pop_soma_init:g}"
        if str(getattr(args, 'data', '')).upper() in ('CINC2017', 'SHD'):
            _iso += f"_f{args.fold}" + ("_ce" if getattr(args, 'loss_kind', 'cal') == 'ce' and str(args.data).upper() == 'CINC2017' else '')
            if getattr(args, 'crop_tokens', 0):
                _iso += f"_crop{args.crop_tokens}"
        self.save_result_path = os.path.join(os.getcwd(), args.log_dir, self.date, self.date + tag, _iso)
        
        self.save_log_path = os.path.join(self.save_result_path, 'log')
        os.makedirs(self.save_log_path, exist_ok=True)
        
        self.device = torch.device(f'cuda:{args.num_device}' if torch.cuda.is_available() else 'cpu')
        
        for karg, varg in args._get_kwargs():
            setattr(self, f"{karg}", varg)

        if self.best_save: 
            self.save_model_state_path = os.path.join(self.save_result_path, "model_state")
            os.makedirs(self.save_model_state_path, exist_ok=True) 
            
        if self.config : delattr(self, "config")
        delattr(self, "num_device")
        delattr(self, "log_dir")
        delattr(self, "tag")
        
    def load_args(self, config_path:str, config:argparse.ArgumentParser):
        
        args_info_path = os.path.join(config_path, "model_state", "config.pt")
        args = torch.load(args_info_path, map_location='cpu')
        
        for kargs, vargs in args.items():
            if hasattr(config, kargs) and kargs in ['test', 'only_path_test']:
                setattr(self, f"{kargs}", getattr(config, kargs))
            else:
                setattr(self, f"{kargs}", vargs)
                
        self.saved_epoch = config.saved_epoch if self.saved_epoch.__len__() < 3 else self.saved_epoch
            
        self.save_result_path = config_path
        self.save_log_path = os.path.join(self.save_result_path, 'log')
        self.device = torch.device(f'cuda:{config.num_device}' if torch.cuda.is_available() else 'cpu')
        self.save_model_state_path = os.path.join(self.save_result_path, "model_state")
    
    def save_arg(self):
        
        """call only in training phase
        """
        
        args = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
        
        torch.save(args, self.save_model_state_path + "/config.pt")

    def print_info(self):

        args = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
        
        print(f"{' PARAMETERS INFO ':=^100s}")

        with open(self.save_log_path + 'args.txt', 'w', encoding='utf-8') as args_txt:
            for k, v in args.items():
                arg = f"{k:-<30s}{str(v):->70s}"
                print(arg)
                args_txt.write(str(arg) + '\n')
                
            print('\n')
    