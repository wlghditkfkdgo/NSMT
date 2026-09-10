import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import torch
import pandas as pd
from scipy.io import arff

from matplotlib import pyplot as plt
import numpy as np

import time
import argparse
import random

import torch.backends
from timm.utils import random_seed


def set_random_seed(seed):
    random_seed(seed, 0)
    torch.manual_seed(seed)
    # torch.random.initial_seed()  
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-gpu
    torch.backends.cudnn.deterministic = True # reduce operation speed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
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
    
    parser.add_argument('--model', dest='model', default='myModel')
    
    parser.add_argument('--seed', dest='seed', type=int, default=42)
    parser.add_argument('--config', dest='config', default=None, help='If you have config file, input the file path')
    parser.add_argument('--saved_epoch', dest='saved_epoch', nargs='+', type=int, default=[1, ], help='The epoch number saved as check point')

    parser.add_argument('--snr_list', dest='snr_list', nargs='+', type=int, default=[100, ])
    
    save_arg = parser.add_argument_group("save information")
    save_arg.add_argument('--tag', dest='tag', nargs='+')
    save_arg.add_argument('-s', '--save', dest='best_save', action='store_true', help='Save best model state')
    save_arg.add_argument('--print_epoch', nargs='?', type=int, default=5, help='Epoch print infomation period')
    save_arg.add_argument('--log_dir', nargs='?', default='log', help='The directory name to save log file \n\t default: %(default)s')
    save_arg.add_argument('--save_patience', dest='save_log_patience', nargs='?', type=int, default=2)
    
    save_arg.add_argument('--test', dest='test', action='store_true', help='If true, model is tested after training.')
    save_arg.add_argument('--analysis', dest='analysis', action='store_true')
    save_arg.add_argument('--tsne', dest='tsne', action='store_true', help='If true, the emb of model is visualized.')
    
    train_arg = parser.add_argument_group("training parameters")
    train_arg.add_argument('-nd', '--num_device', dest='num_device', nargs='?', type=int, default=0, help='CUDA device index (default: %(default)s)')
    train_arg.add_argument('-e', '--epoch', dest='epoch', nargs='?', type=int, default=50, help='# of total epoch')
    train_arg.add_argument('--warm_up_epoch', dest='warm_up_epoch', nargs='?', type=int, default=100, help='# of total epoch for pre-training')
    train_arg.add_argument('-bs', '--batch_size', dest='batch_size', nargs='?', type=int, default=128, help='Batch size')
    train_arg.add_argument('-lr', '--learning_rate', dest='lr', nargs='?', type=float, default=1e-4, help='Learning rate')
    train_arg.add_argument('--max_lr', dest='max_lr', nargs='?', type=float, default=0.001, help='maximum learning rate in cosine lr scheduler')
    train_arg.add_argument('--scheduler', dest='scheduler', nargs='?', type=str, choices=['step', 'lambda', 'exponential', 'cosine', 'reduce'], default='reduce', help='scheduler (default: %(default)s)')
    train_arg.add_argument('--weight_decay', dest='weight_decay', nargs='?', type=float, default=6e-2)
    train_arg.add_argument('--num_workers', dest='num_workers', nargs='?', type=int, default=8)
    train_arg.add_argument('--patience', dest='patience', nargs='?', type=int, default=3)
    train_arg.add_argument('--perm', dest='perm', action='store_true')

    model_arg = parser.add_argument_group("Transformer parameters")
    model_arg.add_argument('--alpha', dest='alpha', nargs='?', type=float, default=0.1, help='hyperparameter for gating')
    model_arg.add_argument('-emb', '--embedding_dim', dest='embed_dim', nargs='?', type=int, default=256, help='d_model in transformer')
    model_arg.add_argument('-nh', '--num_heads', dest='num_heads', type=int, default=16, help='num of heads in self-attention layer')
    model_arg.add_argument('--layers', dest='num_layers', nargs='?', type=int, default=1, help='number of encoder layers')
    model_arg.add_argument('--time_layers', dest='time_num_layers', nargs='?', type=float, default=2, help='number of layers in neocortex')
    model_arg.add_argument('--patch_size', dest='patch_size', type=int, default=16, help='patch size in data stream')
    model_arg.add_argument('--mlp_ratios', dest='mlp_ratios', type=float, default=4)
    model_arg.add_argument('--max_ratio', dest='max_ratio', type=int, default=2)
    model_arg.add_argument('--connect_f', dest='connect_f', nargs='?', choices=['ADD', 'AND', 'IAND'], default='IAND', help='the types of residual connection in SNN')
    model_arg.add_argument('--gating', dest='gating', default='original', help='gating type')
    model_arg.add_argument('--keep_ratio', dest='keep_ratio', nargs='?', type=float, default=0.25, help='keep_ratio in DCT')
    
    snn_arg = parser.add_argument_group("snn parameters")
    snn_arg.add_argument('-b', '--bias', dest='bias', action=argparse.BooleanOptionalAction, help='spiking neuron bias (option: `--no-bias`)')
    snn_arg.add_argument('-spk', '--spk_encoding', dest='spk_encoding', action='store_true', help='spike encoding')
    snn_arg.add_argument('--tau', dest='tau', type=float, default=2.0)

    bank_arg = parser.add_argument_group("neocortex memory bank")
    bank_arg.add_argument('--use_neo_bank', dest='use_neo_bank', action='store_true',
                          help='enable EMA neocortex memory bank (cross-window trend prior)')
    bank_arg.add_argument('--neo_bank_decay', dest='neo_bank_decay', type=float, default=0.99,
                          help='EMA decay for the neocortex memory bank')
    bank_arg.add_argument('--neo_bank_alpha', dest='neo_bank_alpha', type=float, default=0.1,
                          help='IAND-style nudge strength applied to OFF positions of x_neo')
    bank_arg.add_argument('--neo_bank_thresh', dest='neo_bank_thresh', type=float, default=0.10,
                          help='firing-rate threshold for binarizing the bank before injection')

    kd_arg = parser.add_argument_group("teacher-student distillation")
    kd_arg.add_argument('--use_kd', dest='use_kd', action='store_true',
                        help='Hippocampus(Teacher)->Neocortex(Student) distillation: '
                             'add a student head on x_neo + KD/task losses (training only)')
    kd_arg.add_argument('--kd_weight', dest='kd_weight', type=float, default=0.5,
                        help='weight on the student (KD + task) loss term')
    kd_arg.add_argument('--kd_head_mode', dest='kd_head_mode', choices=['pooled', 'joint'],
                        default='pooled',
                        help="student head: 'pooled' (D->pred_len per token) or "
                             "'joint' (N*D->pred_len, mirrors teacher; Option-1)")

    decomp_arg0 = parser.add_argument_group('multi-scale neocortex (--model msmem)')
    decomp_arg0.add_argument('--ms_fuse', dest='ms_fuse', choices=['replay','direct','mult','xgate','inject'], default='replay', help='multi-scale Neo fusion: replay(Memory Replay IAND mask), direct(additive), mult(z=z_hippo*g_ms*ms_head), xgate(symmetric mean-gate), or inject(pure Neo memory as cross-attn K/V: drop 0.05 throttle + Hippo∩Neo mask)')
    decomp_arg0.add_argument('--ms_scales', dest='ms_scales', type=int, default=4, help='number of LIF timescale branches (1=single-tau)')
    # T=N PE-free positional study (--model tnpe)
    decomp_arg0.add_argument('--head_mode', dest='head_mode', choices=['flatten','pool'], default='flatten', help="forecast head: flatten(position-dependent) or pool(perm-invariant, position only from PE/T=N)")
    decomp_arg0.add_argument('--use_pe', dest='use_pe', action='store_true', help='add learned positional encoding to patch tokens')
    decomp_arg0.add_argument('--use_tn_pos', dest='use_tn_pos', action='store_true', help='add T=N spiking membrane positional code to patch tokens')
    decomp_arg0.add_argument('--neo_tau_pos', dest='neo_tau_pos', type=float, default=8.0, help='tau for the T=N positional membrane (probe optimum ~8)')
    decomp_arg0.add_argument('--tn_ntau', dest='tn_ntau', type=int, default=1, help='number of log-spaced taus in the T=N membrane bank (>1 = multi-scale positional code)')
    # T=N membrane Neocortex (--model neomem)
    decomp_arg0.add_argument('--mr_fusion', dest='mr_fusion', choices=['attn','gate','swap','vadd','viand','padd'], default='attn', help='Memory Replay fusion: attn(A), gate(B), swap(C), vadd/viand(D), padd(position-wise add AFTER data_block: preserves Neo T=N order, no cross-N attention mixing)')
    decomp_arg0.add_argument('--neo_plif', dest='neo_plif', action='store_true', default=False, help='learnable membrane tau (ParametricLIF) for the recurrent Neo: auto-tunes attractor memory timescale per dataset')
    decomp_arg0.add_argument('--neo_recurrent', dest='neo_recurrent', action='store_true', default=False, help='P1: re-entrant recurrent Neocortex (single-step LIF + W_rec feedback = Zipser attractor, long-term memory)')
    decomp_arg0.add_argument('--neo_out_residual', dest='neo_out_residual', action='store_true', default=False, help='review 8.7: add zero-init trend residual from Neocortex to the forecast head (y = head(hippo) + tanh(b)*trend_head(neo), b init 0)')
    decomp_arg0.add_argument('--neo_legacy', dest='neo_legacy', action='store_true', default=False, help='review 8.1: use OLD separate passive/active classes instead of the exact-passive-superset NeoLayer (default: exact superset)')
    decomp_arg0.add_argument('--wrec_mode', dest='wrec_mode', type=str, default='full', choices=['full','zero','shuffle','frozen'], help='R1 W_rec causal-isolation ablation: full | zero(remove recurrence@inference) | shuffle(permute connectivity) | frozen(recurrence present but not learned)')
    decomp_arg0.add_argument('--probe_neo', dest='probe_neo', action='store_true', default=False, help='mechanism probe: log x_neo firing/autocorr/lowfreq stats (full vs zero)')
    decomp_arg0.add_argument('--mask_past', dest='mask_past', type=int, default=0, help='task-relevant memory probe: blank the earliest k look-back steps at inference (reliance on far-past)')
    decomp_arg0.add_argument('--load_ckpt', dest='load_ckpt', type=str, default='', help='R1: load this trained checkpoint and go straight to test (no retraining); combine with --wrec_mode for inference-time W_rec ablation')
    decomp_arg0.add_argument('--no_neo', dest='no_neo', action='store_true', default=False, help='ablation: Hippocampus only (no Neocortex/Memory Replay/low-pass aux)')
    decomp_arg0.add_argument('--dual_stream', dest='dual_stream', action='store_true', default=False,
                        help='two SINGLE-LAYER nets (Hippo=freq, Neo=time) interacting only by mutual cross-attention')
    decomp_arg0.add_argument('--dual_attn', type=str, default='cross', choices=['cross','self'],
                        help='cross = streams fuse by mutual cross-attention | self = independent self-attention')
    decomp_arg0.add_argument('--neo_recall', dest='neo_recall', action='store_true', default=False,
                        help='Neo reconstructs the window from Hippo pre-prediction repr; read back by a gate (both directions stop-grad)')
    decomp_arg0.add_argument('--neo_recall_gate', type=str, default='alpha', choices=['alpha','mul','none'])
    decomp_arg0.add_argument('--neo_recall_tgt', type=str, default='raw', choices=['raw','lowfreq'])
    decomp_arg0.add_argument('--neo_mask_ratio', type=float, default=0.0,
                        help="exp A: fraction of the Neocortex's input patches to zero out; the "
                             "reconstruction aux is then scored on those held-out patches ONLY, so "
                             "it stops being an autoencoding whose answer sits in the input. 0=off.")
    decomp_arg0.add_argument('--hippo_mixer', type=str, default='attn', choices=['attn','max','reservoir'],
                        help="Hippo Block token mixer. attn(default)=spiking self/cross-attention. "
                             "max=Max-Former sliding max over the PATCH axis (Linear->BN->MaxPool(k,s=1)->LIF), "
                             "the high-pass counterpart (arXiv 2505.18608). Pure module swap; IAND kept.")
    decomp_arg0.add_argument('--hf_layers', type=int, default=2, help='MaxMixer layers')
    decomp_arg0.add_argument('--hf_kernel', type=int, default=3, help='MaxMixer pool kernel over the patch axis')
    decomp_arg0.add_argument('--no_aux', dest='no_aux', action='store_true', default=False,
                        help='the auxiliary reconstruction path does not exist: no weak_decoder, and the '
                             'Neocortex pathway is dropped entirely when it is not read back into the '
                             'prediction (gate=none, or a self-contained mixer). Equivalent to --alpha 0 '
                             'but without the 8,961 unreachable parameters and the wasted forward.')
    decomp_arg0.add_argument('--serial_mode', type=str, default='off', choices=['off','fold','skip'],
                        help="off(default)=two-pathway wiring. fold=single path, Block output pooled over the "
                             "spiking axis then straight to the head (CONTROL). skip=fold + the reservoir on the "
                             "critical path, its output added back through a zero-init gate. No auxiliary loss.")
    decomp_arg0.add_argument('--neo_xattn', dest='neo_xattn', action='store_true', default=False,
                        help="Block does TRUE cross-attention (q=Hippo, k/v=Neo) instead of self-attention. "
                             "The Neocortex input is the Block OUTPUT; the circularity is resolved by a "
                             "2-pass forward (pass 0 bootstraps from the patch embedding), identical at "
                             "train and test time. Implies condition B wiring.")
    decomp_arg0.add_argument('--refine_steps', type=int, default=1,
                        help='extra Block passes after the bootstrap pass under --neo_xattn (1 = 2 passes total)')
    decomp_arg0.add_argument('--neo_backend', type=str, default='neolayer', choices=['neolayer','reservoir'],
                        help="Neocortex core. neolayer(default)=trained non-recurrent spiking MLP over the "
                             "patch axis. reservoir=FROZEN random recurrent spiking liquid (W_in/W_rec/tau "
                             "are buffers, scan under no_grad); only the feature->embedding projection trains.")
    decomp_arg0.add_argument('--res_size', type=int, default=256, help='reservoir neurons R')
    decomp_arg0.add_argument('--res_rec_density', type=float, default=0.10)
    decomp_arg0.add_argument('--res_in_density', type=float, default=0.10)
    decomp_arg0.add_argument('--res_ei_ratio', type=float, default=0.80, help='excitatory fraction (Dale)')
    decomp_arg0.add_argument('--res_rho', type=float, default=0.90, help='target spectral radius of W_rec')
    decomp_arg0.add_argument('--res_in_scale', type=float, default=30.0,
                        help='input gain into the liquid. Calibrated so the liquid fires at ~0.22 with 7%% silent and 0%% saturated on the Hippo representation; at the code default of 1.0 it is COMPLETELY SILENT and the recurrence contrast becomes vacuous (doc reservoir_summary 12.5).')
    decomp_arg0.add_argument('--res_traces', type=int, default=2, help='filtered firing-rate traces in the readout')
    decomp_arg0.add_argument('--res_wrec', type=str, default='full', choices=['full','zero'],
                        help="zero = parameter-matched control that removes ONLY the recurrent connectivity")
    decomp_arg0.add_argument('--res_readout', type=str, default='spike', choices=['spike','analog'],
                        help="reservoir readout. spike(default)=Linear+BN+LIF, binary out, AC accounting, "
                             "but the LIF integrates along the PATCH axis and at ~2%% firing whitens the "
                             "liquid's slow structure (DCT low-freq 0.98 -> 0.24). analog=Linear+BN only, "
                             "trend preserved, key/value Linear becomes MAC.")
    decomp_arg0.add_argument('--res_seed', type=int, default=-1,
                        help='reservoir topology seed, drawn from a LOCAL generator. -1 = use --seed, so the '
                             'three run seeds also sample three topologies rather than one lucky liquid.')
    decomp_arg0.add_argument('--aux_l1', dest='aux_l1', action='store_true', default=False,
                        help='reconstruction aux uses MAE instead of MSE (was hard-wired to --neo_aux_next)')
    decomp_arg0.add_argument('--neo_recall_grad', type=str, default='stop', choices=['stop','full','to_neo','to_hippo'],
                        help="gradient across the Hippo<->Neo interface in --neo_recall mode. "
                             "stop(default)=both detached (original); full=neither detached; "
                             "to_neo=prediction loss reaches Neo (drop the Neo->Hippo detach); "
                             "to_hippo=reconstruction loss reaches Hippo (drop the Hippo->Neo detach)")
    decomp_arg0.add_argument('--dual_head', type=str, default='f', choices=['f','sum','cat','attnpool'],
                        help="how the two streams reach the head: f=stream1 only (blk_time is then DEAD) | sum (0 extra params) | cat")
    decomp_arg0.add_argument('--dual_mode', type=str, default='ft', choices=['ft','tt','ff'],
                        help="stream inputs: ft=freq+time (proposal) | tt/ff = input-only controls (identical params)")
    decomp_arg0.add_argument('--dual_dim', type=int, default=0, help='per-stream width (0 = embed_dim); capacity knob')
    decomp_arg0.add_argument('--neo_stream', dest='neo_stream', action='store_true', default=False,
                        help='two-stream: Neocortex = SAME spiking transformer as Hippocampus, fed the TIME domain')
    decomp_arg0.add_argument('--neo_stream_input', type=str, default='time', choices=['time','freq'],
                        help="'freq' = control: both streams get the wavelet bands")
    decomp_arg0.add_argument('--wavelet_enc', dest='wavelet_enc', action='store_true', default=False,
                        help='Interpretation-B: wavelet SCALE axis becomes the spiking simulation axis')
    decomp_arg0.add_argument('--wavelet_levels', type=int, default=4, help='J detail bands (= runtime T)')
    decomp_arg0.add_argument('--wavelet_name', type=str, default='b3', choices=['b3','haar'])
    decomp_arg0.add_argument('--wavelet_mode', type=str, default='wavelet', choices=['wavelet','rand'],
                        help="'rand' = control: random smoothing kernel, same band count and compute")
    decomp_arg0.add_argument('--wavelet_shuffle', dest='wavelet_shuffle', action='store_true', default=False,
                        help='control: scramble the scale ordering along the T axis')
    decomp_arg0.add_argument('--no_wavelet_neo_appr', dest='wavelet_neo_appr', action='store_false', default=True,
                        help='do NOT route the approximation band to the Neocortex')
    decomp_arg0.add_argument('--init_order_fix', dest='init_order_fix', action='store_true', default=False,
                             help='reproducibility: name-seeded (order-independent) weight init, so adding/removing\n                                   an unused module does not shift every other module''s weights')
    decomp_arg0.add_argument('--aux_membrane', dest='aux_membrane', action='store_true', default=False, help='compute the low-pass reconstruction aux from the Neocortex last-LIF MEMBRANE potential (leaky integrator=low-pass) instead of spikes')
    decomp_arg0.add_argument('--neo_tau', dest='neo_tau', type=float, default=8.0, help='tau of the T=N membrane Neocortex (probe optimum ~8)')
    decomp_arg0.add_argument('--neo_control', dest='neo_control', choices=['none','zero','shuffle','mean','randn'], default='none', help='clean causal control on the Neo membrane: none(full)|zero|shuffle(destroy position)|mean(content-free)|randn')
    decomp_arg0.add_argument('--neo_query', dest='neo_query', choices=['membrane','learn'], default='membrane', help='swap fusion query: membrane(spiking Neo) or learn(learnable per-patch prior, swap-structure analysis)')
    # causal-roll spiking encoder (--model roll)
    decomp_arg0.add_argument('--roll_mode', dest='roll_mode', choices=['roll','repeat','sine','rand'], default='roll', help='roll(linear causal shift) | repeat(identical-copy baseline) | sine(raised-cosine shift, APE-like) | rand(random past-token selection, time order preserved)')
    decomp_arg0.add_argument('--roll_boundary', dest='roll_boundary', choices=['wrap','zero','edge'], default='wrap', help='boundary for tokens lacking enough past: wrap(cyclic, default) | zero(0-fill, causal no-leakage) | edge(replicate oldest)')
    decomp_arg0.add_argument('--hippo_enc', dest='hippo_enc', choices=['repeat','sine','rand'], default='repeat', help='ours.py myModel Hippocampus spike-encoding over the spiking T axis: repeat(orig) | sine | rand (parameter-free shift-encoding, keeps CLS + low-pass aux)')
    decomp_arg0.add_argument('--hippo_boundary', dest='hippo_boundary', choices=['wrap','zero','edge'], default='wrap', help='boundary for hippo_enc shift (wrap=cyclic default)')
    decomp_arg0.add_argument('--hippo_unroll', dest='hippo_unroll', choices=['pre','post'], default='pre', help='when to unroll the hippo shift: pre(before Memory-Replay fusion, Exp A) | post(after data_block, fuse in rolled frame, Exp2)')
    decomp_arg0.add_argument('--no_overlap', dest='no_overlap', action='store_true', help='non-overlapping patches (stride=patch_size); tests if 50%% overlap was making the roll local-context redundant')
    decomp_arg0.add_argument('--no_attn', dest='no_attn', action='store_true', default=False, help='B-3: drop self-attention so the roll-LIF embedding is the ONLY inter-patch temporal mixer (pure-spiking, attention-free, O(N))')
    decomp_arg0.add_argument('--aux_next', dest='aux_next', action='store_true', default=False, help='next-token aux loss: each position token predicts the next patch (1-step future), MAE, weighted by --alpha')
    decomp_arg0.add_argument('--neo_input', dest='neo_input', choices=['patch','delta','shift'], default='patch', help='twin neocortex spike-embedding input along its time axis: patch(orig) | delta(1-step forward diff toward next patch) | shift(next patch)')
    decomp_arg0.add_argument('--neo_full_grad', dest='neo_full_grad', action='store_true', default=False, help='ours.py: full gradient to the Neocortex (remove the 0.05 Memory-Replay throttle)')
    decomp_arg0.add_argument('--neo_fuse', dest='neo_fuse', choices=['gate_attn','gate_only','gate_mix','xattn_hq','xattn_nq','cmpl','cmpl_add','bidir'], default='gate_attn', help='ours.py Neocortex fusion: gate_attn(orig) | gate_only(attn-free) | gate_mix(attn-free: neo gate + depthwise token mixing) | bidir | xattn_hq | xattn_nq | cmpl | cmpl_add')
    decomp_arg0.add_argument('--neo_seq', dest='neo_seq', choices=['identity','delta','shift','sine'], default='identity', help='ours.py Neocortex input composition along time axis N: identity | delta | shift | sine(cosine/sine-based causal selection of past tokens, APE-like)')
    decomp_arg0.add_argument('--neo_membrane_out', dest='neo_membrane_out', action='store_true', default=False, help='#1: feed LIF membrane (continuous trend) to Memory Replay instead of spikes')
    decomp_arg0.add_argument('--neo_predcode', dest='neo_predcode', action='store_true', default=False, help='#2: latent predictive-coding aux (neo predicts next-patch embedding)')
    decomp_arg0.add_argument('--neo_aux_next', dest='neo_aux_next', action='store_true', default=False, help='ours.py: turn the Neocortex reconstruction aux into NEXT-patch (1-step) prediction, MAE, weighted by --alpha')
    decomp_arg0.add_argument('--mca_swap', dest='mca_swap', action='store_true', help='swap Memory Replay cross-attention q/kv: q<-Neo memory, kv<-Hippo (Neo queries Hippo)')
    decomp_arg = parser.add_argument_group("trend/detail decomposition (--model decomp)")
    decomp_arg.add_argument('--lam_trend', dest='lam_trend', type=float, default=0.5,
                            help="weight of the trend supervision loss (trend vs low-freq(y))")
    decomp_arg.add_argument('--trend_k_ratio', dest='trend_k_ratio', type=float, default=0.25,
                            help="fraction of pred_len kept as low-freq trend coefficients K")
    decomp_arg.add_argument('--ablate_trend', dest='ablate_trend', action='store_true',
                            help="causal ablation: zero the Neocortex trend / K/V / injection (residual-only)")
    decomp_arg.add_argument('--neo_grad', dest='neo_grad', type=float, default=0.05,
                            help="fraction of task gradient allowed into the Neocortex via Memory Replay "
                                 "(0.05=original throttle, 1.0=full)")
    decomp_arg.add_argument('--neo_patch_size', dest='neo_patch_size', type=int, default=32,
                            help="coarse patch size for the global neo branch (--model mscale)")
    decomp_arg.add_argument('--xvar_mode', dest='xvar_mode', type=str, default='A',
                            help="cross-variate Memory Replay (--model xvarMR): "
                                 "'A'=faithful(neo grad throttle 0.05 + IAND gate), 'B'=de-risk(full grad + additive)")
    decomp_arg.add_argument('--t_steps', dest='t_steps', type=int, default=0,
                            help="alignment ablation (--model align): override spiking-time T (0=keep T=N). "
                                 "probe whether the spiking-time axis contributes (T=1) and if T=N is the sweet spot")

    data_arg = parser.add_argument_group("data arguments")
    data_arg.add_argument('--data', help='The name of dataset')
    data_arg.add_argument('--task_name', default='forecasting')
    data_arg.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    data_arg.add_argument('--root_path', type=str, default='data/ETT-small',
                        help='root path of the data file')
    data_arg.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    data_arg.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    data_arg.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    data_arg.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    data_arg.add_argument('--c_in', type=int, default=1)
    # forecasting lengths
    data_arg.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    data_arg.add_argument('--label_len', type=int, default=48, help='start token length')
    data_arg.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    data_arg.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    data_arg.add_argument('--sampling', dest='sampling', choices=['avg', 'min', 'cut', 'smote', 'None'], default='None')
    data_arg.add_argument('--ratio', dest='train_val_ratio', nargs='+', type=float, default=None, help='ratio for splitting training and validation(test) dataset (default:%(default)s)')
    config = parser.parse_args()
    
    return config


class Config():
    def __init__(self):
        self.date = time.strftime("%y%m%d", time.localtime(time.time()))
    
    def set_args(self, args:argparse.ArgumentParser):
        
        args.dataset = args.data_path.split('.')[0]
        
        tag = ""
        if args.tag is not None:
            for t in args.tag:
                tag = tag + f"+{t}"
                
        tag_args = ["model", "dataset", "seq_len", "pred_len", "patch_size", "keep_ratio", "embed_dim", "num_heads", "alpha", "mlp_ratios", "time_num_layers", "gating", "lr"]

        for ktag in tag_args:
            vtag = getattr(args, ktag, 'na')
            tag = tag + f"+{ktag}+{vtag}"

        # review 6.1: isolate seed + variant in a SEPARATE short path level (not appended to the
        # already-long flat tag -> avoids the 255-byte per-component filename limit). This is what
        # prevents parallel multi-seed / multi-variant runs from sharing one best+model.pt.
        _iso = "seed{}_rec{}_tau{}_wm{}".format(
            getattr(args, 'seed', 'na'), int(bool(getattr(args, 'neo_recurrent', False))),
            getattr(args, 'neo_tau', 'na'), getattr(args, 'wrec_mode', 'full'))
        self.save_result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.log_dir, args.dataset, self.date, self.date + tag, _iso)
        
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
    