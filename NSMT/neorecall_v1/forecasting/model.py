from timm.models import create_model, load_checkpoint
import os
import torch

from ours import mymodel
try:
    from ours_xvar import mymodel_xvar  # Neocortex = spiking cross-variate attention
except ImportError:
    mymodel_xvar = None
try:
    from ours_xvar_eff import mymodel_xvar_eff  # efficient O(M*k) cross-variate (latent bottleneck)
except ImportError:
    mymodel_xvar_eff = None
try:
    from ours_xvarN import mymodel_xvarN  # T=N-preserving cross-variate on Neocortex output
except ImportError:
    mymodel_xvarN = None
try:
    from ours_xvarMR import mymodel_xvarMR  # cross-variate via Memory Replay (Neocortex exposes M)
except ImportError:
    mymodel_xvarMR = None
try:
    from ours_align import mymodel_align  # alignment ablation: decouple spiking-T from token-N (--t_steps)
except ImportError:
    mymodel_align = None
try:
    from ours_hpf import mymodel_hpf  # parallel temporal high-pass branch (N-axis) in Hippocampus
except ImportError:
    mymodel_hpf = None
try:
    from ours_mscale import mymodel_mscale  # D1+D2: coarse global neo branch + additive trend
except ImportError:
    mymodel_mscale = None
try:
    from ours_neograd import mymodel_neograd  # configurable neo gradient fraction (throttle sweep)
except ImportError:
    mymodel_neograd = None
try:
    from ours_dualkv import mymodel_dualkv  # I1(a): separate hippo/neo K/V streams in Memory Replay
except ImportError:
    mymodel_dualkv = None
try:
    from ours_gauss import mymodel_gauss  # DCT-domain Gaussian taper low-freq target
except ImportError:
    mymodel_gauss = None
try:
    from ours_wo_neo import mymodel_wo_neo  # w/o Memory-Replay injection ablation
except ImportError:
    mymodel_wo_neo = None
try:
    from ours_twin import mymodel_twin  # spiking time-twin + variate-twin (iTransformer) fused
except ImportError:
    mymodel_twin = None
try:
    from ours_msmem import mymodel_msmem  # multi-scale recurrent Neocortex + Memory Replay
except ImportError:
    mymodel_msmem = None
try:
    from ours_tnpe import mymodel_tnpe  # T=N membrane PE-free positional study (2x2)
except ImportError:
    mymodel_tnpe = None
try:
    from ours_neomem import mymodel_neomem  # T=N membrane Neocortex + aux loss + full-grad Memory Replay (A/B fusion)
except ImportError:
    mymodel_neomem = None
try:
    from ours_roll import mymodel_roll  # causal-roll spiking encoder (Hippo-only + self-attn)
except ImportError:
    mymodel_roll = None
try:
    from ours_decomp import mymodel_decomp  # trend(Neo)+residual(Hippo) decomposition
except ImportError:
    mymodel_decomp = None
try:
    from ours_decomp_plif import mymodel_decomp_plif  # M1: parametric-LIF temporal block
except ImportError:
    mymodel_decomp_plif = None
try:
    from ours_decomp_vmem import mymodel_decomp_vmem  # M2+M3: membrane final-state trend readout
except ImportError:
    mymodel_decomp_vmem = None
# Optional ablation / baseline registrations. Some of these files are not
# distributed in every checkout; we keep the imports best-effort so that the
# canonical 'myModel' path always works.
try:
    from ours_degree import mymodel_degree
except ImportError:
    mymodel_degree = None
try:
    from Spikformer import Spikformer
except ImportError:
    Spikformer = None
try:
    from ours_ablation1 import mymodel_ab1
except ImportError:
    mymodel_ab1 = None
try:
    from ours_ablation1_1 import mymodel_ab1_1
except ImportError:
    mymodel_ab1_1 = None
try:
    from ours_ablation2 import mymodel_ab2
except ImportError:
    mymodel_ab2 = None
try:
    from ours_ablation3 import mymodel_ab3
except ImportError:
    mymodel_ab3 = None
try:
    from ours_ablation4 import mymodel_ab4
except ImportError:
    mymodel_ab4 = None

def load_mymodel(args, train=True, _model_name='mymodel'):
    if train:
        model = create_model(
                _model_name,
                pretrained=False,
                pretrained_cfg=None,
                pretrained_cfg_overlay=None,
                drop_rate=0.,
                drop_path_rate=0.,
                drop_block_rate=None,
                keep_ratio = args.keep_ratio,
                gating=args.gating,
                max_ratio=args.max_ratio,
                train_mode='training',
                patch_size=args.patch_size,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                pred_len=args.pred_len,
                seq_len=args.seq_len,
                qkv_bias=False, 
                mlp_ratios=args.mlp_ratios,
                depths=args.num_layers, 
                sr_ratios=1,
                time_num_layers=args.time_num_layers,
                c_in=args.c_in,
                bias=args.bias,
                tau=args.tau,
                perm=args.perm,
                spk_encoding=args.spk_encoding,
                use_neo_bank=getattr(args, 'use_neo_bank', False),
                neo_bank_decay=getattr(args, 'neo_bank_decay', 0.99),
                neo_bank_alpha=getattr(args, 'neo_bank_alpha', 0.1),
                neo_bank_thresh=getattr(args, 'neo_bank_thresh', 0.10),
                use_kd=getattr(args, 'use_kd', False),
                kd_head_mode=getattr(args, 'kd_head_mode', 'pooled'),
                ablate_trend=getattr(args, 'ablate_trend', False),
                lam_trend=getattr(args, 'lam_trend', 0.5),
                trend_k_ratio=getattr(args, 'trend_k_ratio', 0.25),
                ms_scales=getattr(args, 'ms_scales', 4),
                ms_fuse=getattr(args, 'ms_fuse', 'replay'),
                mca_swap=getattr(args, 'mca_swap', False),
                head_mode=getattr(args, 'head_mode', 'flatten'),
                use_pe=getattr(args, 'use_pe', False),
                use_tn_pos=getattr(args, 'use_tn_pos', False),
                neo_tau_pos=getattr(args, 'neo_tau_pos', 8.0),
                tn_ntau=getattr(args, 'tn_ntau', 1),
                mr_fusion=getattr(args, 'mr_fusion', 'attn'),
                neo_tau=getattr(args, 'neo_tau', 8.0),
                aux_membrane=getattr(args, 'aux_membrane', False),
                neo_recurrent=getattr(args, 'neo_recurrent', False),
                neo_exact=(not getattr(args, 'neo_legacy', False)),
                neo_out_residual=getattr(args, 'neo_out_residual', False),
                wrec_mode=getattr(args, 'wrec_mode', 'full'),
                mask_past=getattr(args, 'mask_past', 0),
                probe_neo=getattr(args, 'probe_neo', False),
                no_neo=getattr(args, 'no_neo', False),
                neo_plif=getattr(args, 'neo_plif', False),
                neo_control=getattr(args, 'neo_control', 'none'),
                neo_query=getattr(args, 'neo_query', 'membrane'),
                roll_mode=getattr(args, 'roll_mode', 'roll'),
                hippo_enc=getattr(args, 'hippo_enc', 'repeat'),
                hippo_boundary=getattr(args, 'hippo_boundary', 'wrap'),
                hippo_unroll=getattr(args, 'hippo_unroll', 'pre'),
                no_attn=getattr(args, 'no_attn', False),
                aux_next=getattr(args, 'aux_next', False),
                roll_boundary=getattr(args, 'roll_boundary', 'wrap'),
                neo_input=getattr(args, 'neo_input', 'patch'),
                neo_full_grad=getattr(args, 'neo_full_grad', False),
                neo_fuse=getattr(args, 'neo_fuse', 'gate_attn'),
                init_order_fix=getattr(args, 'init_order_fix', False),
                wavelet_enc=getattr(args,'wavelet_enc',False), wavelet_levels=getattr(args,'wavelet_levels',4),
                neo_stream=getattr(args,'neo_stream',False), neo_stream_input=getattr(args,'neo_stream_input','time'),
                dual_stream=getattr(args,'dual_stream',False), dual_dim=getattr(args,'dual_dim',0),
                dual_mode=getattr(args,'dual_mode','ft'),
                dual_head=getattr(args,'dual_head','f'),
                neo_recall=getattr(args,'neo_recall',False), neo_recall_gate=getattr(args,'neo_recall_gate','alpha'),
                neo_recall_tgt=getattr(args,'neo_recall_tgt','raw'),
                neo_recall_grad=getattr(args,'neo_recall_grad','stop'),
                dual_attn=getattr(args,'dual_attn','cross'),
                wavelet_name=getattr(args,'wavelet_name','b3'), wavelet_mode=getattr(args,'wavelet_mode','wavelet'),
                wavelet_shuffle=getattr(args,'wavelet_shuffle',False), wavelet_neo_appr=getattr(args,'wavelet_neo_appr',True),
                neo_seq=getattr(args, 'neo_seq', 'identity'),
                neo_membrane_out=getattr(args, 'neo_membrane_out', False),
                neo_predcode=getattr(args, 'neo_predcode', False),
                neo_aux_next=getattr(args, 'neo_aux_next', False),
                no_overlap=getattr(args, 'no_overlap', False),
                neo_grad=getattr(args, 'neo_grad', 0.05),
                neo_patch_size=getattr(args, 'neo_patch_size', 32),
                xvar_mode=getattr(args, 'xvar_mode', 'A'),
                t_steps=getattr(args, 't_steps', 0),
            )
    else:
        saved_model_path = os.path.join(args.save_result_path, "model_state", f"best+model.pt")
        model = create_model(
            _model_name,
            pretrained=False,
            pretrained_cfg=None,
            pretrained_cfg_overlay=None,
            checkpoint_path=saved_model_path,
            drop_rate=0.,
            keep_ratio = args.keep_ratio,
            drop_path_rate=0.,
            drop_block_rate=None,
            gating=args.gating,
            max_ratio=args.max_ratio,
            train_mode='training',
            pred_len=args.pred_len,
            seq_len=args.seq_len,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            qkv_bias=False,
            mlp_ratios=args.mlp_ratios,
            depths=args.num_layers,
            sr_ratios=1,
            time_num_layers=args.time_num_layers,
            c_in=args.c_in,
            bias=args.bias,
            tau=args.tau,
            spk_encoding=args.spk_encoding,
            use_neo_bank=getattr(args, 'use_neo_bank', False),
            neo_bank_decay=getattr(args, 'neo_bank_decay', 0.99),
            neo_bank_alpha=getattr(args, 'neo_bank_alpha', 0.1),
            neo_bank_thresh=getattr(args, 'neo_bank_thresh', 0.10),
            use_kd=getattr(args, 'use_kd', False),
            kd_head_mode=getattr(args, 'kd_head_mode', 'pooled'),
            ablate_trend=getattr(args, 'ablate_trend', False),
            lam_trend=getattr(args, 'lam_trend', 0.5),
            trend_k_ratio=getattr(args, 'trend_k_ratio', 0.25),
            xvar_mode=getattr(args, 'xvar_mode', 'A'),
            t_steps=getattr(args, 't_steps', 0),
            # review 6.5: eval branch MUST pass the same structure-determining Neocortex args
            # as the train branch, or an active/variant checkpoint loads into a passive skeleton
            # (state_dict mismatch). Keep this list in sync with the train branch above.
            neo_tau=getattr(args, 'neo_tau', 8.0),
            neo_recurrent=getattr(args, 'neo_recurrent', False),
            neo_exact=(not getattr(args, 'neo_legacy', False)),
                neo_out_residual=getattr(args, 'neo_out_residual', False),
            wrec_mode=getattr(args, 'wrec_mode', 'full'),
            no_neo=getattr(args, 'no_neo', False),
            neo_plif=getattr(args, 'neo_plif', False),
            neo_fuse=getattr(args, 'neo_fuse', 'gate_attn'),
            init_order_fix=getattr(args, 'init_order_fix', False),
                wavelet_enc=getattr(args,'wavelet_enc',False), wavelet_levels=getattr(args,'wavelet_levels',4),
                neo_stream=getattr(args,'neo_stream',False), neo_stream_input=getattr(args,'neo_stream_input','time'),
                dual_stream=getattr(args,'dual_stream',False), dual_dim=getattr(args,'dual_dim',0),
                dual_mode=getattr(args,'dual_mode','ft'),
                dual_head=getattr(args,'dual_head','f'),
                neo_recall=getattr(args,'neo_recall',False), neo_recall_gate=getattr(args,'neo_recall_gate','alpha'),
                neo_recall_tgt=getattr(args,'neo_recall_tgt','raw'),
                neo_recall_grad=getattr(args,'neo_recall_grad','stop'),
                dual_attn=getattr(args,'dual_attn','cross'),
                wavelet_name=getattr(args,'wavelet_name','b3'), wavelet_mode=getattr(args,'wavelet_mode','wavelet'),
                wavelet_shuffle=getattr(args,'wavelet_shuffle',False), wavelet_neo_appr=getattr(args,'wavelet_neo_appr',True),
            neo_membrane_out=getattr(args, 'neo_membrane_out', False),
            neo_predcode=getattr(args, 'neo_predcode', False),
            aux_membrane=getattr(args, 'aux_membrane', False),
            use_pe=getattr(args, 'use_pe', False),
            head_mode=getattr(args, 'head_mode', 'flatten'),
            mr_fusion=getattr(args, 'mr_fusion', 'attn'),
        )

        model = model.to(args.device)
        model.train_mode = 'testing'
        model.init_testing()
        print(f"{args.model} was successfully loaded. (epoch = {args.saved_epoch[-1]:03d})")

    return model


def load_mymodel_degree(args, train=True):
    if train:
        model = create_model(
                'mymodel_degree',
                pretrained=False,
                pretrained_cfg=None,
                pretrained_cfg_overlay=None,
                drop_rate=0.,
                drop_path_rate=0.,
                drop_block_rate=None,
                keep_ratio = args.keep_ratio,
                gating=args.gating,
                max_ratio=args.max_ratio,
                train_mode='training',
                patch_size=args.patch_size,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                pred_len=args.pred_len,
                seq_len=args.seq_len,
                qkv_bias=False, 
                mlp_ratios=args.mlp_ratios,
                depths=args.num_layers, 
                sr_ratios=1,
                time_num_layers=args.time_num_layers,
                c_in=args.c_in,
                bias=args.bias, 
                tau=args.tau,
                perm=args.perm,
                spk_encoding=args.spk_encoding,
            )
    else:
        saved_model_path = os.path.join(args.save_result_path, "model_state", f"best+model.pt")
        model = create_model(
            'mymodel_degree',
            pretrained=False,
            pretrained_cfg=None,
            pretrained_cfg_overlay=None,
            checkpoint_path=saved_model_path,
            drop_rate=0.,
            keep_ratio = args.keep_ratio,
            drop_path_rate=0.,
            drop_block_rate=None,
            gating=args.gating,
            max_ratio=args.max_ratio,
            train_mode='training',
            pred_len=args.pred_len,
            seq_len=args.seq_len,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            qkv_bias=False, 
            mlp_ratios=args.mlp_ratios,
            depths=args.num_layers, 
            sr_ratios=1,
            time_num_layers=args.time_num_layers,
            c_in=args.c_in,
            bias=args.bias, 
            tau=args.tau,
            spk_encoding=args.spk_encoding,
        )

        model = model.to(args.device)
        model.train_mode = 'testing'
        model.init_testing()
        print(f"{args.model} was successfully loaded. (epoch = {args.saved_epoch[-1]:03d})")
    
    return model

def load_mymodel_ab1(args, train=True):
    if train:
        model = create_model(
                'mymodel_ab1',
                pretrained=False,
                pretrained_cfg=None,
                pretrained_cfg_overlay=None,
                drop_rate=0.,
                drop_path_rate=0.,
                drop_block_rate=None,
                keep_ratio = args.keep_ratio,
                gating=args.gating,
                max_ratio=args.max_ratio,
                train_mode='training',
                patch_size=args.patch_size,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                pred_len=args.pred_len,
                seq_len=args.seq_len,
                qkv_bias=False, 
                mlp_ratios=args.mlp_ratios,
                depths=args.num_layers, 
                sr_ratios=1,
                time_num_layers=args.time_num_layers,
                c_in=args.c_in,
                bias=args.bias, 
                tau=args.tau,
                perm=args.perm,
                spk_encoding=args.spk_encoding,
            )
    else:
        saved_model_path = os.path.join(args.save_result_path, "model_state", f"best+model.pt")
        model = create_model(
            'mymodel_ab1',
            pretrained=False,
            pretrained_cfg=None,
            pretrained_cfg_overlay=None,
            checkpoint_path=saved_model_path,
            drop_rate=0.,
            keep_ratio = args.keep_ratio,
            drop_path_rate=0.,
            drop_block_rate=None,
            gating=args.gating,
            max_ratio=args.max_ratio,
            train_mode='training',
            pred_len=args.pred_len,
            seq_len=args.seq_len,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            qkv_bias=False, 
            mlp_ratios=args.mlp_ratios,
            depths=args.num_layers, 
            sr_ratios=1,
            time_num_layers=args.time_num_layers,
            c_in=args.c_in,
            bias=args.bias, 
            tau=args.tau,
            spk_encoding=args.spk_encoding,
        )

        model = model.to(args.device)
        model.train_mode = 'testing'
        model.init_testing()
        print(f"{args.model} was successfully loaded. (epoch = {args.saved_epoch[-1]:03d})")
    
    return model


def load_mymodel_ab1_1(args, train=True):
    if train:
        model = create_model(
                'mymodel_ab1_1',
                pretrained=False,
                pretrained_cfg=None,
                pretrained_cfg_overlay=None,
                drop_rate=0.,
                drop_path_rate=0.,
                drop_block_rate=None,
                keep_ratio = args.keep_ratio,
                gating=args.gating,
                max_ratio=args.max_ratio,
                train_mode='training',
                patch_size=args.patch_size,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                pred_len=args.pred_len,
                seq_len=args.seq_len,
                qkv_bias=False, 
                mlp_ratios=args.mlp_ratios,
                depths=args.num_layers, 
                sr_ratios=1,
                time_num_layers=args.time_num_layers,
                c_in=args.c_in,
                bias=args.bias, 
                tau=args.tau,
                perm=args.perm,
                spk_encoding=args.spk_encoding,
            )
    else:
        saved_model_path = os.path.join(args.save_result_path, "model_state", f"best+model.pt")
        model = create_model(
            'mymodel_ab1_1',
            pretrained=False,
            pretrained_cfg=None,
            pretrained_cfg_overlay=None,
            checkpoint_path=saved_model_path,
            drop_rate=0.,
            keep_ratio = args.keep_ratio,
            drop_path_rate=0.,
            drop_block_rate=None,
            gating=args.gating,
            max_ratio=args.max_ratio,
            train_mode='training',
            pred_len=args.pred_len,
            seq_len=args.seq_len,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            qkv_bias=False, 
            mlp_ratios=args.mlp_ratios,
            depths=args.num_layers, 
            sr_ratios=1,
            time_num_layers=args.time_num_layers,
            c_in=args.c_in,
            bias=args.bias, 
            tau=args.tau,
            spk_encoding=args.spk_encoding,
        )

        model = model.to(args.device)
        model.train_mode = 'testing'
        model.init_testing()
        print(f"{args.model} was successfully loaded. (epoch = {args.saved_epoch[-1]:03d})")
    
    return model

def load_mymodel_ab2(args, train=True):
    if train:
        model = create_model(
                'mymodel_ab2',
                pretrained=False,
                pretrained_cfg=None,
                pretrained_cfg_overlay=None,
                drop_rate=0.,
                drop_path_rate=0.,
                drop_block_rate=None,
                keep_ratio = args.keep_ratio,
                gating=args.gating,
                max_ratio=args.max_ratio,
                train_mode='training',
                patch_size=args.patch_size,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                pred_len=args.pred_len,
                seq_len=args.seq_len,
                qkv_bias=False, 
                mlp_ratios=args.mlp_ratios,
                depths=args.num_layers, 
                sr_ratios=1,
                time_num_layers=args.time_num_layers,
                c_in=args.c_in,
                bias=args.bias, 
                tau=args.tau,
                perm=args.perm,
                spk_encoding=args.spk_encoding,
            )
    else:
        saved_model_path = os.path.join(args.save_result_path, "model_state", f"best+model.pt")
        model = create_model(
            'mymodel_ab2',
            pretrained=False,
            pretrained_cfg=None,
            pretrained_cfg_overlay=None,
            checkpoint_path=saved_model_path,
            drop_rate=0.,
            keep_ratio = args.keep_ratio,
            drop_path_rate=0.,
            drop_block_rate=None,
            gating=args.gating,
            max_ratio=args.max_ratio,
            train_mode='training',
            pred_len=args.pred_len,
            seq_len=args.seq_len,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            qkv_bias=False, 
            mlp_ratios=args.mlp_ratios,
            depths=args.num_layers, 
            sr_ratios=1,
            time_num_layers=args.time_num_layers,
            c_in=args.c_in,
            bias=args.bias, 
            tau=args.tau,
            spk_encoding=args.spk_encoding,
        )

        model = model.to(args.device)
        model.train_mode = 'testing'
        model.init_testing()
        print(f"{args.model} was successfully loaded. (epoch = {args.saved_epoch[-1]:03d})")
    
    return model


def load_mymodel_ab3(args, train=True):
    if train:
        model = create_model(
                'mymodel_ab3',
                pretrained=False,
                pretrained_cfg=None,
                pretrained_cfg_overlay=None,
                drop_rate=0.,
                drop_path_rate=0.,
                drop_block_rate=None,
                keep_ratio = args.keep_ratio,
                gating=args.gating,
                max_ratio=args.max_ratio,
                train_mode='training',
                patch_size=args.patch_size,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                pred_len=args.pred_len,
                seq_len=args.seq_len,
                qkv_bias=False, 
                mlp_ratios=args.mlp_ratios,
                depths=args.num_layers, 
                sr_ratios=1,
                time_num_layers=args.time_num_layers,
                c_in=args.c_in,
                bias=args.bias, 
                tau=args.tau,
                perm=args.perm,
                spk_encoding=args.spk_encoding,
            )
    else:
        saved_model_path = os.path.join(args.save_result_path, "model_state", f"best+model.pt")
        model = create_model(
            'mymodel_ab3',
            pretrained=False,
            pretrained_cfg=None,
            pretrained_cfg_overlay=None,
            checkpoint_path=saved_model_path,
            drop_rate=0.,
            keep_ratio = args.keep_ratio,
            drop_path_rate=0.,
            drop_block_rate=None,
            gating=args.gating,
            max_ratio=args.max_ratio,
            train_mode='training',
            pred_len=args.pred_len,
            seq_len=args.seq_len,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            qkv_bias=False, 
            mlp_ratios=args.mlp_ratios,
            depths=args.num_layers, 
            sr_ratios=1,
            time_num_layers=args.time_num_layers,
            c_in=args.c_in,
            bias=args.bias, 
            tau=args.tau,
            spk_encoding=args.spk_encoding,
        )

        model = model.to(args.device)
        model.train_mode = 'testing'
        model.init_testing()
        print(f"{args.model} was successfully loaded. (epoch = {args.saved_epoch[-1]:03d})")
    
    return model


def load_mymodel_ab4(args, train=True):
    if train:
        model = create_model(
                'mymodel_ab4',
                pretrained=False,
                pretrained_cfg=None,
                pretrained_cfg_overlay=None,
                drop_rate=0.,
                drop_path_rate=0.,
                drop_block_rate=None,
                keep_ratio = args.keep_ratio,
                gating=args.gating,
                max_ratio=args.max_ratio,
                train_mode='training',
                patch_size=args.patch_size,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                pred_len=args.pred_len,
                seq_len=args.seq_len,
                qkv_bias=False, 
                mlp_ratios=args.mlp_ratios,
                depths=args.num_layers, 
                sr_ratios=1,
                time_num_layers=args.time_num_layers,
                c_in=args.c_in,
                bias=args.bias, 
                tau=args.tau,
                perm=args.perm,
                spk_encoding=args.spk_encoding,
            )
    else:
        saved_model_path = os.path.join(args.save_result_path, "model_state", f"best+model.pt")
        model = create_model(
            'mymodel_ab4',
            pretrained=False,
            pretrained_cfg=None,
            pretrained_cfg_overlay=None,
            checkpoint_path=saved_model_path,
            drop_rate=0.,
            keep_ratio = args.keep_ratio,
            drop_path_rate=0.,
            drop_block_rate=None,
            gating=args.gating,
            max_ratio=args.max_ratio,
            train_mode='training',
            pred_len=args.pred_len,
            seq_len=args.seq_len,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            qkv_bias=False, 
            mlp_ratios=args.mlp_ratios,
            depths=args.num_layers, 
            sr_ratios=1,
            time_num_layers=args.time_num_layers,
            c_in=args.c_in,
            bias=args.bias, 
            tau=args.tau,
            spk_encoding=args.spk_encoding,
        )

        model = model.to(args.device)
        model.train_mode = 'testing'
        model.init_testing()
        print(f"{args.model} was successfully loaded. (epoch = {args.saved_epoch[-1]:03d})")
    
    return model


def load_spikformer(args, train=True):
    model = Spikformer(
            drop_rate=0.,
            drop_path_rate=0.,
            drop_block_rate=None,
            pred_len=args.pred_len,
            seq_len=args.seq_len,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            qkv_bias=False, 
            mlp_ratios=args.mlp_ratios,
            depths=args.num_layers, 
            sr_ratios=1,
            c_in=args.c_in,
            bias=args.bias, 
            tau=args.tau,
            spk_encoding=args.spk_encoding,
        )
    if not train:
        saved_model_path = os.path.join(args.save_result_path, "model_state", f"best+model.pt")
        model.load_state_dict(torch.load(saved_model_path))
        model = model.to(args.device)
        
        print(f"{args.model} was successfully loaded. (epoch = {args.saved_epoch[-1]:03d})")
        
    return model
        
# Teacher(Hippocampus)->Student(Neocortex) variant (self-attention + KD,
# student-only inference). Architecture lives in ours_ts.py / model_ts.py.
try:
    from model_ts import load_mymodel_ts
except ImportError:
    load_mymodel_ts = None

def load_mymodel_wo_neo(args, train=True):
    # w/o Memory-Replay injection ablation; same pipeline as myModel.
    return load_mymodel(args, train, _model_name='mymodel_wo_neo')


def load_mymodel_xvar(args, train=True):
    # Neocortex = spiking cross-variate attention; same pipeline as myModel.
    return load_mymodel(args, train, _model_name='mymodel_xvar')


def load_mymodel_xvar_eff(args, train=True):
    # efficient O(M*k) cross-variate (latent bottleneck); same pipeline as myModel.
    return load_mymodel(args, train, _model_name='mymodel_xvar_eff')


def load_mymodel_xvarMR(args, train=True):
    return load_mymodel(args, train, _model_name='mymodel_xvarMR')

def load_mymodel_align(args, train=True):
    return load_mymodel(args, train, _model_name='mymodel_align')

def load_mymodel_xvarN(args, train=True):
    # T=N-preserving cross-variate on Neocortex output; same pipeline as myModel.
    return load_mymodel(args, train, _model_name='mymodel_xvarN')


def load_mymodel_hpf(args, train=True):
    # parallel temporal high-pass branch in Hippocampus; same pipeline as myModel.
    return load_mymodel(args, train, _model_name='mymodel_hpf')


def load_mymodel_mscale(args, train=True):
    # D1+D2: coarse global neo branch + additive trend; same pipeline as myModel.
    return load_mymodel(args, train, _model_name='mymodel_mscale')


def load_mymodel_neograd(args, train=True):
    # configurable neo gradient fraction (--neo_grad); same pipeline as myModel.
    return load_mymodel(args, train, _model_name='mymodel_neograd')


def load_mymodel_dualkv(args, train=True):
    # I1(a): separate hippo/neo K/V streams in Memory Replay; same pipeline as myModel.
    return load_mymodel(args, train, _model_name='mymodel_dualkv')


def load_mymodel_gauss(args, train=True):
    # DCT-domain Gaussian taper for the low-freq auxiliary target; same pipeline as myModel.
    return load_mymodel(args, train, _model_name='mymodel_gauss')


def load_mymodel_twin(args, train=True):
    # spiking time-twin + variate-twin fused; same pipeline as myModel.
    return load_mymodel(args, train, _model_name='mymodel_twin')


def load_mymodel_msmem(args, train=True):
    return load_mymodel(args, train, _model_name='mymodel_msmem')


def load_mymodel_tnpe(args, train=True):
    return load_mymodel(args, train, _model_name='mymodel_tnpe')


def load_mymodel_neomem(args, train=True):
    return load_mymodel(args, train, _model_name='mymodel_neomem')


def load_mymodel_roll(args, train=True):
    return load_mymodel(args, train, _model_name='mymodel_roll')


def load_mymodel_decomp(args, train=True):
    # trend(Neocortex)+residual(Hippocampus) decomposition; same pipeline as myModel.
    return load_mymodel(args, train, _model_name='mymodel_decomp')


def load_mymodel_decomp_plif(args, train=True):
    # M1: decomposition with parametric-LIF (learnable tau) temporal block.
    return load_mymodel(args, train, _model_name='mymodel_decomp_plif')


def load_mymodel_decomp_vmem(args, train=True):
    # M2+M3: decomposition with membrane final-state trend readout.
    return load_mymodel(args, train, _model_name='mymodel_decomp_vmem')


LOAD_MODEL = {
    'myModel' : load_mymodel,
    'xvar' : load_mymodel_xvar,
    'xvar_eff' : load_mymodel_xvar_eff,
    'xvarN' : load_mymodel_xvarN,
    'xvarMR' : load_mymodel_xvarMR,
    'align' : load_mymodel_align,
    'hpf' : load_mymodel_hpf,
    'mscale' : load_mymodel_mscale,
    'neograd' : load_mymodel_neograd,
    'dualkv' : load_mymodel_dualkv,
    'gauss' : load_mymodel_gauss,
    'wo_neo' : load_mymodel_wo_neo,
    'decomp' : load_mymodel_decomp,
    'msmem' : load_mymodel_msmem,
    'tnpe' : load_mymodel_tnpe,
    'neomem' : load_mymodel_neomem,
    'roll' : load_mymodel_roll,
    'twin' : load_mymodel_twin,
    'decomp_plif' : load_mymodel_decomp_plif,
    'decomp_vmem' : load_mymodel_decomp_vmem,
    'ts' : load_mymodel_ts,
    'degree' : load_mymodel_degree,
    'Spikformer' : load_spikformer,
    'ab1' : load_mymodel_ab1,
    'ab1_1' : load_mymodel_ab1_1,
    'ab2' : load_mymodel_ab2,
    'ab3' : load_mymodel_ab3,
    'ab4' : load_mymodel_ab4,
}