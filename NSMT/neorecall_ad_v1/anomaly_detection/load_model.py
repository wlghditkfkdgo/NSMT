from timm.models import create_model, load_checkpoint
import os
import torch

from model import mymodel
try:
    from ours_wo_neo import mymodel_wo_neo  # w/o Memory-Replay injection ablation
except ImportError:
    mymodel_wo_neo = None
try:
    from model_wo_mem import mymodel_ab1_1
except ImportError:
    mymodel_ab1_1 = None
# Teacher(Hippocampus)->Student(Neocortex) variant (self-attention + KD,
# student-only inference). Architecture lives in model_ts.py.
try:
    from model_ts import mymodel_ts
except ImportError:
    mymodel_ts = None


def load_mymodel_ts(args, train=True):
    """Loader for the Teacher->Student AD variant (mirrors load_mymodel)."""
    common = dict(
        keep_ratio=args.keep_ratio, gating=args.gating, train_mode='training',
        patch_size=args.patch_size, embed_dim=args.embed_dim,
        num_heads=args.num_heads, pred_len=args.pred_len, seq_len=args.seq_len,
        qkv_bias=False, mlp_ratios=args.mlp_ratios, depths=args.num_layers,
        sr_ratios=1, time_num_layers=args.time_num_layers, c_out=args.c_out,
        bias=args.bias, tau=args.tau, features=args.features,
        spk_encoding=args.spk_encoding,
        kd_head_mode=getattr(args, 'kd_head_mode', 'joint'),
    )
    if train:
        model = create_model('mymodel_ts', pretrained=False, pretrained_cfg=None,
                             pretrained_cfg_overlay=None, drop_rate=0.,
                             drop_path_rate=0., drop_block_rate=None, **common)
    else:
        saved_model_path = os.path.join(args.save_result_path, "model_state", f"best+model.pt")
        model = create_model('mymodel_ts', pretrained=False, pretrained_cfg=None,
                             pretrained_cfg_overlay=None, checkpoint_path=saved_model_path,
                             drop_rate=0., drop_path_rate=0., drop_block_rate=None, **common)
        model = model.to(args.device)
        model.train_mode = 'testing'
        model.init_testing()
        print(f"{args.model} was successfully loaded. (epoch = {args.saved_epoch[-1]:03d})")
    return model

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
                train_mode='training',
                patch_size=args.patch_size,
                neo_recall=getattr(args,'neo_recall',False), neo_recall_gate=getattr(args,'neo_recall_gate','alpha'),
                neo_recall_next=getattr(args,'neo_recall_next',True),
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                pred_len=args.pred_len,
                seq_len=args.seq_len,
                qkv_bias=False, 
                mlp_ratios=args.mlp_ratios,
                depths=args.num_layers, 
                sr_ratios=1,
                time_num_layers=args.time_num_layers,
                c_out=args.c_out,
                bias=args.bias, 
                tau=args.tau,
                features=args.features,
                spk_encoding=args.spk_encoding,
                roll_mode=getattr(args, 'roll_mode', 'none'),
                neo_recurrent=getattr(args, 'neo_recurrent', False),
                no_neo=getattr(args, 'no_neo', False),
                neo_tau=getattr(args, 'neo_tau', 2.0),
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
            train_mode='training',
            pred_len=args.pred_len,
            seq_len=args.seq_len,
            patch_size=args.patch_size,
            neo_recall=getattr(args,'neo_recall',False), neo_recall_gate=getattr(args,'neo_recall_gate','alpha'),
            neo_recall_next=getattr(args,'neo_recall_next',True),
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            qkv_bias=False, 
            mlp_ratios=args.mlp_ratios,
            depths=args.num_layers, 
            sr_ratios=1,
            time_num_layers=args.time_num_layers,
            c_out=args.c_out,
            bias=args.bias, 
            tau=args.tau,
            features=args.features,
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
                train_mode='training',
                patch_size=args.patch_size,
                neo_recall=getattr(args,'neo_recall',False), neo_recall_gate=getattr(args,'neo_recall_gate','alpha'),
                neo_recall_next=getattr(args,'neo_recall_next',True),
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                pred_len=args.pred_len,
                seq_len=args.seq_len,
                qkv_bias=False, 
                mlp_ratios=args.mlp_ratios,
                depths=args.num_layers, 
                sr_ratios=1,
                time_num_layers=args.time_num_layers,
                c_out=args.c_out,
                bias=args.bias, 
                tau=args.tau,
                features=args.features,
                spk_encoding=args.spk_encoding,
                roll_mode=getattr(args, 'roll_mode', 'none'),
                neo_recurrent=getattr(args, 'neo_recurrent', False),
                no_neo=getattr(args, 'no_neo', False),
                neo_tau=getattr(args, 'neo_tau', 2.0),
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
            train_mode='training',
            pred_len=args.pred_len,
            seq_len=args.seq_len,
            patch_size=args.patch_size,
            neo_recall=getattr(args,'neo_recall',False), neo_recall_gate=getattr(args,'neo_recall_gate','alpha'),
            neo_recall_next=getattr(args,'neo_recall_next',True),
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            qkv_bias=False, 
            mlp_ratios=args.mlp_ratios,
            depths=args.num_layers, 
            sr_ratios=1,
            time_num_layers=args.time_num_layers,
            c_out=args.c_out,
            bias=args.bias, 
            tau=args.tau,
            features=args.features,
            spk_encoding=args.spk_encoding,
        )

        model = model.to(args.device)
        model.train_mode = 'testing'
        model.init_testing()
        print(f"{args.model} was successfully loaded. (epoch = {args.saved_epoch[-1]:03d})")
    
    return model


def load_mymodel_wo_neo(args, train=True):
    # w/o Memory-Replay injection ablation; same pipeline as myModel.
    return load_mymodel(args, train, _model_name='mymodel_wo_neo')


LOAD_MODEL = {
    'myModel' : load_mymodel,
    'wo_neo' : load_mymodel_wo_neo,
    'ts' : load_mymodel_ts,
    'ab2' : load_mymodel_ab1_1,
}