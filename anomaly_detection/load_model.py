from timm.models import create_model, load_checkpoint
import os
import torch

from model import mymodel
from model_wo_mem import mymodel_ab1_1

def load_mymodel(args, train=True):
    if train:
        model = create_model(
                'mymodel',
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
            )
    else:
        saved_model_path = os.path.join(args.save_result_path, "model_state", f"best+model.pt")
        model = create_model(
            'mymodel',
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


LOAD_MODEL = {
    'myModel' : load_mymodel,
    'ab2' : load_mymodel_ab1_1,
}