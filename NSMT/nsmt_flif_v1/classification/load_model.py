from timm.models import create_model, load_checkpoint
import os
import torch

from model import myModel
# nsmt_flif_v1: `from ab4 import myModelAB4` dropped (the ab4 ablation model is not copied; ECG/ab4.py keeps it)

def _neo_kwargs(args):
    """nsmt_flif_v1: neocortex-neuron options (defaults = the ECG model of nsmt_base/classification/ECG)."""
    return dict(neo_neuron=getattr(args, 'neo_neuron', 'sj_lif'), neo_where=getattr(args, 'neo_where', 'both'),
                neo_alpha=getattr(args, 'neo_alpha', 0.5), fde_method=getattr(args, 'fde_method', 'pred'),
                fde_step=getattr(args, 'fde_step', 1.0), fde_memory=getattr(args, 'fde_memory', 0),
                hippo_only=getattr(args, 'hippo_only', False),
                neo_bidir=getattr(args, 'neo_bidir', False),                                        # stage 6
                in_channels=getattr(args, 'num_channels', 1), hippo_ssa=getattr(args, 'hippo_ssa', False),   # stage 9
                hippo_T=getattr(args, 'hippo_T', 0), pop_soma_init=getattr(args, 'pop_soma_init', None),
                pop_taus=tuple(float(t) for t in str(getattr(args, 'pop_taus', '4,8,16,32')).split(',')))


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
                data_patch_size=args.patch_size,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                num_classes=args.num_classes,
                num_channels=args.num_channels,
                seq_len=args.seq_len,
                qkv_bias=False, 
                mlp_ratios=args.mlp_ratios,
                depths=args.num_layers, 
                sr_ratios=1,
                time_num_layers=args.time_num_layers,
                bias=args.bias, 
                tau=args.tau,
                spk_encoding=args.spk_encoding,
                roll_mode=getattr(args, 'roll_mode', 'none'),
                neo_recurrent=getattr(args, 'neo_recurrent', False),
                neo_exact=(not getattr(args, 'neo_legacy', False)),
                neo_rec_gate=getattr(args, 'neo_rec_gate', False),
                no_neo=getattr(args, 'no_neo', False),
                neo_tau=getattr(args, 'neo_tau', 2.0),
                **_neo_kwargs(args),                               # nsmt_flif_v1
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
            num_classes=args.num_classes,
            num_channels=args.num_channels,
            seq_len=args.seq_len,
            data_patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            qkv_bias=False, 
            mlp_ratios=args.mlp_ratios,
            depths=args.num_layers, 
            sr_ratios=1,
            time_num_layers=args.time_num_layers,
            bias=args.bias, 
            tau=args.tau,
            spk_encoding=args.spk_encoding,
            roll_mode=getattr(args, 'roll_mode', 'none'),
                neo_recurrent=getattr(args, 'neo_recurrent', False),
                neo_exact=(not getattr(args, 'neo_legacy', False)),
                neo_rec_gate=getattr(args, 'neo_rec_gate', False),
                no_neo=getattr(args, 'no_neo', False),
                neo_tau=getattr(args, 'neo_tau', 2.0),
                **_neo_kwargs(args),                               # nsmt_flif_v1
        )

        model = model.to(args.device)
        model.train_mode = 'testing'
        # model.init_testing()
        print(f"{args.model} was successfully loaded. (epoch = {args.saved_epoch[-1]:03d})")
    
    return model

# def load_spikformer(args, train=True):
#     model = Spikformer(
#             drop_rate=0.,
#             drop_path_rate=0.,
#             drop_block_rate=None,
#             num_classes=args.num_classes,
#             num_channels=args.num_channels,
#             seq_len=args.seq_len,
#             patch_size=args.patch_size,
#             embed_dim=args.embed_dim,
#             num_heads=args.num_heads,
#             qkv_bias=False, 
#             mlp_ratios=args.mlp_ratios,
#             depths=args.num_layers, 
#             sr_ratios=1,
#             c_in=args.c_in,
#             bias=args.bias, 
#             tau=args.tau,
#             spk_encoding=args.spk_encoding,
#             roll_mode=getattr(args, 'roll_mode', 'none'),
#             neo_recurrent=getattr(args, 'neo_recurrent', False),
#             neo_rec_gate=getattr(args, 'neo_rec_gate', False),
#             no_neo=getattr(args, 'no_neo', False),
#             neo_tau=getattr(args, 'neo_tau', 2.0),
#         )
#     if not train:
#         saved_model_path = os.path.join(args.save_result_path, "model_state", f"best+model.pt")
#         model.load_state_dict(torch.load(saved_model_path))
#         model = model.to(args.device)
        
#         print(f"{args.model} was successfully loaded. (epoch = {args.saved_epoch[-1]:03d})")
        
#     return model
        
def load_neo_mixer(args, train=True):
    """stage 7: the attention-free, T = N aligned model (neo_mixer.NeoMixer)."""
    from neo_mixer import NeoMixer
    model = NeoMixer(seq_len=args.seq_len, data_patch_size=args.patch_size, embed_dim=args.embed_dim, num_classes=args.num_classes,
                     depths=args.num_layers, mlp_ratios=args.mlp_ratios, mixer=getattr(args, 'mixer', 'pop_flif'),
                     mixer_bidir=getattr(args, 'mixer_bidir', False), mixer_layers=args.time_num_layers,
                     neo_tau=getattr(args, 'neo_tau', 2.0), neo_alpha=getattr(args, 'neo_alpha', 0.5),
                     fde_method=getattr(args, 'fde_method', 'pred'), fde_step=getattr(args, 'fde_step', 1.0),
                     fde_memory=getattr(args, 'fde_memory', 0),
                     pop_taus=tuple(float(t) for t in str(getattr(args, 'pop_taus', '4,8,16,32')).split(',')),
                     bias=args.bias, tau=args.tau, train_mode='training',
                     in_channels=getattr(args, 'num_channels', 1), pop_soma_init=getattr(args, 'pop_soma_init', None))   # stage 9
    if not train:
        saved_model_path = os.path.join(args.save_result_path, "model_state", "best+model.pt")
        model.load_state_dict(torch.load(saved_model_path, map_location='cpu'))
        model = model.to(args.device)
        model.train_mode = 'testing'
        print(f"{args.model} was successfully loaded. (epoch = {args.saved_epoch[-1]:03d})")
    return model


LOAD_MODEL = {
    'myModel' : load_mymodel,
    'neoMixer' : load_neo_mixer,                                       # stage 7
    # 'Spikformer' : load_spikformer, 
}