from argparse import ArgumentParser
import torch
import torch.nn as nn
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall

import os
from sys import stdout

# snn
from spikingjelly.clock_driven import functional
from syops import get_model_complexity_info
from syops.ops import LIF_syops_counter_hook                    # nsmt_flif_v1
from neurons import LIFNeuron, FLIFNeuron                        # nsmt_flif_v1
from energy import FiringRateRecorder                             # nsmt_flif_v1

# model
from timm import create_model
from timm.optim.optim_factory import create_optimizer_v2

from config import set_random_seed, parse_arguments, Config
from utils import get_pred, get_scheduler, get_class_weights, get_energy_consumption, tsne_visual, plot_eval
from dataloader import create_loader
# from data_provider.data_factory import data_provider

import model as model
from load_model import LOAD_MODEL
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def set_shuffle_replay_flag(model, enabled=False):
    hit = []
    for name, module in model.named_modules():
        # replay를 소유한 Block만 잡기
        if hasattr(module, "mca") and hasattr(module, "shuffle_replay"):
            module.shuffle_replay = enabled
            hit.append(name)

    print(f"[shuffle_replay={enabled}] patched modules = {hit}")
    if enabled and len(hit) == 0:
        print("[Warning] No Block with mca + shuffle_replay was found.")


def load_model(args):
    saved_model_path = os.path.join(args.save_result_path, "model_state", f"best+model.pt")
    
    spikformer = create_model(
        'spikformer',
        pretrained=False,
        pretrained_cfg=None,
        pretrained_cfg_overlay=None,
        checkpoint_path=saved_model_path,
        drop_rate=0.,
        drop_path_rate=0.,
        drop_block_rate=None,
        gating=args.gating,
        train_mode='training',
        seq_len=args.seq_len,
        data_patch_size=args.data_patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        qkv_bias=False, 
        mlp_ratios=args.mlp_ratios,
        depths=args.num_layers, 
        sr_ratios=1,
        time_num_layers=args.time_num_layers,
        T=args.time_steps, 
        lif_bias=args.bias, 
        data_patching_stride=args.stride,
        num_channels=args.num_channels,
        padding_patches=None,
        tau=args.tau,
        spk_encoding=args.spk_encoding,
        attn=args.attn, 
        keep_ratio=args.keep_ratio,
    )

    spikformer = spikformer.to(args.device)
    functional.reset_net(spikformer)

    
    print(f"Model was successfully loaded. (epoch = {args.saved_epoch[-1]:03d})")
    
    return spikformer


def tsne_plot_2d(features, labels, save_path=None, title="t-SNE (2D)", random_state=42):
    """
    features: [N, D] (np.ndarray or torch.Tensor)
    labels:   [N,]   (np.ndarray or torch.Tensor)
    save_path: 저장 경로 (예: 'tsne.png'), None이면 화면 표시
    """
    # torch 텐서면 numpy로 변환
    try:
        import torch
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
    except ImportError:
        pass

    features = np.asarray(features)
    labels   = np.asarray(labels)

    # 간단한 검증
    assert features.ndim == 2, f"features shape must be [N, D], got {features.shape}"
    assert labels.ndim == 1 and labels.shape[0] == features.shape[0], f"labels must be [N,] and match N, got {labels.shape} and features shape {features.shape}"

    # t-SNE 실행 (perplexity는 N보다 작아야 함)
    from sklearn.manifold import TSNE
    N = features.shape[0]
    perp = max(5, min(30, N - 1))  # 아주 단순한 안전 설정
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto",
                random_state=random_state)
    X2 = tsne.fit_transform(features)

    # 시각화
    plt.figure(figsize=(8, 6))
    for ul in np.unique(labels):
        m = labels == ul
        plt.scatter(X2[m, 0], X2[m, 1], s=10, alpha=0.75, label=str(ul))
    plt.title(title)
    plt.xticks([]); plt.yticks([])
    # 클래스 수가 적으면 범례 표시
    if len(np.unique(labels)) <= 20:
        plt.legend(title="Class", markerscale=2, frameon=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def evaluation(args:Config, model=None):
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    
    loader, class_num_samples = create_loader(train=False, num_classes=args.num_classes, batch_size=args.batch_size, data_root=args.root_path+'/5class_mitbih_test_3_187.csv', data_window_size=187, num_workers=args.num_workers) 

    overall_acc = MulticlassAccuracy(num_classes=args.num_classes).to(args.device)
    acc = MulticlassAccuracy(num_classes=args.num_classes, average=None).to(args.device)
    f1 = MulticlassF1Score(num_classes=args.num_classes, average='macro').to(args.device)
    pre = MulticlassPrecision(num_classes=args.num_classes, average='macro').to(args.device)
    re = MulticlassRecall(num_classes=args.num_classes, average='macro').to(args.device)
    f1_cls = MulticlassF1Score(num_classes=args.num_classes, average=None).to(args.device)   # nsmt_flif_v1
    conf = torch.zeros(args.num_classes, args.num_classes, dtype=torch.long)                 # nsmt_flif_v1: rows = true class
    
    if model is None:
        model = LOAD_MODEL[args.model](args, False)

    model.eval()
    model.train_mode = 'testing'

    # if hasattr(spikformer, 'weak_decoder'): delattr(spikformer, 'weak_decoder')
    # if hasattr(spikformer, 'replay') : delattr(spikformer, 'replay')
 
    set_shuffle_replay_flag(model, getattr(args, "shuffle_replay", False))

    with torch.no_grad():
        epoch_loss = 0
        sim = 0
        all_features = []
        all_labels = []
        rec = FiringRateRecorder(model)                              # nsmt_flif_v1: neuron-intrinsic energy
        functional.reset_net(model)
        for batch in loader:
            data = batch[0].float().to(args.device)
            label = batch[1].long().squeeze(-1).to(args.device)
                
            output = model(data)
            rec.add_samples(data.shape[0])
            if isinstance(output, tuple):
                output, feature = output
                all_features.append(feature.transpose(0,1).mean(1).mean(1).flatten(1).detach().cpu())
                all_labels.append(label.detach().cpu())

            loss = criterion(output, label)
            epoch_loss += loss.item()
            
            functional.reset_net(model)

            pred = get_pred(output)
            
            overall_acc.update(pred, label)
            acc.update(pred, label)
            f1.update(pred, label)
            pre.update(pred, label)
            re.update(pred, label)
            f1_cls.update(pred, label)
            conf += torch.bincount((label * args.num_classes + pred).cpu(), minlength=args.num_classes ** 2).reshape(args.num_classes, args.num_classes)
        rec.detach()                                                 # before syops re-runs the loader

        # t-SNE plot after all batches
        # if all_features:
        #     all_features = torch.cat(all_features, dim=0).numpy()
        #     all_labels = torch.cat(all_labels, dim=0).numpy()
        #     tsne_plot_2d(all_features, all_labels, save_path=os.path.join(args.save_log_path, "tsne_plot.png"))
            
        input_res = data[0].shape
        
        model_info_per_layer_path = os.path.join(args.save_log_path, 'model+info+per+layer.txt') 
        file_out = open(model_info_per_layer_path, 'w', encoding='utf-8')
       
        model.train_mode = 'testing'
        functional.reset_net(model)
        ops, params, fr = get_model_complexity_info(
                                    model=model,
                                    input_res=(input_res,), 
                                    dataloader=loader,
                                    as_strings=False,
                                    print_per_layer_stat=True,
                                    custom_modules_hooks={LIFNeuron: LIF_syops_counter_hook, FLIFNeuron: LIF_syops_counter_hook},   # nsmt_flif_v1
                                    # ignore_modules=ignore_modules,
                                    verbose=False,
                                    ost=file_out,
                                )
        
        file_out.close()
        
    test_result = {
        'loss' : epoch_loss/len(loader),
        'overall_acc' : overall_acc.compute(),
        'acc' : acc.compute(),
        'f1' : f1.compute(),
        'pre' : pre.compute(),
        're' : re.compute(),
        'sim' : sim/len(loader)
    }
    
    print("Test was successfully done")

    # nsmt_flif_v1: per-class F1, the neuron-intrinsic energy columns and n_test are appended AFTER the original columns
    f1c = f1_cls.compute()
    en = rec.summary(syn_energy_J=get_energy_consumption(O_ac=ops[1], O_mac=ops[2]) * 1e-12)
    rec.write_layers(os.path.join(args.save_log_path, 'neuron_energy.csv'))
    with open(os.path.join(args.save_log_path, 'confusion_matrix.csv'), 'w', encoding='utf-8') as f:
        f.write('true\\pred,' + ','.join(str(i) for i in range(args.num_classes)) + '\n')
        for i in range(args.num_classes):
            f.write(f'{i},' + ','.join(str(int(v)) for v in conf[i]) + '\n')

    with open(args.save_log_path + '/final+result.csv', 'a', encoding='utf-8') as log_csv:
        acc_col = ', '.join(f"acc{i}" for i in range(args.num_classes))
        f1_col = ', '.join(f"f1_{i}" for i in range(args.num_classes))
        print("shuffle_replay", "loss", "overall_acc", "average_acc", "f1", "precision", "sensitivity", "total_op", "ACop", "MACop", "capacity", "firing_rate", "energy", acc_col,
              f1_col, ', '.join(en['columns']), "n_test", sep=", ", end="\n", file=log_csv)
        
        acc_value = ', '.join(f"{test_result['acc'][i]:.6f}" for i in range(args.num_classes))
        f1_value = ', '.join(f"{f1c[i]:.6f}" for i in range(args.num_classes))
            
        print(f"{getattr(args, 'shuffle_replay', False)}",
              f"{test_result['loss']:.6f}", 
              f"{test_result['overall_acc']:.6f}",
              f"{test_result['acc'].clone().mean():.6f}", 
              f"{test_result['f1']:.6f}", 
              f"{test_result['pre']:.6f}", 
              f"{test_result['re']:.6f}",
            #   f"{test_result['sim']:.6f}",
              f"{ops[0] / 1e6:.2f} M Ops",
              f"{ops[1] / 1e6:.2f} M Ops",
              f"{ops[2] / 1e6:.2f} M Ops",
              f"{params / 1e6:.4f} M",
              f"{fr:.4f} %",
              f"{get_energy_consumption(O_ac=ops[1], O_mac=ops[2], unit='u'):.2f} uJ",
              acc_value,
              f1_value, ', '.join(en['values']), str(int(conf.sum())),
              sep=", ", end="\n", file=log_csv)
        
    print(f"Final result saved to `{args.save_log_path}`")
    
    # savefig_path = args.save_log_path
    
    # spikformer.eval()
    # spikformer.train_mode = 'visual'
    # spikformer.head = nn.Identity()
    
    # plot_eval(model=spikformer, loader=loader, num_classes=args.num_classes, save_path=savefig_path, device=args.device)
    
    # print(f"Final tsne result saved to `{savefig_path}`")

def test(args, test, only_path_test):
    
    # TODO: criterion

    evaluation(args=args)
    
        
if __name__ == '__main__':
    
    
    config = parse_arguments()
    args = Config()
    
    if config.config:
        config_path = config.config
        args.load_args(config_path, config)
    
    else:
        args.set_args(config)
        
    set_random_seed(args.seed)
    args.print_info()
    base_save_log_path = args.save_log_path

    # test(args, config.test, config.only_path_test)

    for shuffle_replay in [False, True]:
        setattr(args, 'shuffle_replay', shuffle_replay)

        exp_name = "full" if not shuffle_replay else "shifted_replay"
        args.save_log_path = os.path.join(base_save_log_path, exp_name)
        os.makedirs(args.save_log_path)
        test(args, config.test, config.only_path_test)

    torch.set_grad_enabled(True)
    
    