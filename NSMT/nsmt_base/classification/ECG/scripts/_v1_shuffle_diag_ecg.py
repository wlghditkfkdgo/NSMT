"""V1 patch-shuffle diagnostic for MIT-BIH ECG (5-class).
Loads trained ECG_4 baseline (myModel) into mymodel_shuf, shuffles the Neocortex
patch order, measures x_neo change + classification Acc/F1 change.
Unlike UEA classification, ECG classes differ by heartbeat WAVEFORM MORPHOLOGY
(temporal shape), so order may matter here.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import numpy as np
import torch
from timm.models import create_model
import ours_shuf  # registers mymodel_shuf
from spikingjelly.clock_driven import functional
from dataloader import create_loader
from torchmetrics.classification import MulticlassF1Score

CKPT = glob.glob('./ECG/260612/*myModel+keep_ratio+0.0+patch_size+32*/model_state/best+model.pt')
TEST_CSV = './data/5class_mitbih_test_3_187.csv'


def build():
    return create_model('mymodel_shuf', pretrained=False, pretrained_cfg=None,
        pretrained_cfg_overlay=None, drop_rate=0., drop_path_rate=0., drop_block_rate=None,
        keep_ratio=0.0, gating='attn', train_mode='training', data_patch_size=32,
        embed_dim=384, num_heads=8, num_classes=5, num_channels=1, seq_len=187,
        qkv_bias=False, mlp_ratios=1.0, depths=1, sr_ratios=1, time_num_layers=2,
        bias=False, tau=2.0, spk_encoding=False)


def fwd(m, x, shuffle):
    m.shuffle_neo = shuffle
    functional.reset_net(m)
    out = m(x)
    return out[0], out[2]   # logits [B,5], x_neo


m = build()
m.load_state_dict(torch.load(CKPT[0], map_location='cpu'), strict=False)
m = m.cuda().eval(); m.train_mode = 'training'
ld, _ = create_loader(train=False, batch_size=512, num_classes=5, data_root=TEST_CSV,
                      data_window_size=187, num_workers=0)

rx = []; rz = []; n = 0; c0 = 0; c1 = 0
f1_0 = MulticlassF1Score(num_classes=5, average='macro').cuda()
f1_1 = MulticlassF1Score(num_classes=5, average='macro').cuda()
with torch.no_grad():
    for batch in ld:
        x = batch[0].float().cuda(); y = batch[1].long().squeeze(-1).cuda()
        z0, nn0 = fwd(m, x, False); z1, nn1 = fwd(m, x, True)
        rx.append((torch.norm(nn1 - nn0) / (torch.norm(nn0) + 1e-8)).item())
        rz.append((torch.norm(z1 - z0) / (torch.norm(z0) + 1e-8)).item())
        c0 += (z0.argmax(-1) == y).sum().item(); c1 += (z1.argmax(-1) == y).sum().item()
        f1_0.update(z0.argmax(-1), y); f1_1.update(z1.argmax(-1), y)
        n += y.numel()

print("MIT-BIH ECG (5-class) V1 patch-shuffle:")
print(f"  rel_xneo = {np.mean(rx):.4f}   rel_z(logit) = {np.mean(rz):.4f}")
print(f"  Acc  orig {c0/n:.4f} -> shuf {c1/n:.4f}   (dAcc {c1/n - c0/n:+.4f})")
print(f"  F1   orig {f1_0.compute().item():.4f} -> shuf {f1_1.compute().item():.4f}   "
      f"(dF1 {f1_1.compute().item() - f1_0.compute().item():+.4f})")
