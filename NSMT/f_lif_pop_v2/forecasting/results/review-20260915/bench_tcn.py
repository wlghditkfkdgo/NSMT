"""Throughput probe only: synthetic batches, same model code and determinism flags as config.py; writes nothing."""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch  # noqa: E402

sys.path.insert(0, '/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT')
from f_lif_pop_v2.forecasting.ours import myModel  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='tcn')
parser.add_argument('--policy', default='off')
parser.add_argument('--iters', type=int, default=8)
parser.add_argument('--tag', default='')
args = parser.parse_args()

torch.manual_seed(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)
torch.set_num_threads(2)
device = torch.device('cuda:0')
model = myModel(architecture=args.arch, heterogeneous=True, retrieval=args.policy != 'off',
                read_mode='dense' if args.policy == 'dense' else 'sparse').to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
x, y = torch.randn(128, 336, 7, device=device), torch.randn(128, 96, 7, device=device)


def step():
    loss = (model(x) - y).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    loss.item()


for _ in range(3):
    step()
torch.cuda.synchronize()
start = time.time()
for _ in range(args.iters):
    step()
torch.cuda.synchronize()
print(json.dumps({'tag': args.tag, 'arch': args.arch, 'policy': args.policy,
                  'sec_per_train_iter': round((time.time() - start) / args.iters, 4),
                  'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}), flush=True)
