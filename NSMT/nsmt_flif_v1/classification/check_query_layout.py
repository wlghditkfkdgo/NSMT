# MODI 2026-11-17: read-only check of the hippocampus / Neocortex coupling in the query form (user question 11-17); not imported by training.
"""Read-only check (no repo edit): in the msn base model, is the query input of the hippocampus attention exactly
'Neocortex output [N, B', 1, D] repeated 4 times along the repeat axis and laid out as [4, B', N, D]'?
Builds the model as train_forecast.py does (same flags as run_ett_dual.sh), CPU, spikingjelly backend 'torch'.
usage: python check_query_layout.py <pop_lif|pop_flif> <ETTh2|ETTm2> <seq_len> <pred_len> <trained best+model.pt> <classification dir>
The trained weights are loaded (strict) so that the Neocortex actually fires; with all-zero spikes any layout would pass."""
import sys, torch
n, ds, L, H, ckpt, cls_dir = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
sys.path.insert(0, cls_dir)
HP = {'ETTh2': '--patch_size 8 -emb 64 -nh 8 --mlp_ratios 0.5 --time_layers 1 -lr 0.0005 -bs 64',
      'ETTm2': '--patch_size 8 -emb 64 -nh 16 --mlp_ratios 1 --time_layers 2 -lr 0.0005 -bs 64'}[ds]
CF = {'pop_lif': '--neo_neuron pop_lif --neo_where both --pop_soma_init 0.5',
      'pop_flif': '--neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0'}[n]
argv = (f'--seed 42 -s -nd 0 --gating attn --no-bias --scheduler reduce -e 1 --warm_up_epoch 0 --layers 1 --tau 2.0 '
        f'--weight_decay 0.06 --data {ds} --data_path {ds}.csv --root_path /nonexistent --seq_len {L} '
        f'{HP} --model myModel {CF} --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms').split()
sys.argv = ['x'] + argv
from config import Config, parse_arguments, set_random_seed
from load_model import LOAD_MODEL
from forecast_wrap import Forecaster
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
args = Config(); args.set_args(parse_arguments())
args.num_classes, args.num_channels = 1, 1
set_random_seed(42)
model = Forecaster(LOAD_MODEL[args.model](args, True), args.embed_dim, H, bias=args.bias, seq_len=args.seq_len)
for m in model.modules():
    if isinstance(m, MultiStepLIFNode):
        m.backend = 'torch'                      # CPU run; same LIF equations
sd = torch.load(ckpt, map_location='cpu')
model.load_state_dict(sd, strict=True)
model.eval(); functional.reset_net(model)
inner = model.inner
blk = inner.data_block[0]
cap = {}
inner.time_block.register_forward_hook(lambda m, i, o: cap.__setitem__('neo', o.detach().clone()))
inner.encoding.register_forward_hook(lambda m, i, o: cap.__setitem__('hip_emb', (o[0] if isinstance(o, tuple) else o).detach().clone()))
blk.ssa.q_linear.register_forward_pre_hook(lambda m, i: cap.__setitem__('q_in', i[0].detach().clone()))
blk.ssa.k_linear.register_forward_pre_hook(lambda m, i: cap.__setitem__('k_in', i[0].detach().clone()))
blk.ssa.v_linear.register_forward_pre_hook(lambda m, i: cap.__setitem__('v_in', i[0].detach().clone()))
B, C = 2, 7
torch.manual_seed(0)
x = torch.randn(B, L, C).cumsum(1)               # random-walk input [B, L, C]
with torch.no_grad():
    y = model(x)
neo, q_in = cap['neo'], cap['q_in']
N, Bp, one, D = neo.shape
T_h = inner.T_h
print(f'{n} {ds} L={L}: N={N} B\'={Bp} D={D} T_h={T_h}  neo out {tuple(neo.shape)}  q_linear input {tuple(q_in.shape)}  '
      f'neo spike rate {neo.mean():.4f}  output {tuple(y.shape)}')
assert (N, Bp, one, D) == (L // 4, B * C, 1, 64) and T_h == 4
# what the user asks for: repeat 4 times, then lay out as [4, B', N, D]
want = neo.repeat(1, 1, T_h, 1).permute(2, 1, 0, 3).contiguous()     # [N,B',1,D] -> [N,B',4,D] -> [4,B',N,D]
got = q_in.reshape(T_h, Bp, N, D)                                     # the tensor the query Linear actually reads
ok1 = torch.equal(got, want)
ok2 = all(torch.equal(got[t, b, i], neo[i, b, 0]) for t in range(T_h) for b in range(Bp) for i in range(N))
wrong = neo.repeat(1, 1, T_h, 1).reshape(T_h, Bp, N, D)               # a layout error of the kind one would worry about
hip = cap['hip_emb'].reshape(T_h * Bp, N, D)
ok3 = torch.equal(cap['k_in'], hip) and torch.equal(cap['v_in'], hip)
print(f'  query input == Neocortex output repeated 4x as [4, B\', N, D]: {ok1}')
print(f'  element by element (step t, row b, token i) == Neocortex token i of row b: {ok2}')
print(f'  key / value input == hippocampus embedding spikes [4, B\', N, D]: {ok3}')
print(f'  control: a plain reshape without the transpose would differ: {not torch.equal(got, wrong)}')
diff = (got != wrong).any(-1).float().mean().item()
print(f'  control: share of (step, row, token) positions where that wrong layout differs: {diff:.3f}')
print('PASS' if (neo.sum() > 0) and (ok1 and ok2 and ok3 and not torch.equal(got, wrong)) else 'FAIL')
