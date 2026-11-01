"""check_tune2.py c3 helper (2026-11-01, exp/f-lif-pop-v3): build the Spikformer-style CinC model (spk_ssa, fold 0, seed 42, the
run_rhythm_spk_mux.sh / run_rhythm.sh arguments, lr 1e-3) with the code of a given classification folder -- this branch's
cinc_mixer_v1 or the other session's nsmt_flif_v1/classification exported at the baseline commit 9094bfaf9 (read only: run with
`python -B`; its dataset/ must point to the same CinC cache; every output goes to <log dir>) -- in train_rhythm.py's order (seed,
rhythm_data.build, model), and save the initial state_dict, the first 16 validation windows, the eval / training outputs and the
CAL-weighted CE gradients.
usage: python -B cinc_c3_snapshot.py <classification dir> <out.pt> <scratch log dir>
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT, OUT, LOGD = (os.path.abspath(p) for p in sys.argv[1:4])
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]   # import only from ROOT
sys.path.insert(0, ROOT)
sys.argv = ['train_rhythm.py', '--data', 'CINC2017', '--seed', '42', '--log_dir', LOGD, '--gating', 'attn', '-s', '-nd', '0', '--warm_up_epoch', '0',
            '-emb', '384', '-nh', '8', '--mlp_ratios', '1', '--layers', '1', '--time_layers', '2', '--keep_ratio', '0.15', '--alpha', '0.1',
            '--tau', '2.0', '--no-bias', '--scheduler', 'cosine', '-bs', '32', '--eval_bs', '32', '-lr', '0.001', '--weight_decay', '1e-4',
            '--max_lr', '0.005', '--patch_size', '64', '-e', '30', '--loss_kind', 'cal', '--fold', '0',
            '--model', 'myModel', '--hippo_ssa', '--hippo_T', '4', '--tag', 'spk_ssa', 'lr0.001', 'seed42']

import torch                                                              # noqa: E402
import train_rhythm as TR                                                 # noqa: E402  (ROOT's train_rhythm, config, model, rhythm_data ...)
from spikingjelly.clock_driven import functional                          # noqa: E402

if os.path.dirname(os.path.abspath(TR.__file__)) != ROOT:
    raise SystemExit(f'train_rhythm imported from {TR.__file__}, not {ROOT}')
args = TR.Config(); args.set_args(TR.parse_arguments())
TR.set_random_seed(args.seed)
data = TR.rd.build(args, args.seed)
args.num_classes, args.num_channels, args.seq_len = data['num_classes'], data['num_channels'], data['seq_len']
model = TR.LOAD_MODEL[args.model](args, True).to(args.device)
functional.reset_net(model)
sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
xs, ys = zip(*[data['val'][i][:2] for i in range(16)])
x = torch.stack([torch.as_tensor(v) for v in xs]).float(); y = torch.as_tensor([int(v) for v in ys])
xd, yd = x.to(args.device), y.to(args.device)
model.eval(); model.train_mode = 'testing'
with torch.no_grad():
    o = model(xd)
out_eval = (o[0] if isinstance(o, tuple) else o).cpu()
functional.reset_net(model)
model.train(); model.train_mode = 'training'
o = model(xd)
z = o[0] if isinstance(o, tuple) else o
w = TR.cal_weights(data['counts']).to(args.device)
torch.nn.functional.cross_entropy(z, yd, weight=w).backward()
functional.reset_net(model)
grads = {k: p.grad.detach().cpu().clone() for k, p in model.named_parameters() if p.grad is not None}
torch.save(dict(sd=sd, x=x, y=y, out_eval=out_eval, out_train=z.detach().cpu(), grads=grads, info=str(data['info']), source=TR.__file__), OUT)
print('saved', OUT, len(sd), 'tensors', len(grads), 'grads', data['info'])
