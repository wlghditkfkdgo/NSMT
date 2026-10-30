"""check_ett_mux.py t1 helper (2026-10-31, exp/f-lif-pop-v3): build the Spikformer-style ETT forecaster (spk_ssa, ETTh2 H96, seed 42,
the run_ett_spk.sh arguments) with the code of a given classification folder -- this branch's cinc_mixer_v1 or the other session's
nsmt_flif_v1/classification (read only: run with `python -B` so no bytecode is written there; every output goes to <log dir>) --
and save the initial state_dict, the first test batch, the eval-mode output, one training-mode output and the L1 gradients.
usage: python -B ett_t1_snapshot.py <classification dir> <out.pt> <scratch log dir> <ETT-small dir>
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT, OUT, LOGD, ETT = (os.path.abspath(p) for p in sys.argv[1:5])
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]   # import only from ROOT
sys.path.insert(0, ROOT)
sys.argv = ['train_forecast.py', '--data', 'ETTh2', '--data_path', 'ETTh2.csv', '--root_path', ETT, '--seq_len', '96', '--pred_len', '96',
            '--features', 'M', '--target', 'OT', '--log_dir', LOGD, '--seed', '42', '-s', '-nd', '0', '--gating', 'attn', '--no-bias',
            '--scheduler', 'reduce', '-e', '100', '--warm_up_epoch', '0', '--layers', '1', '--tau', '2.0', '--weight_decay', '0.06',
            '--patience', '10', '--patch_size', '8', '-emb', '64', '-nh', '8', '--mlp_ratios', '0.5', '--time_layers', '1', '-lr', '0.0005',
            '-bs', '64', '--model', 'myModel', '--hippo_ssa', '--hippo_T', '4', '--tag', 'spk_ssa', 'seed42']

import torch                                                              # noqa: E402
import train_forecast as TF                                               # noqa: E402  (ROOT's train_forecast, config, model, ...)
from spikingjelly.clock_driven import functional                          # noqa: E402

if os.path.dirname(os.path.abspath(TF.__file__)) != ROOT:
    raise SystemExit(f'train_forecast imported from {TF.__file__}, not {ROOT}')
fc = TF.split_args()
args = TF.Config(); args.set_args(TF.parse_arguments())
for k, v in vars(fc).items():
    setattr(args, k, v)
args.num_classes, args.num_channels = 1, 1
TF.set_random_seed(args.seed)
da = TF.data_args(args, fc)
_, tr = TF.data_provider(da, 'train'); _, va = TF.data_provider(da, 'val'); _, te = TF.data_provider(da, 'test')
model = TF.Forecaster(TF.LOAD_MODEL[args.model](args, True), args.embed_dim, fc.pred_len, bias=args.bias).to(args.device)
functional.reset_net(model)
sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
x, y = next(iter(te))[:2]
xd, yd = x.float().to(args.device), y.float().to(args.device)[:, -fc.pred_len:, :]
model.eval(); model.train_mode = 'testing'
with torch.no_grad():
    out_eval = model(xd)
functional.reset_net(model)
model.train(); model.train_mode = 'training'
out_tr = model(xd)
torch.nn.functional.l1_loss(out_tr, yd).backward()
functional.reset_net(model)
grads = {k: p.grad.detach().cpu().clone() for k, p in model.named_parameters() if p.grad is not None}
torch.save(dict(sd=sd, x=x, y=y, out_eval=out_eval.cpu(), out_train=out_tr.detach().cpu(), grads=grads, kind=model.kind,
                n=(len(tr.dataset), len(va.dataset), len(te.dataset)), source=TF.__file__), OUT)
print('saved', OUT, model.kind, len(sd), 'tensors', len(grads), 'grads')
