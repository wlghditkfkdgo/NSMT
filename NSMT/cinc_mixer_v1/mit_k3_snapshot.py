"""check_tune_mit.py k3 helper (2026-11-01, exp/f-lif-pop-v3): build the Spikformer-style MIT-BIH model (spk_ssa, seed 42, the
run_mitbih_spk.sh / run_mitbih2.sh arguments) with the code of a given classification folder -- this branch's cinc_mixer_v1 or the
other session's nsmt_flif_v1/classification exported at the baseline commit (read only: run with `python -B`; every output goes to
<log dir>) -- in train.py's order (seed, train / test loaders, model), and save the initial state_dict, the first test batch, the
eval-mode output, one training-mode output and the CE gradients.
usage: python -B mit_k3_snapshot.py <classification dir> <out.pt> <scratch log dir> <MIT-BIH data dir>
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT, OUT, LOGD, DATA = (os.path.abspath(p) for p in sys.argv[1:5])
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]   # import only from ROOT
sys.path.insert(0, ROOT)
sys.argv = ['train.py', '--model', 'myModel', '--seed', '42', '--log_dir', LOGD, '--gating', 'attn', '--attn', 'SSA', '-s', '--test',
            '--print_epoch', '1', '-nd', '0', '-e', '30', '--warm_up_epoch', '0', '--num_channels', '1', '-bs', '512', '-lr', '0.01',
            '--max_lr', '0.005', '-emb', '384', '-nh', '8', '--mlp_ratios', '1', '--layers', '1', '--keep_ratio', '0.15', '--time_layers', '2',
            '--alpha', '0.1', '--patch_size', '32', '--no-bias', '--scheduler', 'cosine', '--data', 'UEA', '--data_path', 'ECG',
            '--seq_len', '187', '--root_path', DATA, '--model', 'myModel', '--hippo_ssa', '--hippo_T', '4', '--tag', 'spk_ssa', 'seed42']

import torch                                                              # noqa: E402
import train as TR                                                        # noqa: E402  (ROOT's train.py, config, model, dataloader ...)
from spikingjelly.clock_driven import functional                          # noqa: E402

if os.path.dirname(os.path.abspath(TR.__file__)) != ROOT:
    raise SystemExit(f'train imported from {TR.__file__}, not {ROOT}')
args = TR.Config(); args.set_args(TR.parse_arguments())
TR.set_random_seed(args.seed)
loader, cns = TR.create_loader(train=True, batch_size=args.batch_size, num_classes=args.num_classes, data_root=DATA + '/5class_mitbih_train_7_187.csv',
                               train_val_ratio=None, data_window_size=187, sampling='None', num_workers=args.num_workers)
val_loader, _ = TR.create_loader(train=False, batch_size=args.batch_size, num_classes=args.num_classes, data_root=DATA + '/5class_mitbih_test_3_187.csv',
                                 data_window_size=187, num_workers=args.num_workers)
model = TR.LOAD_MODEL[args.model](args, True).to(args.device)
functional.reset_net(model)
sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
batch = next(iter(val_loader))
x, y = batch[0], batch[1]
xd, yd = x.float().to(args.device), y.long().to(args.device)
model.eval(); model.train_mode = 'testing'
with torch.no_grad():
    o = model(xd)
out_eval = (o[0] if isinstance(o, tuple) else o).cpu()
functional.reset_net(model)
model.train(); model.train_mode = 'training'
o = model(xd)
z = o[0] if isinstance(o, tuple) else o
torch.nn.functional.cross_entropy(z, yd).backward()
functional.reset_net(model)
grads = {k: p.grad.detach().cpu().clone() for k, p in model.named_parameters() if p.grad is not None}
torch.save(dict(sd=sd, x=x, y=y, out_eval=out_eval, out_train=z.detach().cpu(), grads=grads, class_counts=list(map(int, cns)),
                n_test=len(val_loader.dataset), source=TR.__file__), OUT)
print('saved', OUT, len(sd), 'tensors', len(grads), 'grads', 'class counts', list(map(int, cns)))
