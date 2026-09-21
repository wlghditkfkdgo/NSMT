import torch
from torch.utils.data import DataLoader

from .synthetic import Dataset_Recall
from config import set_seed_worker


def data_provider(args, flag):
    if args.task == 'recall':
        build = Dataset_Recall
    else:
        from .data_loader import Dataset_ETT_hour     # pandas는 ETT 경로에서만 필요하다
        build = Dataset_ETT_hour
    data_set = build(args, flag=flag)
    generator = torch.Generator().manual_seed(args.seed + 1000)
    data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=(flag == 'train'),
                             num_workers=args.num_workers, drop_last=False,
                             pin_memory=(args.device.type == 'cuda'),
                             worker_init_fn=set_seed_worker, generator=generator)
    print(flag, len(data_set))
    return data_set, data_loader
