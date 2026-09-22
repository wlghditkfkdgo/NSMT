import os

import torch

from ours import myModel, GRUBaseline

NEURON_KEYS = ['num_population', 'alpha', 'tau', 'heterogeneous', 'max_length', 'theta',
               'eta_init', 'eta_fixed', 'cap', 'tau_s', 'threshold', 'surrogate_scale']
SHAPE_KEYS = ['task', 'seq_len', 'pred_len', 'patch_size', 'embed_dim', 'head_dim',
              'head_mode', 'readout', 'input_scale', 'input_norm']


# 뒤에 추가된 buffer들. 기본값이 항등이라 없으면 옛 동작과 같다.
OPTIONAL_BUFFERS = ('embedding.neuron.selector.key_mean', 'embedding.neuron.selector.key_std',
                    'embedding.norm_mean', 'embedding.norm_std',
                    'embedding.neuron.selector.eta_value')


def _restore(model, args, train):
    if not train:
        path = os.path.join(args.save_model_state_path, 'best+model.pt')
        state = torch.load(path, map_location='cpu')
        missing, unexpected = model.load_state_dict(state, strict=False)
        # strict=False로 열되 무엇이 빠졌는지 직접 검사한다. frozen 통계가 조용히
        # 항등으로 되돌아가면 재로드 모델이 다른 동작점에서 돌게 된다.
        surprising = [k for k in missing if k not in OPTIONAL_BUFFERS]
        if surprising or unexpected:
            raise RuntimeError(f'checkpoint schema mismatch: missing {surprising}, '
                               f'unexpected {list(unexpected)}')
        if missing:
            print(f"[model] checkpoint predates {len(missing)} optional buffer(s) "
                  f"({', '.join(missing)}); they keep their identity defaults")

    return model.to(args.device)


def load_mymodel(args, train=True):
    from config import neuron_kwargs
    shape = {key: getattr(args, key) for key in SHAPE_KEYS}

    return _restore(myModel(**shape, **neuron_kwargs(args)), args, train)


def load_gru(args, train=True):
    shape = {key: getattr(args, key) for key in SHAPE_KEYS if key not in ('readout', 'input_scale',
                                                                          'input_norm')}
    return _restore(GRUBaseline(**shape), args, train)


LOAD_MODEL = {'myModel': load_mymodel, 'GRU': load_gru}
