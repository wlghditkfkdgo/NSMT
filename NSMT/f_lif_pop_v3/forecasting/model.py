import os

import torch

from ours import myModel, GRUBaseline

NEURON_KEYS = ['num_population', 'alpha', 'tau', 'heterogeneous', 'max_length', 'theta',
               'eta_init', 'eta_fixed', 'cap', 'tau_s', 'threshold', 'surrogate_scale']
SHAPE_KEYS = ['task', 'seq_len', 'pred_len', 'patch_size', 'embed_dim', 'head_dim',
              'head_mode', 'readout', 'input_scale', 'input_norm']


def _restore(model, args, train):
    if not train:
        path = os.path.join(args.save_model_state_path, 'best+model.pt')
        # strict=True: frozen normalisation buffers must come back with the weights,
        # otherwise a reloaded model silently runs at a different operating point.
        model.load_state_dict(torch.load(path, map_location='cpu'), strict=True)

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
