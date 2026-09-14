import os
import torch

from f_lif_pop_v2.forecasting.ours import myModel


def load_mymodel(args, train=True):
    model = myModel(**{key: getattr(args, key) for key in [
        'seq_len', 'pred_len', 'patch_size', 'embed_dim', 'num_population',
        'head_dim', 'head_mode', 'heterogeneous', 'retrieval', 'tau_min', 'tau_max',
        'threshold', 'memory_strength', 'temperature', 'input_scale', 'architecture', 'read_mode', 'null_logit_init', 'num_heads']})
    if not train:
        path = os.path.join(args.save_model_state_path, 'best+model.pt')
        model.load_state_dict(torch.load(path, map_location='cpu'))
    return model.to(args.device)


LOAD_MODEL = {'myModel': load_mymodel}
