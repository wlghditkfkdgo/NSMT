"""Reload a selected config/checkpoint in a fresh process without rewriting logs."""
import argparse
import json
from types import SimpleNamespace

import numpy as np

from config import Config, TASK, set_random_seed
from test import test
from utils import write_json


def check(suite, pred_len, device):
    root = TASK / 'results' / suite
    run_id = f'ETTh1_p{pred_len}_flatten_heterogeneous_retrieval_seed7'
    original = json.loads((root / (run_id + '.json')).read_text())
    args = Config()
    args.load_args(original['config']['save_result_path'],
                   SimpleNamespace(cpu=device == 'cpu', num_device=int(device.split(':')[-1]) if device != 'cpu' else 0))
    set_random_seed(args.seed)
    actual = test(args, save=False)
    for metric in ['mse', 'mae']:
        np.testing.assert_allclose(actual[metric], original['test'][metric], rtol=0, atol=1e-12)
        for mode in ['off', 'uniform', 'recent']:
            np.testing.assert_allclose(actual['interventions'][mode][metric],
                                       original['test']['interventions'][mode][metric], rtol=0, atol=1e-12)
    result = {'status': 'passed', 'run_id': run_id, 'device': device,
              'checks': ['fresh config/checkpoint reload', 'whole-test MSE/MAE',
                         'whole-test off/uniform/recent MSE/MAE'], 'absolute_tolerance': 1e-12,
              'original_logs_rewritten': False}
    write_json(root / 'check_reload.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', required=True)
    parser.add_argument('--pred_len', required=True, type=int)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    check(args.suite, args.pred_len, args.device)
