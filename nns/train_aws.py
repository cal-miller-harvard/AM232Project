import argparse
import copy
import os

import torch
import numpy as np

ttype = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor
torch.set_default_tensor_type(ttype)

from lqg_env import LQG
from util import seiler_state_space, StateSpace, train_model
from controllers import RNNController

def save_best_model(best_model, save_dir, name, idx=None):
    best_loss, best_state, best_initial_state = best_model
    idx_str = '' if idx is None else f'_{idx}'

    with open(os.path.join(save_dir, f'{name}_best{idx_str}'), 'wb') as f:
        torch.save(best_state, f)

    with open(os.path.join(save_dir, f'{name}_best_initial{idx_str}'), 'wb') as f:
        torch.save(best_initial_state, f)


if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--simulation-steps', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--num-inits', type=int, default=200)

    parser.add_argument('--controller', type=str, default='rnn')
    parser.add_argument('--rnn-num-layers', type=int, default=3)
    parser.add_argument('--rnn-dim-hidden', type=int, default=10)

    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    args, _ = parser.parse_known_args()

    # get current index across all distributed jobs
    idx = None
    if getattr(args, 'distributed', False):
        idx = args.hosts.index(args.current_host)

    if args.controller == 'rnn':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        controller = RNNController(1, 1, dim_h=args.rnn_dim_hidden, layers=args.rnn_num_layers).to(device)
    else:
        raise Exception('Controller not found')

    ss, Q, R = seiler_state_space()
    env = LQG(ss, Q, R, batch_size=args.batch_size)

    best = (np.inf, None, None)
    opt = torch.optim.Adam(controller.parameters(), lr=args.learning_rate)

    for i in range(args.num_inits):
        controller.init_weights()
        initial_state = copy.deepcopy(controller.state_dict())

        losses = train_model(controller, env, opt, args.epochs, args.simulation_steps)

        if losses[-1] < best[0]:
            best = (losses[-1], copy.deepcopy(controller.state_dict()), initial_state)

        if i % 20 == 0:
            save_best_model(best, args.model_dir, args.controller, idx=idx)
