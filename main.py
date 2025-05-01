import torch
import numpy as np
import os
import random
import argparse
from exp import Exp
from graph_transfer import GraphTransferExp
from logger import create_logger
from utils.config import load_config, save_config, list2str
from utils.eval_utils import set_seed


set_seed(3047)

parser = argparse.ArgumentParser(description='')

# Experiment settings
parser.add_argument('--task', type=str, default='Transfer',
                    choices=['NC', 'Transfer'])
parser.add_argument('--dataset', type=str, default='line',
                    help="[Wisconsin, Texas, Cornell]")
parser.add_argument('--root_path', type=str, default='./datasets')
parser.add_argument('--val_every', type=int, default=10)
parser.add_argument('--exp_iters', type=int, default=10)
parser.add_argument('--log_path', type=str, default="./results/Physics.log")
parser.add_argument('--task_model_path', type=str)  # necessary
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

# Base Params
parser.add_argument('--add_self_loop', action='store_false', help='add self loop to adjacency')
parser.add_argument('--n_layers', type=int, default=5)
parser.add_argument('--hid_dim', type=int, default=32, help='hidden dimension')
parser.add_argument('--embed_dim', type=int, default=32, help='embedding dimension')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--act', type=str, default='gelu', help='activation function')
parser.add_argument('--input_act', type=str, default='gelu', help='activation function for input layer')
parser.add_argument('--norm', type=str, default='ln', help='Normalization of Batch Norm or Layer Norm')
parser.add_argument('--bias', action='store_true', help='use bias for linear layer')

# Node Classification
parser.add_argument('--lr_nc', type=float, default=3e-5)
parser.add_argument('--weight_decay_nc', type=float, default=0)
parser.add_argument('--epochs_nc', type=int, default=2000)
parser.add_argument('--patience_nc', type=int, default=100)

# Graph Transfer
parser.add_argument('--additional_layers', type=int, default=0)
parser.add_argument('--distance_list', type=int, nargs="+", default=[50, 10, 5, 3])
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr_trans', type=float, default=1e-3)
parser.add_argument('--weight_decay_trans', type=float, default=0)
parser.add_argument('--epochs_trans', type=int, default=2000)
parser.add_argument('--patience_trans', type=int, default=100)

# GPU
parser.add_argument('--use_gpu', action='store_false', help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multiple gpus')

configs = parser.parse_args()
results_dir = f"./results/logs"
log_path = f"{results_dir}/{configs.task}_{configs.dataset}.log"
configs.log_path = log_path
if not os.path.exists(results_dir):
    os.makedirs(results_dir, exist_ok=True)
if configs.task_model_path is None:
    configs.task_model_path = f"{configs.task}_{configs.dataset}_model.pt"
json_dir = f"./configs/{configs.task}"
json_path = f"{json_dir}/{configs.dataset}.json"
if not os.path.exists(json_dir):
    os.makedirs(json_dir, exist_ok=True)
# times_dir = f"./results/times"
# if not os.path.exists(times_dir):
#     os.makedirs(times_dir, exist_ok=True)

# print(f"Saving config file: {json_path}")
# save_config(vars(configs), json_path)
if os.path.exists(json_path):
    print(f"Loading config file: {json_path}")
    configs = load_config(vars(configs), json_path)

print(f"Log path: {configs.log_path}")
logger = create_logger(configs.log_path)
logger.info(configs)

if configs.task == "NC":
    exp = Exp(configs)
elif configs.task == "Transfer":
    exp = GraphTransferExp(configs)
else:
    raise NotImplementedError
exp.train()
torch.cuda.empty_cache()