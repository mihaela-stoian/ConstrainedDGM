import datetime
import os
import sys
import warnings
import argparse 

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append('.')


from cdgm.data_processors.wgan.helpers import prepare_data_torch_scaling
from cdgm.synthetizers.WGAN.wgan import SingleTaskDataset, train_model
from utils import _load_json, read_csv, set_seed
from run_dgm.run_utils import evaluate_model_performance, runtime_evaluation, add_common_args, generate_evaluation_data, set_pac

warnings.filterwarnings(action='ignore')
torch.set_printoptions(sci_mode=False)

import wandb


DATETIME = datetime.datetime.now()

def get_args():
    parser = argparse.ArgumentParser(description='C-DGMs Command Line Interface')
    parser.add_argument("--model_type", default="WGAN", type=str)

    add_common_args(parser)
    parser.add_argument("--pac", default=10, type=int)
    parser.add_argument("--optimiser", default='rmsprop', type=str, choices=['rmsprop', 'adam', 'sgd'])
    parser.add_argument("--disc_repeats", default=2, type=int)
    parser.add_argument("--gp_weight", default=10, type=float)
    parser.add_argument("--d_lr", default=0.00005, type=float)
    parser.add_argument("--g_lr", default=0.00005, type=float)
    parser.add_argument("--alpha", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--momentum", default=0, type=float)  # 0.00005
    parser.add_argument("--clamp", default=None, type=float)  # 0.01
    parser.add_argument("--enable_sigmoid", default=False, action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    set_pac(args)
    args.exp_id = f"{args.version}_{args.label_ordering}_{args.seed}_{args.epochs}_{args.batch_size}_{args.disc_repeats}_{args.gp_weight}_{args.d_lr}_{args.g_lr}_{args.enable_sigmoid}_{DATETIME:%d-%m-%y--%H-%M-%S}"
    args.path = f"outputs/{args.model_type}/{args.use_case}/{args.version}/{args.exp_id}"
    os.makedirs(args.path)

   ######################################################################
    wandb_run = wandb.init(project=args.wandb_project, id=args.exp_id, reinit=True,  mode=args.wandb_mode)
    for k,v in args._get_kwargs():
        wandb_run.config[k] = v
    ######################################################################
    args.constraints_file = f'./data/{args.use_case}/{args.use_case}_constraints.txt'
    dataset_info = _load_json("data/datasets_info.json")[args.use_case]
    print(dataset_info)
    ######################################################################

    X_train, (cat_cols, cat_idx), (roundable_idx, round_digits) = read_csv(f"data/{args.use_case}/train_data.csv", args.use_case, dataset_info["manual_inspection_categorical_cols_idx"])

    if args.use_case == "lcld":
        X_test = pd.read_csv(f"data/{args.use_case}/tiny/test_data.csv")
        X_val = pd.read_csv(f"data/{args.use_case}/tiny/val_data.csv")     
    else:
        X_test = pd.read_csv(f"data/{args.use_case}/test_data.csv")
        X_val = pd.read_csv(f"data/{args.use_case}/val_data.csv")

    real_data = {"train": X_train, "val": X_val, "test": X_test}

    columns = X_train.columns.values.tolist()
    args.dtypes = X_train.dtypes
    args.train_data_cols = columns

    X_train_scaled, scaler = prepare_data_torch_scaling(X_train, args.use_case, cat_idx)

    train_ds = SingleTaskDataset(X_train_scaled)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    args.input_length = X_train_scaled.shape[1]
    args.bin_cols_idx = cat_idx


    print("Created new directory {:}, the experiment is starting".format(args.path))
    model = train_model(args, train_loader, scaler, args.path, X_train, X_test)


    if args.runtime_evaluation_only:
        runtime_evaluation(args, model, wandb_run)

    wandb.finish()

    unconstrained_generated_data, _ = generate_evaluation_data(args, model, real_data, columns)

    if  args.performance_evaluation:
        evaluate_model_performance(args, unconstrained_generated_data, roundable_idx, round_digits, columns, real_data, dataset_info)
