"""CLI."""
import argparse
import datetime
import os
import pickle
import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd

import wandb

sys.path.append('.')

from cdgm.synthetizers.TVAE.tvae import TVAE
from run_dgm.run_utils import evaluate_model_performance, runtime_evaluation, add_common_args, generate_evaluation_data
from utils import _load_json, read_csv, set_seed

DATETIME = datetime.datetime.now()


def _parse_args():
    parser = argparse.ArgumentParser(description='C-DGMs Command Line Interface')
    parser.add_argument("--model_type", default="TVAE", type=str)

    add_common_args(parser)

    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimension of input z to the encoder.')
    parser.add_argument('--compress_dims', type=str, default='128,128',
                        help='Dimension of each encoder layer. '
                        'Comma separated integers with no whitespaces.')
    parser.add_argument('--decompress_dims', type=str, default='128,128',
                        help='Dimension of each decoder layer. '
                        'Comma separated integers with no whitespaces.')
    parser.add_argument('--l2scale', type=float, default=1e-5,
                    help='')
    parser.add_argument('--loss_factor', type=float, default=2,
                help='Multiplier for the reconstruction error. Defaults to 2' )
    parser.add_argument('--save', default=None, type=str,
                        help='A filename to save the trained synthesizer.')
    parser.add_argument('--load', default=None, type=str,
                        help='A filename to load a trained synthesizer.')
 
    return parser.parse_args()


def main():
    """CLI."""
    args = _parse_args()
    set_seed(args.seed)
    args.exp_id = f"{args.version}_{args.label_ordering}_{args.seed}_{args.epochs}_{args.batch_size}_{DATETIME:%d-%m-%y--%H-%M-%S}"
    args.path = f"outputs/{args.model_type}/{args.use_case}/{args.version}/{args.exp_id}"
    os.makedirs(args.path)


    ######################################################################
    wandb_run = wandb.init(project=args.wandb_project, id=args.exp_id, reinit=True, mode=args.wandb_mode)
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



    if args.load:
        model = TVAE.load(args.load)
    else:
        compress_dims = [int(x) for x in args.compress_dims.split(',')]
        decompress_dims = [int(x) for x in args.decompress_dims.split(',')]


        model = TVAE(X_test,
            embedding_dim=args.embedding_dim, compress_dims=compress_dims,
            decompress_dims=decompress_dims, l2scale=args.l2scale, loss_factor=args.loss_factor,
            batch_size=args.batch_size, epochs=args.epochs, path=args.path, bin_cols_idx=cat_idx, version=args.version)

    model.set_random_state(args.seed)
    model.fit(args, X_train, cat_cols)

    if args.save is not None:
        model.save(args.save)

    model.set_random_state(args.seed)

    if args.runtime_evaluation_only:
        runtime_evaluation(args, model, wandb_run)

    wandb.finish()

    unconstrained_generated_data, _ = generate_evaluation_data(args, model, real_data, columns)

    if  args.performance_evaluation:
        evaluate_model_performance(args, unconstrained_generated_data, roundable_idx, round_digits, columns, real_data, dataset_info)


        
if __name__ == '__main__':

    main()
