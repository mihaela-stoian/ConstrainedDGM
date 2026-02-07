"""CLI."""
import argparse
import datetime
import os
import sys

import pandas as pd
import wandb

sys.path.append('.')

from cdgm.synthetizers.TableGAN.tableGAN import TableGAN
from run_dgm.run_utils import evaluate_model_performance, runtime_evaluation, add_common_args, generate_evaluation_data
from utils import _load_json, read_csv, set_seed

DATETIME = datetime.datetime.now()


def _parse_args():
    parser = argparse.ArgumentParser(description='C-DGMs Command Line Interface')
    parser.add_argument("--model_type", default="TableGAN", type=str)

    add_common_args(parser)
    
    parser.add_argument('--random_dim', type=int, default=100,
                        help='')
    parser.add_argument('--num_channels', type=int, default=64,
                        help='Dimension of each generator layer. '
                        'Comma separated integers with no whitespaces.')
    parser.add_argument('--l2scale', type=float, default=1e-5, help='')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--optimiser', type=str, default="adam", choices=['adam','rmsprop','sgd'], help='')
    parser.add_argument('--save', default=None, type=str,
                        help='A filename to save the trained synthesizer.')
    parser.add_argument('--load', default=None, type=str,
                        help='A filename to load a trained synthesizer.')
    return parser.parse_args()


def main():
    """CLI."""
    args = _parse_args()
    set_seed(args.seed)
    args.exp_id = f"{args.version}_{args.label_ordering}_{args.seed}_{args.optimiser}_{args.epochs}_{args.batch_size}_{args.random_dim}_{args.num_channels}_{DATETIME:%d-%m-%y--%H-%M-%S}"
    args.path = f"outputs/{args.model_type}/{args.use_case}/{args.version}/{args.exp_id}"
    os.makedirs(args.path)
    ######################################################################
    wandb_run = wandb.init(project=args.wandb_project, id=args.exp_id, reinit=True,  mode=args.wandb_mode)
    for k,v in args._get_kwargs():
        wandb_run.config[k] = v
    ######################################################################
    ######################################################################
    args.constraints_file = f'./data/{args.use_case}/{args.use_case}_constraints.txt'
    ######################################################################
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
    args.train_data_cols = columns
    args.dtypes = X_train.dtypes

    if args.load:
        model = TableGAN.load(args.load)
    else:
        model = TableGAN(X_test,
            random_dim=args.random_dim, num_channels=args.num_channels,
            l2scale=args.l2scale, batch_size=args.batch_size,
            epochs=args.epochs, path=args.path, bin_cols_idx=cat_idx, version=args.version)

    model.fit(args, X_train, cat_idx)

    if args.save is not None:
        model.save(args.save)

    if args.sample_condition_column is not None:
        assert args.sample_condition_column_value is not None

    model.set_random_state(args.seed)

    if args.runtime_evaluation:
        runtime_evaluation(args, model, wandb_run)

    wandb.finish()

    if  args.performance_evaluation:
        evaluate_model_performance(args, model, roundable_idx, round_digits, columns, real_data, dataset_info)


if __name__ == '__main__':

    main()
