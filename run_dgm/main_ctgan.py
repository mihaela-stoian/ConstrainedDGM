"""CLI."""
import argparse
import datetime
import os
import sys

import pandas as pd
import wandb

sys.path.append('.')

from cdgm.synthetizers.CTGAN.ctgan import CTGAN
from run_dgm.run_utils import evaluate_model_performance, runtime_evaluation, add_common_args, generate_evaluation_data
from utils import _load_json, all_div_gt_n, read_csv, set_seed

DATETIME = datetime.datetime.now()


def _parse_args():
    parser = argparse.ArgumentParser(description='C-DGMs Command Line Interface')
    parser.add_argument("--model_type", default="CTGAN", type=str)

    add_common_args(parser)

    parser.add_argument("--pac", default=10, type=int)
    parser.add_argument('--generator_lr', type=float, default=2e-4,
                        help='Learning rate for the generator.')
    parser.add_argument('--discriminator_lr', type=float, default=2e-4,
                        help='Learning rate for the discriminator.')
    parser.add_argument('--generator_decay', type=float, default=1e-6,
                        help='Weight decay for the generator.')
    parser.add_argument('--discriminator_decay', type=float, default=0,
                        help='Weight decay for the discriminator.')
    parser.add_argument('--optimiser', type=str, default="adam", choices=['adam','rmsprop','sgd'], help='')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimension of input z to the generator.')
    parser.add_argument('--generator_dim', type=str, default='256,256',
                        help='Dimension of each generator layer. '
                        'Comma separated integers with no whitespaces.')
    parser.add_argument('--discriminator_dim', type=str, default='256,256',
                        help='Dimension of each discriminator layer. '
                        'Comma separated integers with no whitespaces.')
    parser.add_argument('--save', default=None, type=str,
                        help='A filename to save the trained synthesizer.')
    parser.add_argument('--load', default=None, type=str,
                        help='A filename to load a trained synthesizer.')
    parser.add_argument('--sample_condition_column', default=None, type=str,
                        help='Select a discrete column name.')
    parser.add_argument('--sample_condition_column_value', default=None, type=str,
                        help='Specify the value of the selected discrete column.')
    return parser.parse_args()


def main():
    """CLI."""
    args = _parse_args()
    set_seed(args.seed)
    args.exp_id  = f"{args.version}_{args.label_ordering}_{args.seed}_{args.epochs}_{args.batch_size}_{args.discriminator_lr}_{args.generator_lr}_{DATETIME:%d-%m-%y--%H-%M-%S}"
    args.path = f"outputs/{args.model_type}/{args.use_case}/{args.version}/{args.exp_id}"
    os.makedirs(args.path)


    # set args.pac:
    if args.pac != 1:
        if args.batch_size % args.pac != 0:
            original_pac = args.pac
            args.pac = all_div_gt_n(original_pac, args.batch_size)
            print(f'Changed pac {original_pac} to {args.pac}')

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


    if args.load:
        model = CTGAN.load(args.load)
    else:
        generator_dim = [int(x) for x in args.generator_dim.split(',')]
        discriminator_dim = [int(x) for x in args.discriminator_dim.split(',')]
        model = CTGAN(X_test,
            embedding_dim=args.embedding_dim, generator_dim=generator_dim,
            discriminator_dim=discriminator_dim, generator_lr=args.generator_lr,
            generator_decay=args.generator_decay, discriminator_lr=args.discriminator_lr,
            discriminator_decay=args.discriminator_decay, batch_size=args.batch_size,
            epochs=args.epochs, path=args.path, bin_cols_idx=cat_idx, version=args.version, pac=args.pac,
                      feats_in_constraints=dataset_info["feats_in_constraints"])

    model.set_random_state(args.seed)
    model.fit(args, X_train, cat_cols)


    if args.save is not None:
        model.save(args.save)

    if args.sample_condition_column is not None:
        assert args.sample_condition_column_value is not None

    model.set_random_state(args.seed)

    if args.runtime_evaluation_only:
        runtime_evaluation(args, model, wandb_run)

    wandb.finish()

    unconstrained_generated_data, _ = generate_evaluation_data(args, model, real_data, columns)

    if  args.performance_evaluation:
        evaluate_model_performance(args, unconstrained_generated_data, roundable_idx, round_digits, columns, real_data, dataset_info)

if __name__ == '__main__':
    main()
