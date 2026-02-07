# Goggle
import sys
import argparse
import datetime
import os

import pandas as pd
import wandb
import torch

sys.path.append('.')


from cdgm.synthetizers.GOGGLE.GoggleModel import GoggleModel
from cdgm.data_processors.goggle.transformer import CustomTransformer
from utils import _load_json, prepare_data_for_goggle, read_csv, set_seed
from run_dgm.run_utils import evaluate_model_performance, runtime_evaluation, add_common_args, generate_evaluation_data

DATETIME = datetime.datetime.now()


def _parse_args():
    parser = argparse.ArgumentParser(description='C-DGMs Command Line Interface')
    parser.add_argument("--model_type", default="GOGGLE", type=str)

    add_common_args(parser)
    
    parser.add_argument("--fit_train_set", action='store_true', help='if enabled, fits transformer only on train set, otherwise on full set (train+val+test, except last column)')
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--enforce_min_max_values', action='store_true', help='if enabled, enforces min and max values observed in the training data during training time. If not set, no min/max value enforcement is done during training time.')
    parser.add_argument('--inference_time_enforce_min_max_values', action='store_true', help='if enabled, enforces min and max values observed in the training data during inference time. If not set, no min/max value enforcement is done during inference time.')
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
    args.constraints_file = f'data/{args.use_case}/{args.use_case}_constraints.txt'
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
    

 
    # fit transformer on full dataset (concat X_train + X_test + X_val); (reason for including test: target column is not fitted, so it's fine)
    if args.fit_train_set:
        ct_fit_X_set = X_train
    else:
        ct_fit_X_set = pd.concat([X_train, X_val, X_test])
    min_observed = torch.tensor(ct_fit_X_set.min(axis=0).to_numpy(), dtype=torch.float32)
    max_observed = torch.tensor(ct_fit_X_set.max(axis=0).to_numpy(), dtype=torch.float32)

    is_regression_task = dataset_info['problem_type'] == 'regression'
    _, ct = prepare_data_for_goggle(columns, ct_fit_X_set, ct=None, regression=is_regression_task)
    column_transformer = CustomTransformer(means=ct.transformers_[0][1].mean_, stds=ct.transformers_[0][1].scale_,
                                           enforce_min_max_values=args.enforce_min_max_values,
                                           min_value=min_observed, max_value=max_observed, regression=is_regression_task)

    X_train, _ = prepare_data_for_goggle(columns, X_train, ct=ct, regression=is_regression_task)
    X_test, _ = prepare_data_for_goggle(columns, X_test, ct=ct, regression=is_regression_task)
    X_val, _ = prepare_data_for_goggle(columns, X_val, ct=ct, regression=is_regression_task)
    # TODO: do we want to update the columns after this step so that target col is actually named "target"??? if no, then remove the next 3 lines
    columns = X_train.columns.values.tolist()
    args.train_data_cols = columns
    args.dtypes = X_train.dtypes
    args.goggle_data = [X_train, X_val, X_test]

    if cat_cols == None:
        cat_cols = []
        cat_idx = []


    gen = GoggleModel(
        ds_name=args.use_case,
        input_dim=X_train.shape[1],
        column_transformer=column_transformer,
        constrained=args.version=='constrained',
        label_ordering=args.label_ordering,
        constraints_file=args.constraints_file,
        save_path=args.path,
        inference_time_enforce_min_max_values=args.inference_time_enforce_min_max_values,
        het_encoding=True,
        graph_prior=None,
        prior_mask=None,
        device="cpu",
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
        logging=1, # logging every n epochs
    )
    print(gen.model)
    print(f'Start training {args.version} GOGGLE model')
    gen.fit(X_train, X_val)
    print(f'Finished training {args.version} GOGGLE model')


    if args.runtime_evaluation_only:
        runtime_evaluation(args, gen, wandb_run)

    wandb.finish()

    unconstrained_generated_data, _ = generate_evaluation_data(args, gen, real_data, columns)

    if  args.performance_evaluation:
        evaluate_model_performance(args, unconstrained_generated_data, roundable_idx, round_digits, columns, real_data, dataset_info)
    

if __name__ == "__main__":
    main()
