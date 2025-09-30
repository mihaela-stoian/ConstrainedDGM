# Goggle
import pickle
import sys
import argparse
import datetime
import os
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import wandb
import torch

sys.path.append('.')


from cdgm.synthetizers.GOGGLE.GoggleModel import GoggleModel

from cdgm.data_processors.goggle.transformer import CustomTransformer, _load_json, prepare_data_for_goggle, read_csv, set_seed
from evaluation.eval import constraints_sat_check, eval_synthetic_data, sdv_eval_synthetic_data


DATETIME = datetime.datetime.now()


def _parse_args():
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--use_only_target_original_dtype", action='store_true')
    parser.add_argument("--enforce_min_max_values", action='store_true')
    parser.add_argument("--inference_time_enforce_min_max_values", action='store_true')
    parser.add_argument("--fit_train_set", action='store_true', help='if enabled, fits transformer only on train set, otherwise on full set (train+val+test, except last column)')
    parser.add_argument("--wandb_project", default="goggle", type=str)
    parser.add_argument("--wandb_mode", default="online", type=str, choices=['online', 'disabled', 'offline'])
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        help='Number of training epochs')
    parser.add_argument('-n', '--num-samples', type=int,
                        help='Number of rows to sample. Defaults to the training data size')
    parser.add_argument("--label_ordering", default='random', choices=['random', 'corr', 'kde', 'wasserstein', 'jsd', 'causal'])
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default for goggle is 32). Must be an even number.')
    parser.add_argument('--save', default=None, type=str,
                        help='A filename to save the trained synthesizer.')
    parser.add_argument('--load', default=None, type=str,
                        help='A filename to load a trained synthesizer.')
    parser.add_argument("use_case", type=str, choices=["url","wids","botnet","lcld","heloc","news","faults"])
    parser.add_argument("--version", type=str, default='unconstrained', choices=['unconstrained','constrained', "postprocessing"],
                        help='Version of training. Correct values are unconstrained, constrained and postprocessing')
    parser.add_argument('--skip_evaluation', action='store_true')
    parser.add_argument('--runtime_evaluation_only', action='store_true')
    return parser.parse_args()


def main():
    """CLI."""
    args = _parse_args()
    set_seed(args.seed)
    exp_id = f"{args.version}_{args.label_ordering}_{args.seed}_{args.epochs}_{args.batch_size}_{DATETIME:%d-%m-%y--%H-%M-%S}"
    path = f"outputs/GOGGLE_out/{args.use_case}/{args.version}/{exp_id}"
    args.exp_path = path
    os.makedirs(path)


    ######################################################################
    wandb_run = wandb.init(project=args.wandb_project, id=exp_id, reinit=True, mode=args.wandb_mode)
    for k,v in args._get_kwargs():
        wandb_run.config[k] = v
    ######################################################################
    args.constraints_file = f'data/{args.use_case}/{args.use_case}_constraints.txt'
    ######################################################################
    dataset_info = _load_json("data/datasets_info.json")[args.use_case]
    print(dataset_info)
    ######################################################################

    X_train, (cat_cols, cat_idx), (roundable_idx, round_digits) = read_csv(f"data/{args.use_case}/train_data.csv",
                                                                           args.use_case, dataset_info[
                                                                               "manual_inspection_categorical_cols_idx"])
    args.original_dtypes = X_train.dtypes

    if args.use_case == "botnet" or args.use_case == "lcld":
        X_test = pd.read_csv(f"data/{args.use_case}/tiny/test_data.csv")
        X_val = pd.read_csv(f"data/{args.use_case}/tiny/val_data.csv")

    else:
        X_test = pd.read_csv(f"data/{args.use_case}/test_data.csv")
    X_val = pd.read_csv(f"data/{args.use_case}/val_data.csv")

    columns = X_train.columns.values.tolist()
    args.train_data_cols = columns
    args.dtypes = X_train.dtypes

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
        save_path=args.exp_path,
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

    print(f'Output a generated sample from the just trained {args.version} GOGGLE model')
    X_synth, unconstr_X_synth = gen.sample(X_test, inverse_transformed=True)
    print(X_synth.head())
    print(X_synth.iloc[0])
    num_sampling_rounds = 5

    ########### Runtime
    if args.runtime_evaluation_only:
        size = 1000
        runs = []
        for i in range(num_sampling_rounds):
            start = timer()
            sampled_data = gen.sample_runtime(X_train, inverse_transformed=True)
            end = timer()
            runtime = end - start
            runs.append(runtime)
        runtime_df = pd.DataFrame(list(zip([np.mean(runtime)],[np.std(runtime)])), columns=["Mean", "Std"])
        wandb.log({'Runtime/Sampling': runtime_df})
    else:
        #######################################################################################
        # Generate data using the trained model
        #######################################################################################
        gen_data = [[], [], []]
        unconstrained_gen_data = [[], [], []]
        unrounded_generated_data = [[], [], []]
        X_partitions = [X_train, X_val, X_test]
        for r in range(num_sampling_rounds):
            for i in range(3):
                X_partition = X_partitions[i]
                output, unconstrained_output = gen.sample(X_partition, inverse_transformed=True)

                target_col = dataset_info['target_col']
                output = output.rename(columns={'target': target_col})
                unconstrained_output = unconstrained_output.rename(columns={'target': target_col})


                unconstrained_output[target_col] = unconstrained_output[target_col].astype(args.original_dtypes[-1])
                unconstrained_gen_data[i].append(unconstrained_output)

                output[target_col] = output[target_col].astype(args.original_dtypes[-1])
                unrounded_generated_data[i].append(output)

                output = output.astype(args.original_dtypes)
                gen_data[i].append(output)

        generated_data = {"train": gen_data[0], "val": gen_data[1], "test": gen_data[2]}
        unconstrained_generated_data = {"train": unconstrained_gen_data[0], "val": unconstrained_gen_data[1], "test": unconstrained_gen_data[2]}
        unrounded_gen_data = {'train': unrounded_generated_data[0], 'val': unrounded_generated_data[1], 'test': unrounded_generated_data[2]}

        with open(f'{path}/generated_data.pkl', 'wb') as f:
            pickle.dump(generated_data, f)
        with open(f'{path}/unconstrained_generated_data.pkl', 'wb') as f:
            pickle.dump(unconstrained_generated_data, f)
        with open(f'{path}/unrounded_gen_data.pkl', 'wb') as f:
            pickle.dump(unrounded_gen_data, f)
        print(f'Saved generated (mapped to the original space) outputs at {path}')

        #######################################################################################
        # Load real data, and keep it in its original space
        #######################################################################################

        if args.use_case == "botnet" or args.use_case == "lcld":
            tiny_partitions = 'tiny/'
        else:
            tiny_partitions = ''
        X_train = pd.read_csv(f"data/{args.use_case}/{tiny_partitions}train_data.csv")
        X_test = pd.read_csv(f"data/{args.use_case}/{tiny_partitions}test_data.csv")
        X_val = pd.read_csv(f"data/{args.use_case}/{tiny_partitions}val_data.csv")

        real_data = {"train": X_train, "val": X_val, "test": X_test}

        #######################################################################################
        # Evaluate the generated data
        #######################################################################################

        wandb.finish()
        args.real_data_partition = 'test'
        args.model_type = 'goggle'

        wandb_project_name = f"evaluation_{args.model_type}_{args.use_case}_fixed_eval"
        if "hyperparam" in args.wandb_project:
            wandb_project_name += "_hyperparam_search"
        args.wandb_project = wandb_project_name

        wandb_run = wandb.init(project=args.wandb_project, id=exp_id)
        for k, v in args._get_kwargs():
            wandb_run.config[k] = v
        ######################################################################
        args.round_before_cons = False
        args.round_after_cons = False
        args.postprocessing = False
        args.train_data_cols = X_train.columns
        args.dtypes = X_train.dtypes
        if args.version != 'unconstrained':
            args.version = args.label_ordering

        # generated_data, unrounded_generated_data = prepare_gen_data(args, unconstrained_generated_data, roundable_idx, round_digits, columns, X_train)
        for i in range (5):
            print(generated_data["train"][i].shape, generated_data["val"][i].shape, generated_data["test"][i].shape)
            print(unconstrained_generated_data["train"][i].shape, unconstrained_generated_data["val"][i].shape, unconstrained_generated_data["test"][i].shape)

        constraints_sat_check(args, real_data, unrounded_gen_data, log_wandb=True)
        print('Using evaluators with the following specs', dataset_info["problem_type"], dataset_info["target_size"], dataset_info["target_col"])
        eval_synthetic_data(args, args.use_case, real_data, generated_data, X_train.columns,
                            problem_type=dataset_info["problem_type"], target_utility=dataset_info["target_col"],
                            target_utility_size=dataset_info["target_size"], target_detection="", log_wandb=True,
                            wandb_run=wandb_run, unrounded_generated_data_for_cons_sat=unrounded_gen_data)


if __name__ == "__main__":
    main()
