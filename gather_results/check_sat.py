import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from constraints_code.compute_sets_of_constraints import compute_sets_of_constraints
from constraints_code.correct_predictions import correct_preds, check_all_constraints_sat
from constraints_code.feature_orderings import set_ordering
from constraints_code.parser import parse_constraints_file
from evaluation.eval import constraints_sat_check, sdv_eval_synthetic_data, eval_synthetic_data, gen_sat_check
from utils import read_csv, set_seed, _load_json

warnings.filterwarnings(action='ignore')
torch.set_printoptions(sci_mode=False)
np.set_printoptions(precision=3, suppress=True)
from argparse import ArgumentParser
import pickle as pkl

set_seed(0)


def get_args():
    args = ArgumentParser()
    args.add_argument("use_case", type=str)
    args.add_argument("model_type", type=str)
    args.add_argument("version", type=str, choices=['unconstrained', 'random', 'corr', 'kde'])
    # args.add_argument("expdir", type=str)
    args.add_argument("--round_before_cons", action='store_true')
    args.add_argument("--round_after_cons", action='store_true')
    args.add_argument("--use_only_target_original_dtype", action='store_true')
    args.add_argument("--postprocessing", action='store_true')
    args.add_argument("--postprocessing_label_ordering", default='random', choices=['random', 'corr', 'kde'])
    return args.parse_args()


def get_model_paths(args):
    model_paths = _load_json(f"../model_paths/reeval_path_{args.model_type}.json")[args.model_type][args.use_case][args.version]
    return model_paths


def prepare_gen_data(args, data, roundable_idx, round_digits, columns, X_train):
    if type(data) == list:
        data = {"train": data[0], "val": data[1], "test": data[2]}

    ordering, constraints = parse_constraints_file(args.constraints_file)

    model_type = args.model_type
    if '_out' in model_type:
        model_type = model_type.lower()[:-4]

    print('MODEL TYPE FOR SETTING ORDERING', model_type)
    ordering = set_ordering(args.use_case, ordering, args.label_ordering, model_type)
    sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)

    gen_data = {'train':[], 'val':[], 'test':[]}
    unrounded_gen_data = {'train':[], 'val':[], 'test':[]}
    for part in gen_data.keys():
        print("Part", part)
        for j in range(len(data[part])):
            sampled_data = data[part][j]

            if args.round_before_cons:
                # sampled_data = sampled_data.numpy()
                sampled_data[:, roundable_idx] = sampled_data[:, roundable_idx].round()

            # constraint the output:
            if args.version != 'unconstrained' or args.postprocessing:
                sampled_data = torch.tensor(sampled_data)
                sampled_data = correct_preds(sampled_data, ordering, sets_of_constr)
                sat = check_all_constraints_sat(sampled_data, constraints, error_raise=False)
                print(f'Corrected sampled_data for {part}, round {j}', 'sat:', sat)

            sampled_data = pd.DataFrame(sampled_data, columns=columns)
            if args.round_after_cons:
                sampled_data.iloc[:, roundable_idx] = sampled_data.iloc[:, roundable_idx].round(round_digits)

            # sampled_data = sampled_data.astype(float)
            # target_col = columns[-1]
            # sampled_data[target_col] = sampled_data[target_col].astype(X_train.dtypes[-1])

            # store data for constr. sat.
            unrounded_gen_data[part].append(sampled_data)

            # process data for calculating the utility, detection and store it
            if args.use_only_target_original_dtype:
                target_col = columns[-1]
                sampled_data[target_col] = sampled_data[target_col].astype(X_train.dtypes[-1])
            else:
                sampled_data = sampled_data.astype(X_train.dtypes)
            gen_data[part].append(sampled_data)
    return gen_data, unrounded_gen_data


def main():
    args = get_args()
    if args.postprocessing and args.version != 'unconstrained':
        raise Exception('Do postprocessing only for unconstrained models')

    dataset_info = _load_json("./datasets_info.json")[args.use_case]
    args.constraints_file = f'./data/{args.use_case}/{args.use_case}_constraints.txt'

    args.path_names = get_model_paths(args)
    args.num_sampling_rounds = 5
    if args.postprocessing:
        args.wandb_project = f"postprocessing_{args.model_type}_{args.use_case}"
    else:
        args.wandb_project = f"evaluation_{args.model_type}_{args.use_case}"

    args.wandb_project += f"round_only_target-{args.use_only_target_original_dtype}"
    # if args.model_type == 'wgan':
    #     args.model_type = "WGAN_out"
    # elif args.model_type == 'tablegan':
    #     args.model_type = "TableGAN_out"
    # elif args.model_type == 'ctgan':
    #     args.model_type = "CTGAN_out"

    X_train, (cat_cols, cat_idx), (roundable_idx, round_digits) = read_csv(f"data/{args.use_case}/train_data.csv", args.use_case)
    columns = X_train.columns.values.tolist()
    args.train_data_cols = columns
    args.dtypes = X_train.dtypes

    partition = 'tiny' if args.use_case in ['botnet', 'lcld'] else ''
    X_train = pd.read_csv(f"data/{args.use_case}/{partition}/train_data.csv")
    X_test = pd.read_csv(f"data/{args.use_case}/{partition}/test_data.csv")
    X_val = pd.read_csv(f"data/{args.use_case}/{partition}/val_data.csv")
    real_data = {"train": X_train, "val": X_val, "test": X_test}
    args.sampling_sizes = [X_train.shape[0], X_val.shape[0], X_test.shape[0]]

    all_sat_metrics = {'percentage_of_samples_violating_constraints':[], 'percentage_cons_violations_per_pred': [], 'percentage_of_constr_violated_at_least_once':[]}
    for path_name in args.path_names:
        exp_id = Path(path_name).parts[-1]

        if args.version != 'unconstrained':
            args.label_ordering = args.version
        else:
            if args.postprocessing:
                args.label_ordering = args.postprocessing_label_ordering
            else:
                args.label_ordering = 'random'
        args.exp_path = path_name
        args.path_name = path_name
        args.real_data_partition = 'test'

        generated_data = pkl.load(open(f'{args.exp_path}/unconstrained_generated_data.pkl', 'rb'))

        if args.model_type == 'tablegan':
            generated_data = {"train":generated_data[0], "val":generated_data[1], "test":generated_data[2]}

        generated_data, unrounded_generated_data = prepare_gen_data(args, generated_data, roundable_idx, round_digits,  columns, X_train)
        # constraints_sat_check(args, real_data, unrounded_generated_data, log_wandb=False)
        _, constraints = parse_constraints_file(args.constraints_file)
        _,_, sat_metrics = gen_sat_check(args, unrounded_generated_data, constraints, log_wandb=False)

        for e in sat_metrics:
            val = sat_metrics[e][0]
            if type(val) == torch.Tensor:
                val = val.numpy()
            all_sat_metrics[e].append(val)

    for i in reversed(range(len(all_sat_metrics['percentage_of_samples_violating_constraints']))):
        print(' '.join(map(str,[all_sat_metrics[e][i] for e in all_sat_metrics])))


if __name__ == "__main__":

    main()