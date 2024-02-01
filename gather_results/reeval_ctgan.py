import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from constraints_code.compute_sets_of_constraints import compute_sets_of_constraints
from constraints_code.correct_predictions import correct_preds, check_all_constraints_sat

from constraints_code.parser import parse_constraints_file
from data_processors.ctgan.data_transformer import DataTransformer
from evaluation.eval import constraints_sat_check, sdv_eval_synthetic_data, eval_synthetic_data
from utils import read_csv, set_seed

warnings.filterwarnings(action='ignore')
torch.set_printoptions(sci_mode=False)
np.set_printoptions(precision=3, suppress=True)
from argparse import ArgumentParser
# from WGAN_uncons_pac import *
import pickle as pkl


def get_args():
    args = ArgumentParser()
    args.add_argument("use_case", type=str)
    args.add_argument("model_type", type=str)
    # args.add_argument("expdir", type=str)
    args.add_argument("--sample", action='store_true')
    args.add_argument("--round_before_cons", action='store_true')
    args.add_argument("--round_after_cons", action='store_true')
    args.add_argument("--log_wandb", action='store_true')
    return args.parse_args()



def prepare_gen_data(args, data, roundable_idx, round_digits):

    ordering, constraints = parse_constraints_file(args.constraints_file)
    gen_data = {'train':[], 'val':[], 'test':[]}
    for part in gen_data:
        for j in range(len(data[part])):
            sampled_data = data[part][j].detach()

            if args.round_before_cons:
                sampled_data = sampled_data.numpy()
                sampled_data[:, roundable_idx] = sampled_data[:, roundable_idx].round()
                sampled_data = torch.tensor(sampled_data)

            # constraint the output:
            sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)
            sampled_data = correct_preds(sampled_data, ordering, sets_of_constr)
            # sat = check_all_constraints_sat(sampled_data, constraints)

            sampled_data = pd.DataFrame(sampled_data, columns=columns)
            if args.round_after_cons:
                sampled_data.iloc[:, roundable_idx] = sampled_data.iloc[:, roundable_idx].round(round_digits)
            sampled_data = sampled_data.astype(float)
            target_col = columns[-1]
            sampled_data[target_col] = sampled_data[target_col].astype(X_train.dtypes[-1])
            gen_data[part].append(sampled_data)
    return gen_data


def sample_unconstrained_data(args, num_rows, X_train, cat_cols, batch_size):
    model = pkl.load(open(f'{args.exp_path}/model.pt', 'rb'))

    columns = X_train.columns.values.tolist()

    transformer = DataTransformer()
    transformer.fit(X_train, cat_cols)

    steps = num_rows // batch_size + 1
    data = []
    for i in range(steps):
        model.eval()
        fakez = self.generate_noise(global_condition_vec)
        fake = model(fakez)
        fakeact = self._apply_activate(fake)
        data.append(fakeact)
    data = torch.concat(data, axis=0)
    data = data[:n]
    inverse = transformer.inverse_transform(data)
    return inverse.detach().numpy()



if __name__ == "__main__":
    args = get_args()
    args.path_names = ["constrained_9_50_500_0.0002_0.0002_16-08-23--07-49-35"]

    if args.model_type == 'wgan':
        args.model_type = "WGAN_out"
    elif args.model_type == 'tablegan':
        args.model_type = "TableGAN_out"
    elif args.model_type == 'ctgan':
        args.model_type = "CTGAN_out"

    X_train, (cat_cols, cat_idx), (roundable_idx, round_digits) = read_csv(f"data/{args.use_case}/train_data.csv", args.use_case)
    columns = X_train.columns.values.tolist()
    args.train_data_cols = columns
    args.dtypes = X_train.dtypes

    path_names = []
    for p in args.path_names:
        version = 'unconstrained' if 'unconstrained' in p else 'constrained'
        path_name = f"{args.model_type}/{args.use_case}/{version}/{p}"
        path_names.append(path_name)
        args.exp_path = path_name
        args.path_name = path_name
        args.real_data_partition = 'test'
        args.constraints_file = f'./data/{args.use_case}/{args.use_case}_constraints.txt'

        parameters = p.split('_')
        if args.model_type == 'CTGAN_out':
            batch_size = int(parameters[3])
            seed = int(parameters[1])
        set_seed(seed)

        if args.sample:
            generated_data = sample_unconstrained_data(args, num_rows, X_train, cat_cols, batch_size)  # TODO: do we want also numerical discrete here? or just cat_cols?
        else:
            generated_data = pkl.load(open(f'{args.exp_path}/unconstrained_generated_data.pkl', 'rb'))


        generated_data = prepare_gen_data(args, generated_data, roundable_idx, round_digits)

        ######################################################################
        args.wandb_project = f"wandb_{args.model_type}_{args.use_case}_reeval_unrounded"
        exp_id = Path(f"{args.path_name}").parts[-1]
        wandb_run = wandb.init(project=args.wandb_project, id=p)
        for k,v in args._get_kwargs():
            wandb_run.config[k] = v
        ######################################################################

        # wandb/run-20230813_202940-constrained_5_rmsprop_50_128_100_64_13-08-23--20-29-39/
        partition = 'tiny' if args.use_case in ['botnet', 'lcld'] else ''
        X_train = pd.read_csv(f"data/{args.use_case}/{partition}/train_data.csv")
        X_test = pd.read_csv(f"data/{args.use_case}/{partition}/test_data.csv")
        X_val = pd.read_csv(f"data/{args.use_case}/{partition}/val_data.csv")
        real_data = {"train":X_train, "val":X_val, "test":X_test}

        if args.use_case == "url":
            target_utility = "is_phishing"
        elif args.use_case == "lcld":
            target_utility = "charged_off"
        elif args.use_case == "wids":
            target_utility = "hospital_death"
        elif args.use_case == "botnet":
            target_utility = "is_botnet"


        constraints_sat_check(args, real_data, generated_data, log_wandb=True)
        sdv_eval_synthetic_data(args, args.use_case, real_data, generated_data, columns, problem_type='binary_classification', target_utility=target_utility, target_detection="", log_wandb=True, wandb_run=wandb_run)
        eval_synthetic_data(args, args.use_case, real_data, generated_data, columns, problem_type='binary_classification', target_utility=target_utility, target_detection="", log_wandb=True, wandb_run=wandb_run)

