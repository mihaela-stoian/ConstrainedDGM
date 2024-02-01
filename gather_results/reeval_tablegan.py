import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from constraints_code.compute_sets_of_constraints import compute_sets_of_constraints
from constraints_code.correct_predictions import correct_preds, check_all_constraints_sat
from constraints_code.feature_orderings import set_ordering
from rerun_evaluation_TableGAN import sample as sample_tablegan
from constraints_code.parser import parse_constraints_file
from data_processors.ctgan.data_transformer import DataTransformer
from evaluation.eval import constraints_sat_check, sdv_eval_synthetic_data, eval_synthetic_data
from utils import read_csv, set_seed, get_target_col

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
    args.add_argument("--postprocessing", action='store_true')
    args.add_argument("--log_wandb", action='store_true')
    args.add_argument("--postprocessing_label_ordering", default='random', choices=['random', 'corr', 'kde'])
    return args.parse_args()



def prepare_gen_data(args, data, roundable_idx, round_digits, columns, X_train):

    ordering, constraints = parse_constraints_file(args.constraints_file)

    model_type = args.model_type.lower()[:-4]
    print('MODEL TYPE FOR SETTING ORDERING', model_type)
    ordering = set_ordering(args.use_case, ordering, args.label_ordering, model_type)
    sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)

    gen_data = {'train':[], 'val':[], 'test':[]}
    for part in gen_data:
        for j in range(len(data[part])):
            sampled_data = data[part][j]

            if args.round_before_cons:
                # sampled_data = sampled_data.numpy()
                sampled_data[:, roundable_idx] = sampled_data[:, roundable_idx].round()
            sampled_data = torch.tensor(sampled_data)

            # constraint the output:
            if args.version == 'constrained' or args.postprocessing:
                sampled_data = correct_preds(sampled_data, ordering, sets_of_constr)
                # sat = check_all_constraints_sat(sampled_data, constraints)
                print(f'Corrected sampled_data for {part}, round {j}')

            sampled_data = pd.DataFrame(sampled_data, columns=columns)
            if args.round_after_cons:
                sampled_data.iloc[:, roundable_idx] = sampled_data.iloc[:, roundable_idx].round(round_digits)
            # sampled_data = sampled_data.astype(X_train.dtypes)

            sampled_data = sampled_data.astype(float)
            target_col = columns[-1]
            sampled_data[target_col] = sampled_data[target_col].astype(X_train.dtypes[-1])
            gen_data[part].append(sampled_data)
    return gen_data


def sample_unconstrained_data(args, X_train, batch_size, random_dim, cat_idx, roundable_idx, round_digits, columns):
    model = torch.load(open(f'{args.exp_path}/model.pt', 'rb'))

    gen_data = [[], [], []]
    for r in range(args.num_sampling_rounds):
        for i in range (len(args.sampling_sizes)):

            # TODO: Change this function call for WGAN
            sampled_data = sample_tablegan(model, args.sampling_sizes[i], X_train, batch_size, random_dim, cat_idx)
            #sampled_data = model.sample(sizes[i])
            # sampled_data = pd.DataFrame(sampled_data, columns=columns)
            # sampled_data.iloc[:, roundable_idx] = sampled_data.iloc[:, roundable_idx].round(round_digits)
            # sampled_data = sampled_data.astype(X_train.dtypes)
            gen_data[i].append(sampled_data)

    generated_data = {"train":gen_data[0], "val":gen_data[1], "test":gen_data[2]}
    return generated_data




def main():
    args = get_args()
    args.path_names = all_paths[args.use_case]
    args.num_sampling_rounds = 5
    if args.postprocessing:
        args.wandb_project = f"wandb_{args.model_type}_{args.use_case}_reeval_no-rounding_corresp-order_postprocessing"
    else:
        args.wandb_project = f"wandb_{args.model_type}_{args.use_case}_no-rounding_corresp-order"

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

    partition = 'tiny' if args.use_case in ['botnet', 'lcld'] else ''
    X_train = pd.read_csv(f"data/{args.use_case}/{partition}/train_data.csv")
    X_test = pd.read_csv(f"data/{args.use_case}/{partition}/test_data.csv")
    X_val = pd.read_csv(f"data/{args.use_case}/{partition}/val_data.csv")
    real_data = {"train": X_train, "val": X_val, "test": X_test}
    args.sampling_sizes = [X_train.shape[0], X_val.shape[0], X_test.shape[0]]


    path_names = []
    for cons_version in args.path_names:
        if args.postprocessing and cons_version != 'unconstrained':
            continue
        for p in args.path_names[cons_version]:
            version = 'unconstrained' if 'unconstrained' in p else 'constrained'
            path_name = f"{args.model_type}/{args.use_case}/{version}/{p}"

            args.version = version
            if not Path(f'{path_name}/model.pt').exists():
                continue
            args.cons_version = cons_version
            if cons_version != 'unconstrained':
                args.label_ordering = cons_version
            else:
                if args.postprocessing:
                    args.label_ordering = args.postprocessing_label_ordering
                else:
                    args.label_ordering = 'random'
            path_names.append(path_name)
            args.exp_path = path_name
            args.path_name = path_name
            args.real_data_partition = 'test'
            args.constraints_file = f'./data/{args.use_case}/{args.use_case}_constraints.txt'

            ######################################################################
            exp_id = Path(f"{path_name}").parts[-1]
            wandb_run = wandb.init(project=args.wandb_project, id=p+(f'_{args.postprocessing_label_ordering}' if args.postprocessing else ""))
            for k,v in args._get_kwargs():
                wandb_run.config[k] = v
            ######################################################################

            parameters = p.split('_')
            if args.model_type == 'CTGAN_out':
                raise Exception('model_type should be tablegan!')
            elif args.model_type == 'TableGAN_out':
                batch_size = int(parameters[4])
                random_dim = int(parameters[5])
                seed = int(parameters[1])

            set_seed(seed)
            print(batch_size, random_dim, p)

            if args.sample:
                generated_data = sample_unconstrained_data(args, X_train, batch_size, random_dim, cat_idx, roundable_idx, round_digits, columns)  # TODO: do we want also numerical discrete here? or just cat_cols?
            else:
                generated_data = pkl.load(open(f'{args.exp_path}/unconstrained_generated_data.pkl', 'rb'))


            generated_data = prepare_gen_data(args, generated_data, roundable_idx, round_digits,  columns, X_train)



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
            wandb.finish()

if __name__ == "__main__":
    # url:
    url_tablegan_paths = {
                 'unconstrained': "unconstrained_9_adam_300_128_100_64_12-08-23--13-23-14 unconstrained_7_adam_300_128_100_64_12-08-23--12-42-05 unconstrained_5_adam_300_128_100_64_12-08-23--12-03-43 unconstrained_2_adam_300_128_100_64_12-08-23--11-24-53 unconstrained_21_adam_300_128_100_64_14-08-23--17-29-45".split(" "),
                 'kde': "constrained_21_adam_300_128_100_64_13-08-23--23-41-05 constrained_9_adam_300_128_100_64_13-08-23--21-05-26 constrained_7_adam_300_128_100_64_13-08-23--18-26-35 constrained_5_adam_300_128_100_64_13-08-23--15-48-32 constrained_2_adam_300_128_100_64_13-08-23--13-14-56".split(" "),
                 'corr': "constrained_21_adam_300_128_100_64_13-08-23--22-50-43 constrained_9_adam_300_128_100_64_13-08-23--20-12-26 constrained_7_adam_300_128_100_64_13-08-23--17-36-13 constrained_5_adam_300_128_100_64_13-08-23--14-56-19 constrained_2_adam_300_128_100_64_13-08-23--12-22-10".split(" "),
                 'random': "constrained_21_adam_300_128_100_64_14-08-23--00-33-03 constrained_9_adam_300_128_100_64_13-08-23--21-59-23 constrained_7_adam_300_128_100_64_13-08-23--19-18-18 constrained_5_adam_300_128_100_64_13-08-23--16-41-05 constrained_2_adam_300_128_100_64_13-08-23--14-07-05".split(" ")
                 }

    # wids
    wids_tablegan_paths = {
        'unconstrained': "unconstrained_21_rmsprop_50_128_100_64_12-08-23--06-26-19 unconstrained_9_rmsprop_50_128_100_64_12-08-23--04-43-39 unconstrained_7_rmsprop_50_128_100_64_12-08-23--02-51-47 unconstrained_5_rmsprop_50_128_100_64_12-08-23--01-19-21 unconstrained_2_rmsprop_50_128_100_64_11-08-23--23-30-58".split(" "),
                            'kde': "constrained_21_rmsprop_50_128_100_64_15-08-23--08-45-56 constrained_9_rmsprop_50_128_100_64_15-08-23--05-13-06 constrained_7_rmsprop_50_128_100_64_15-08-23--01-29-42 constrained_5_rmsprop_50_128_100_64_14-08-23--21-51-19 constrained_2_rmsprop_50_128_100_64_14-08-23--18-09-32".split(" "),
                            'corr': "constrained_21_rmsprop_50_128_100_64_15-08-23--07-04-46 constrained_9_rmsprop_50_128_100_64_15-08-23--03-18-02 constrained_7_rmsprop_50_128_100_64_14-08-23--23-38-52 constrained_5_rmsprop_50_128_100_64_14-08-23--19-58-06 constrained_2_rmsprop_50_128_100_64_14-08-23--16-14-52".split(" "),
                            'random': "constrained_21_rmsprop_50_128_100_64_15-08-23--23-19-44 constrained_9_rmsprop_50_128_100_64_15-08-23--17-52-27 constrained_7_rmsprop_50_128_100_64_15-08-23--16-10-58 constrained_5_rmsprop_50_128_100_64_15-08-23--14-23-45 constrained_2_rmsprop_50_128_100_64_15-08-23--12-45-35".split(" ")}

    lcld_tablegan_paths = {'unconstrained': "unconstrained_21_adam_20_128_100_64_18-08-23--11-44-14 unconstrained_9_adam_20_128_100_64_18-08-23--08-08-10 unconstrained_7_adam_20_128_100_64_18-08-23--04-16-44 unconstrained_5_adam_20_128_100_64_18-08-23--00-38-47 unconstrained_2_adam_20_128_100_64_17-08-23--21-02-15".split(" "),
                           # 'kde': "",
                           # 'corr': "",
                           # 'random': ""
                           }
    all_paths = {'url': url_tablegan_paths, 'wids': wids_tablegan_paths, 'lcld': lcld_tablegan_paths}
    main()