import warnings

import numpy as np
import pandas as pd
import torch

from constraints_code.parser import parse_constraints_file

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
    args.add_argument("--log_wandb", action='store_true')
    return args.parse_args()


def gather_aggregated(path_name):
    final_results = []
    for path_name in path_names:
        path_name = f"{path_name}/final_INFERENCE_all_avg_scores.csv"
        res = pd.read_csv(path_name)
        res = res[1:]
        final_results.append(res)
    final_results = pd.concat(final_results)
    print(final_results)
    final_results.to_csv(open('del_summary_aggregated.csv', 'w'))


def gather_sdv(path_names):
    final_results = []
    for path_name in path_names:
        path_name = f"{path_name}/final_sdv.csv"
        final_results.append(pd.read_csv(path_name))
    final_results = pd.concat(final_results)
    print(final_results)
    final_results.to_csv(open('del_summary_sdv.csv', 'w'))


def gather_constraints_sat(path_names):
    final_results = []
    for path_name in path_names:
        path_name1 = f"{path_name}/INFERENCE_cons_avg.pkl"
        final_results1 = pkl.load(open(path_name1, 'rb'))
        final_results.append(final_results1)


        # path_name2 = f"{path_name}/INFERENCE_results_cons.pkl"
        # final_results2 = pkl.load(open(path_name2, 'rb'))
        # print(final_results2)
    final_results = pd.concat(final_results)
    print(final_results)
    final_results.to_csv(open('del_summary_cons.csv', 'w'))



def real_sat_check(args, constraints):
    real_data = pd.read_csv(f"data/{args.use_case}/test_data.csv")
    cols = real_data.columns
    print(cols[33:37])
    exit()
    sat_rate_per_constr = {i: [] for i in range(len(constraints))}
    percentage_of_samples_sat_constraints = []

    samples_sat_constr = torch.ones(real_data.shape[0]) == 1.
    # real_data = real_data.iloc[:, :-1].to_numpy()
    real_data = torch.tensor(real_data.to_numpy())
    print(real_data[4813][31:33])
    for j, constr in enumerate(constraints):
        sat_per_datapoint = constr.single_inequality.check_satisfaction(real_data)
        if not sat_per_datapoint.all():
            sample_sat, eval_body_value, constant, ineq_sign = constr.detailed_sample_sat_check(real_data)
            print('REAL', sample_sat.all(), eval_body_value[~sample_sat], constant, ineq_sign, constr.readable(), torch.arange(sample_sat.shape[0])[~sample_sat])
        sat_rate = sat_per_datapoint.sum() / len(sat_per_datapoint)
        # print('Real sat_rate is', sat_rate, sat_per_datapoint.sum(), len(sat_per_datapoint), sat_per_datapoint)
        sat_rate_per_constr[j].append(sat_rate)
        # sat_rate_per_constr[j].append(sat_per_datapoint.sum() / len(sat_per_datapoint))
        samples_sat_constr = samples_sat_constr & sat_per_datapoint

    percentage_of_samples_sat_constraints.append(sum(samples_sat_constr)/len(samples_sat_constr))
    sat_rate_per_constr = {i: [sum(sat_rate_per_constr[i]) / len(sat_rate_per_constr[i]) * 100.0] for i in
                           range(len(constraints))}
    percentage_of_samples_violating_constraints = 100.0 - sum(percentage_of_samples_sat_constraints) / len(
        percentage_of_samples_sat_constraints) * 100.0
    print('sat_rate_per_constr', sat_rate_per_constr)
    print('percentage_of_samples_violating_constraints', percentage_of_samples_violating_constraints)

    sat_rate_per_constr = pd.DataFrame(sat_rate_per_constr, columns=list(range(len(constraints))))
    percentage_of_samples_violating_constraints = pd.DataFrame({'real_percentage_of_samples_violating_constraints': [percentage_of_samples_violating_constraints]}, columns=['real_percentage_of_samples_violating_constraints'])

    return sat_rate_per_constr, percentage_of_samples_violating_constraints



def data_sat_check(args, all_data, constraints):
    sat_rate_per_constr = {i:[] for i in range(len(constraints))}
    percentage_of_samples_sat_constraints = []

    for _, gen_data in enumerate(all_data):
        samples_sat_constr = torch.ones(gen_data.shape[0]) == 1.
        # gen_data = gen_data.iloc[:, :-1].to_numpy()
        gen_data = torch.tensor(gen_data.to_numpy())
        for j, constr in enumerate(constraints):
            sat_per_datapoint = constr.single_inequality.check_satisfaction(gen_data)
            if not sat_per_datapoint.all():
                sample_sat, eval_body_value, constant, ineq_sign = constr.detailed_sample_sat_check(gen_data)
                print(sample_sat.all(), eval_body_value[~sample_sat], constant, ineq_sign)
            sat_rate = sat_per_datapoint.sum()/len(sat_per_datapoint)
            # print('Synth sat_rate is', sat_rate, sat_per_datapoint.sum(), len(sat_per_datapoint), sat_per_datapoint)
            sat_rate_per_constr[j].append(sat_rate)
            samples_sat_constr = samples_sat_constr & sat_per_datapoint
            # print('samples_violating_constr:', samples_violating_constr.sum())

        percentage_of_samples_sat_constraints.append(sum(samples_sat_constr) / len(samples_sat_constr))
    sat_rate_per_constr = {i:[sum(sat_rate_per_constr[i])/len(sat_rate_per_constr[i]) * 100.0] for i in range(len(constraints))}
    percentage_of_samples_violating_constraints = 100.0-sum(percentage_of_samples_sat_constraints)/len(percentage_of_samples_sat_constraints) * 100.0
    print('sat_rate_per_constr', sat_rate_per_constr)
    print('percentage_of_samples_violating_constraints', percentage_of_samples_violating_constraints)

    sat_rate_per_constr = pd.DataFrame(sat_rate_per_constr, columns=list(range(len(constraints))))
    percentage_of_samples_violating_constraints = pd.DataFrame({'percentage_of_samples_violating_constraints': [percentage_of_samples_violating_constraints]}, columns=['percentage_of_samples_violating_constraints'])

    return sat_rate_per_constr, percentage_of_samples_violating_constraints


def eval_preds_constr(args, path_names):
    args.real_data_partition = 'test'
    args.constraints_file = f'./data/{args.use_case}/{args.use_case}_constraints.txt'
    ordering, constraints = parse_constraints_file(args.constraints_file)

    print('REAL data')
    real_sat_check(args, constraints)

    for path_name in path_names:
        path_name = f"{path_name}/generated_data.pkl"
        gen_data = pkl.load(open(path_name, 'rb'))
        gen_data = gen_data[args.real_data_partition]
        print('\nSYNTHETIC data')
        data_sat_check(args, gen_data, constraints)


if __name__ == "__main__":
    args = get_args()
    args.path_names = ["constrained_5_rmsprop_50_128_100_64_13-08-23--20-29-39", "constrained_7_adam_2_128_200_64_12-08-23--14-25-40"]

    if args.model_type == 'wgan':
        args.model_type = "WGAN_out"
    elif args.model_type == 'tablegan':
        args.model_type = "TableGAN_out"
    elif args.model_type == 'ctgan':
        args.model_type = "CTGAN_out"

    path_names = []
    for p in args.path_names:
        version = 'unconstrained' if 'unconstrained' in p else 'constrained'
        path_names.append(f"{args.model_type}/{args.use_case}/{version}/{p}")
    ######################################################################
    # args.wandb_project = f"wandb_{args.model_type}_{args.use_case}_gather_results"
    # exp_id = Path(f"{args.path_name}").parts[-1]
    # wandb_run = wandb.init(project=args.wandb_project, id=exp_id)
    # for k,v in args._get_kwargs():
    #     wandb_run.config[k] = v
    ######################################################################

    # wandb/run-20230813_202940-constrained_5_rmsprop_50_128_100_64_13-08-23--20-29-39/
    gather_sdv(path_names)
    gather_aggregated(path_names)
    gather_constraints_sat(path_names)
    eval_preds_constr(args, path_names)

