import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from argparse import ArgumentParser
# from WGAN_uncons_pac import *
import pickle as pkl

import pandas as pd

datasets = ['FAULTS', 'HELOC', 'LCLD', 'NEWS',  'URL', 'WIDS']
versions = ['uncons.', 'random', 'corr', 'kde']
mapped_version = {'uncons.':'-', 'random':'Rnd', 'kde':'KDE', 'corr':'Corr'}
metrics = ['BinaryF1', 'WeightedF1', 'AUC']
evaluations = ['detection', 'utility']
dgm_models = ['WGAN', 'TableGAN', 'CTGAN', 'TVAE']


def get_args():
    args = ArgumentParser()
    args.add_argument("--include_stddev", action='store_true')
    args.add_argument("--postprocessing", action='store_true')
    # args.add_argument("version", type=str, choices=['uncons.', 'random', 'kde', 'corr'])
    # args.add_argument("expdir", type=str)
    return args.parse_args()


def get_results(model_type, metric):
    bypass = ''
    if model_type == 'wgan':
        model_type = 'WGAN'
    elif model_type == 'tablegan':
        model_type = 'TableGAN'
    elif model_type == 'tvae':
        model_type = 'TVAE'
    elif model_type == 'ctgan':
        model_type = 'CTGAN'
        bypass = 'bypass_'

    if metric == 'binary_f1':
        metric = 'BinaryF1'
    elif metric == 'weighted_f1':
        metric = 'WeightedF1'
    elif metric == 'auc':
        metric = 'AUC'
    elif metric == 'avg':
        metric = 'Average'

    filename = f'./results/{bypass}{model_type}_eval - {metric}_results_{model_type}.csv'
    results = pd.read_csv(filename)

    filename_stddevs = f'./results/{bypass}{model_type}_eval - {metric}_stddev_{model_type}.csv'
    if Path(filename_stddevs).exists():
        stddevs = pd.read_csv(filename_stddevs)
    else:
        stddevs = None
        print(filename_stddevs, 'does not exist!')
    return results, stddevs


def create_results_dict(results):
    rdict = {dataset: {'utility': {v:-1 for v in versions}, 'detection': {v:-1 for v in versions}} for dataset in datasets}
    col_offset = 2
    num_cols_per_dataset = 3
    # csv col order: url, wids, lcld, heloc, faults, news
    # row 2: uncons
    # row 3: random
    # row 4: kde
    # row 5: corr

    # col 2: quality
    # col 3: utility
    # col 4: detection
    dataset_col_multiplier_maps = {'FAULTS':4, 'HELOC':3, 'LCLD':2, 'NEWS':5,  'URL':0, 'WIDS':1}
    if not args.postprocessing:
        version_row_map = {'uncons.': 2, 'random': 3, 'kde': 4, 'corr': 5}
    else:
        version_row_map = {'uncons.':2, 'random':6, 'kde':7, 'corr':8}
    eval_col_offset_map = {'utility': 1, 'detection': 2}

    for dataset in datasets:
        for eval in eval_col_offset_map:
            col_loc = dataset_col_multiplier_maps[dataset] * num_cols_per_dataset + eval_col_offset_map[eval] + col_offset
            for v in versions:
                row_loc = version_row_map[v]
                value = results.iloc[row_loc,col_loc]
                if type(value) == str:
                    if '%' in value:
                        value = value[:-1]
                    if '#DIV/0!' in value:
                        value = 'NaN'
                rdict[dataset][eval][v] = float(value) /100.
    return rdict


def get_avg_cols(evaluation, rdict, stddevs, include_stddev=False):
    col_results = {}
    for v in versions:
        avg_res = 0
        num_res = 0
        for dataset in datasets:
            results = rdict[dataset][evaluation]
            baseline_res = results['uncons.']
            res = results[v]

            if v != 'uncons.':
                if evaluation == 'utility':
                    res = baseline_res + res
                elif evaluation == 'detection':
                    res = baseline_res - res
            if not np.isnan(res):
                avg_res += res
                num_res += 1
        col_results[v] = avg_res / num_res
    return col_results


if __name__ == "__main__":
    args = get_args()

    model_types = ['wgan', 'tablegan', 'ctgan', 'tvae']
    results = {evaluation: {model_type: {metric:{} for metric in metrics} for model_type in model_types} for evaluation in evaluations}
    for evaluation in evaluations:
        for model_type in model_types:
            for metric in metrics:
                model_type = model_type
                temp, stddevs = get_results(model_type, metric)

                rdict = create_results_dict(temp)
                if stddevs is not None:
                    stddevs = create_results_dict(stddevs)

                results[evaluation][model_type][metric]['raw'] = rdict
                results[evaluation][model_type][metric]['raw_stddev'] = stddevs
                results[evaluation][model_type][metric]['avg'] = {v:-1 for v in versions}

                # print(rdict)
                col = get_avg_cols(evaluation, rdict, stddevs, include_stddev=args.include_stddev)
                results[evaluation][model_type][metric]['avg_per_version'] = col

    for evaluation in evaluations:
        print(f'\n\nTABLE FOR {evaluation}, postprocessing is {args.postprocessing}')
        all_rows = ''
        heading = ' '
        record_heading = True
        for v in versions:
            if v == 'uncons.':
                if evaluation == 'detection':
                    all_rows += "{\multirow{4}{*}{Detection}&  "
                elif evaluation == 'utility':
                    all_rows += "{\multirow{4}{*}{Utility}&  "
            else:
                all_rows += " & "
            all_rows += f'{mapped_version[v]}'
            # for evaluation in evaluations:
            if record_heading:
                heading += f' & {evaluation}'
                print(f'    {evaluation}')
            for model_type in model_types:
                if record_heading:
                    # heading += f' & {model_type}'
                    print(f'        {model_type}')
                for metric in metrics:
                    if record_heading:
                        heading += f' & {metric}'
                        print(f'            {metric}')

                    # make bold if needed:
                    all_over_vs = np.array([results[evaluation][model_type][metric]['avg_per_version'][vi] for vi in versions])
                    if evaluation == 'utility':
                        best_over_vs = all_over_vs.max()
                    elif evaluation == 'detection':
                        best_over_vs = all_over_vs.min()
                    make_bold = results[evaluation][model_type][metric]['avg_per_version'][v] == best_over_vs

                    all_over_vs = np.array([results[evaluation][model_type][metric]['avg_per_version'][vi] for vi in versions if results[evaluation][model_type][metric]['avg_per_version'][vi]!=best_over_vs])
                    # if evaluation == 'utility':
                    #     second_best_over_vs = all_over_vs.max()
                    # elif evaluation == 'detection':
                    #     second_best_over_vs = all_over_vs.min()
                    # make_underlined = results[evaluation][model_type][metric]['avg_per_version'][v] == second_best_over_vs

                    if make_bold:
                        all_rows += f" & \\textbf{{{results[evaluation][model_type][metric]['avg_per_version'][v]:.3f}}}"
                    # elif make_underlined:
                    #     all_rows += f" & \\underline{{{results[evaluation][model_type][metric]['avg_per_version'][v]:.3f}}}"
                    else:
                        all_rows += f" & {results[evaluation][model_type][metric]['avg_per_version'][v]:.3f}"
            record_heading = False

            all_rows += f'\\\\ \n'
        all_rows = heading + f'\\\\ \n {"*"*10}  \n' + all_rows
        print(all_rows)

