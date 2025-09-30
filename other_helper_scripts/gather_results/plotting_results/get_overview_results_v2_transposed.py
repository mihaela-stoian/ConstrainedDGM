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
INCLUDED_DATASETS = ['WIDS']
# INCLUDED_DATASETS = datasets

versions = ['uncons.', 'random', 'corr', 'kde', 'wasserstein', 'causal']
INCLUDED_VERSIONS = ['random', 'corr', 'kde', 'wasserstein', 'causal']

mapped_version = {'uncons.':'-', 'random':'Rnd', 'kde':'KDE', 'corr':'Corr', 'wasserstein':'WD', 'causal': 'Caus'}
metrics = ['BinaryF1', 'WeightedF1', 'AUC']
# evaluations = ['detection', 'utility']
evaluations = ['utility', 'detection']
dgm_models = ['WGAN', 'TableGAN', 'CTGAN', 'TVAE', 'GOGGLE']
mapped_model_type = {'wgan': 'WGAN', 'tablegan': 'TableGAN', 'ctgan': 'CTGAN', 'tvae': 'TVAE', 'goggle': 'GOGGLE'}


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
    elif model_type == 'goggle':
        model_type = 'GOGGLE'
    elif model_type == 'tvae':
        model_type = 'TVAE'
    elif model_type == 'ctgan':
        model_type = 'CTGAN'
        # bypass = 'bypass_'
        bypass = ''

    suffix = '_eval'
    if model_type == 'CTGAN':
        suffix += '_final'

    if metric == 'binary_f1':
        metric = 'BinaryF1'
    elif metric == 'weighted_f1':
        metric = 'WeightedF1'
    elif metric == 'auc':
        metric = 'AUC'
    elif metric == 'avg':
        metric = 'Average'

    filename = f'./results/{bypass}{model_type}{suffix} - {metric}_results_{model_type}.csv'
    results = pd.read_csv(filename)

    filename_stddevs = f'./results/{bypass}{model_type}{suffix} - {metric}_stddev_{model_type}.csv'
    if Path(filename_stddevs).exists():
        stddevs = pd.read_csv(filename_stddevs)
    else:
        stddevs = None
        print(filename_stddevs, 'does not exist!')
    return results, stddevs


def create_results_dict(results):
    rdict = {dataset: {'utility': {v:-1 for v in versions}, 'detection': {v:-1 for v in versions}} for dataset in INCLUDED_DATASETS}
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
        version_row_map = {'uncons.': 2, 'random': 3, 'kde': 4, 'corr': 5, 'wasserstein': 6, 'causal': 7}
    else:
        version_row_map = {'uncons.':2, 'random':8, 'kde':9, 'corr':10, 'wasserstein':11, 'causal':12}
    eval_col_offset_map = {'utility': 1, 'detection': 2}

    for dataset in INCLUDED_DATASETS:
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
                rdict[dataset][eval][v] = np.round(float(value) /100.,3)
    return rdict


def get_avg_cols(evaluation, rdict, stddevs, include_stddev=False):
    col_results = {}
    for v in INCLUDED_VERSIONS:
        avg_res = 0
        num_res = 0
        for dataset in INCLUDED_DATASETS:
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
    print(f'postprocessing is {args.postprocessing}')
    print(f'INCLUDED DATASETS are: {INCLUDED_DATASETS}')
    print(f'INCLUDED VERSIONS are: {INCLUDED_VERSIONS}')
    model_types = ['wgan', 'tablegan', 'ctgan', 'tvae', 'goggle']
    # model_types = ['wgan', 'tablegan', 'ctgan', 'tvae']
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
                results[evaluation][model_type][metric]['avg'] = {v:-1 for v in INCLUDED_VERSIONS}

                # print(rdict)
                col = get_avg_cols(evaluation, rdict, stddevs, include_stddev=args.include_stddev)
                results[evaluation][model_type][metric]['avg_per_version'] = col

    for model_type in model_types:
        # print(f'\n\nTABLE FOR {model_type}, postprocessing is {args.postprocessing}')
        prefix = 'C-' if not args.postprocessing else 'P-'
        all_rows = ''
        heading = ' '
        record_heading = True
        for ii, v in enumerate(INCLUDED_VERSIONS):
            if ii == 0:
                all_rows += "\multirow{3}{*}{" + f'{prefix}{mapped_model_type[model_type]}' +"} " + " & "
            else:
                all_rows += " & "

            # if v == 'uncons.':
            #     # continue
            #     # if evaluation == 'detection':
            #     #     all_rows += "{\multirow{4}{*}{Detection}&  "
            #     # elif evaluation == 'utility':
            #     #     all_rows += "{\multirow{4}{*}{Utility}&  "
            #     all_rows += "\multirow{3}{*}{" + f'{prefix}{mapped_model_type[model_type]}' +"} " + " & "
            #     continue
            # else:
            #     all_rows += " & "
            all_rows += f'{mapped_version[v]}'
            # for evaluation in evaluations:
            # if record_heading:
            # #     heading += f' & {evaluation}'
            #     print(f'    {model_type}')
            for ii,evaluation in enumerate(evaluations):
                if ii >= 0:
                    all_rows += " & "
                # if record_heading:
                #     # heading += f' & {model_type}'
                #     print(f'        {model_type}')
                for metric in metrics:
                    # if record_heading:
                    #     heading += f' & {metric}'
                    #     print(f'            {metric}')

                    # make bold if needed:
                    all_over_vs = np.array([results[evaluation][model_type][metric]['avg_per_version'][vi] for vi in INCLUDED_VERSIONS])
                    if evaluation == 'utility':
                        best_over_vs = all_over_vs.max()
                    elif evaluation == 'detection':
                        best_over_vs = all_over_vs.min()
                    make_bold = results[evaluation][model_type][metric]['avg_per_version'][v] == best_over_vs

                    all_over_vs = np.array([results[evaluation][model_type][metric]['avg_per_version'][vi] for vi in INCLUDED_VERSIONS if results[evaluation][model_type][metric]['avg_per_version'][vi]!=best_over_vs])
                    # if evaluation == 'utility':
                    #     second_best_over_vs = all_over_vs.max()
                    # elif evaluation == 'detection':
                    #     second_best_over_vs = all_over_vs.min()
                    # make_underlined = results[evaluation][model_type][metric]['avg_per_version'][v] == second_best_over_vs

                    if make_bold:
                        temp_row_res = f"\\textbf{{{results[evaluation][model_type][metric]['avg_per_version'][v]:.3f}}}"

                    # elif make_underlined:
                    #     all_rows += f" & \\underline{{{results[evaluation][model_type][metric]['avg_per_version'][v]:.3f}}}"
                    else:
                        temp_row_res = f"{results[evaluation][model_type][metric]['avg_per_version'][v]:.3f}"
                    # if model_type == 'tvae':
                    #     temp_row_res = f'\\add{{{temp_row_res}}}'
                    # elif model_type == 'goggle':
                    #     temp_row_res = f'\\add{{{temp_row_res}}}'
                    temp_row_res = f'\\add{{{temp_row_res}}}'
                    all_rows += f" & {temp_row_res}"
            record_heading = False

            all_rows += f'\\\\ \n'
        # all_rows = heading + f'\\\\ \n {"*"*10}  \n' + all_rows
        all_rows = heading + "\cmidrule{1-1} \n" + all_rows
        print(all_rows)

