import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from argparse import ArgumentParser
# from WGAN_uncons_pac import *
import pickle as pkl

import pandas as pd

from gather_results.plotting_results.get_overview_results import get_avg_cols

datasets = ['URL', 'WIDS', 'LCLD', 'HELOC', 'FAULTS', 'NEWS']
EXCLUDED_DATASETS = ['URL', 'LCLD', 'HELOC', 'FAULTS', 'NEWS']
# EXCLUDED_DATASETS = ['LCLD']
versions = ['uncons.', 'random', 'corr', 'kde', 'wasserstein', 'causal']
mapped_version = {'uncons.':'-', 'random':'Rnd', 'kde':'KDE', 'corr':'Corr', 'wasserstein':'WD', 'causal': 'Caus.'}
mapped_model_type = {'wgan': 'WGAN', 'tablegan': 'TableGAN', 'ctgan': 'CTGAN', 'tvae': 'TVAE', 'goggle': 'GOGGLE'}
evaluations = ['detection', 'utility']
dgm_models = ['WGAN', 'TableGAN', 'CTGAN', 'TVAE', 'GOGGLE']
model_types = ['wgan', 'tablegan', 'ctgan', 'tvae', 'goggle']
# model_types = ['tablegan']


def get_args():
    args = ArgumentParser()
    args.add_argument("evaluation", type=str, choices=['utility','detection'])
    args.add_argument("metric", type=str, choices=['binary_f1', 'weighted_f1', 'auc'])
    args.add_argument("--include_stddev", action='store_true')
    args.add_argument("--postprocessing", action='store_true')
    # args.add_argument("version", type=str, choices=['uncons.', 'random', 'kde', 'corr'])
    # args.add_argument("expdir", type=str)
    return args.parse_args()


def get_results(args):
    bypass = ''
    if args.model_type == 'wgan':
        model_type = 'WGAN'
    elif args.model_type == 'tablegan':
        model_type = 'TableGAN'
    elif args.model_type == 'tvae':
        model_type = 'TVAE'
    elif args.model_type == 'goggle':
        model_type = 'GOGGLE'
    elif args.model_type == 'ctgan':
        model_type = 'CTGAN'
        # bypass = 'bypass_'
        bypass = ''

    suffix = '_eval'
    if args.model_type == 'ctgan':
        suffix += '_final'

    if args.metric == 'binary_f1':
        metric = 'BinaryF1'
    elif args.metric == 'weighted_f1':
        metric = 'WeightedF1'
    elif args.metric == 'auc':
        metric = 'AUC'
    elif args.metric == 'avg':
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
        version_row_map = {'uncons.': 2, 'random': 3, 'kde': 4, 'corr': 5, 'wasserstein': 6, 'causal': 7}
    else:
        version_row_map = {'uncons.':2, 'random':8, 'kde':9, 'corr':10, 'wasserstein':11, 'causal':12}
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
                    if '#DIV/0!' in value or '#REF!' in value:
                        value = 'NaN'
                rdict[dataset][eval][v] = float(value)/100.
    return rdict

def create_results_dict_main(results, evaluation):
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
        version_row_map = {'uncons.': 2, 'random': 3, 'kde': 4, 'corr': 5, 'wasserstein': 6, 'causal': 7}
    else:
        version_row_map = {'uncons.':2, 'random':8, 'kde':9, 'corr':10, 'wasserstein':11, 'causal':12}
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
                    if '#DIV/0!' in value or '#REF!' in value:
                        value = 'NaN'
                value = float(value)/100.
                if v != 'uncons.':
                    baseline_res = rdict[dataset][eval]['uncons.']
                    if evaluation == 'utility':
                        value = baseline_res + value
                    elif evaluation == 'detection':
                        value = baseline_res - value
                rdict[dataset][eval][v] = value
    return rdict


def get_tex_table_rows(args, results, stddevs, include_stddev=False):
    evaluation = args.evaluation
    metric = args.metric
    model_type = args.model_type

    all_rows = ''
    # for evaluation in evaluations:
    #     print(f'\n\nTABLE FOR {evaluation}, postprocessing is {args.postprocessing}')
    #     all_rows = ''
    #     heading = ' '
    #     record_heading = True
    for v in versions:
        if v == 'uncons.':
            all_rows += f'{mapped_model_type[args.model_type]} & '
        elif v == 'random':
            # all_rows += "\parbox[t]{3mm}{\multirow{3}{*}{\\rotatebox[origin=c]{90}" + f'{{C-{mapped_model_type[args.model_type]}}}}}}} & '
            all_rows += "\multirow{3}{*}" + f'{{C-{mapped_model_type[args.model_type]}}} & '
        else:
            all_rows += " &  "
        all_rows += f'{mapped_version[v]} & '

        row_res = []
        for dataset in datasets:
            if dataset == 'NEWS' and args.evaluation == 'utility':
                continue
            # results = rdict[dataset][args.evaluation]
            # baseline_res = results['uncons.']
            # res = results[v]

            # make bold if needed:
            all_over_vs = np.array([results[evaluation][model_type][metric]['raw'][dataset][evaluation][vi] for vi in versions])
            if evaluation == 'utility':
                best_over_vs = all_over_vs.max()
            elif evaluation == 'detection':
                best_over_vs = all_over_vs.min()
            make_bold = results[evaluation][model_type][metric]['raw'][dataset][evaluation][v] == best_over_vs

            if make_bold:
                main_res = f"\\textbf{{{results[evaluation][model_type][metric]['raw'][dataset][evaluation][v]:.3f}}}"
            else:
                main_res = f"{results[evaluation][model_type][metric]['raw'][dataset][evaluation][v]:.3f}"


            if not include_stddev:
                pass
                # if v != 'uncons.':
                #     improvement = res
                #     if args.evaluation == 'utility':
                #         res = baseline_res + res
                #     elif args.evaluation == 'detection':
                #         res = baseline_res - res
                #     sign = '+' if improvement > 0 else '-'
                #     difference = np.abs(improvement)
                #     res = f'{res:.3f} (${sign}${difference:.3f})'
                #
                # else:
                #     res = f'{results[v]:.3f}'
            else:
                # res = f'{main_res} ({stddevs[dataset][args.evaluation][v]:.3f})'
                # if dataset in EXCLUDED_DATASETS and model_type == 'goggle':
                if dataset in EXCLUDED_DATASETS:
                    res = f'---'
                else:
                    # res = f'\pgreen{{{main_res}\msmall{{\pm{{{stddevs[dataset][args.evaluation][v]:.3f}}}}}}}'
                    res = f'{main_res}\msmall{{\pm{{{stddevs[dataset][args.evaluation][v]:.3f}}}}}'
            if model_type == 'tvae':
                res = f'\\add{{{res}}}'
            elif model_type == 'goggle':
                res = f'\\add{{{res}}}'
            row_res.append(res)

        row_results = ''
        row_results += ' & '.join(row_res)
        row_results += ' \\\\'
        # print(row_results)
        all_rows += row_results + '\n'
    return all_rows

# \msmall{\pm{43}}
if __name__ == "__main__":
    args = get_args()
    for mm in ['binary_f1', 'weighted_f1', 'auc']:
        args.metric = mm
        results = {evaluation: {model_type: {args.metric:{}} for model_type in model_types} for evaluation in evaluations}
        for evaluation in evaluations:
            for model_type in model_types:
                # for metric in metrics:
                    metric = args.metric
                    args.model_type = model_type
                    temp, stddevs = get_results(args)

                    rdict = create_results_dict(temp)
                    rdict_main = create_results_dict_main(temp, evaluation)
                    if stddevs is not None:
                        stddevs = create_results_dict(stddevs)

                    results[evaluation][model_type][metric]['raw_improvement'] = rdict
                    results[evaluation][model_type][metric]['raw'] = rdict_main
                    results[evaluation][model_type][metric]['raw_stddev'] = stddevs
                    results[evaluation][model_type][metric]['avg'] = {v:-1 for v in versions}

                    # print(rdict)
                    col = get_avg_cols(evaluation, rdict, stddevs, include_stddev=args.include_stddev)
                    results[evaluation][model_type][metric]['avg_per_version'] = col


        row_results = []
        for model_type in model_types:
            args.model_type = model_type
            print(args.model_type, args.evaluation, args.metric, '        postprocessing?', args.postprocessing)
            _, stddevs = get_results(args)
            # results_rdict, stddevs = get_results(args)
            # rdict = create_results_dict(results_rdict)
            if stddevs is not None:
                stddevs = create_results_dict(stddevs)
            # print(rdict)
            res = get_tex_table_rows(args, results, stddevs, include_stddev=args.include_stddev)
            row_results.append(res)
            row_results.append('\midrule')

        print()
        for res in row_results:
            print(res)

