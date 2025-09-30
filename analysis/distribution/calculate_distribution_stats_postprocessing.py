import argparse

import pandas as pd
import numpy as np
import torch
import wandb
import datetime
from scipy.stats import wasserstein_distance 
from scipy.spatial.distance import jensenshannon

from calculate_distribution_stats import calculate_distances
from cdgm.constraints_code.compute_sets_of_constraints import compute_sets_of_constraints
from cdgm.constraints_code.feature_orderings import set_ordering
from cdgm.constraints_code.parser import parse_constraints_file
from run_evaluation.reeval_final import prepare_gen_data
from utils import read_csv, set_seed, _load_json


def _parse_args():
    parser = argparse.ArgumentParser(description=' Command Line Interface')
    parser.add_argument("model_type", type=str, choices=['wgan', 'tablegan', 'ctgan', 'octgan', "tvae", "goggle"])
    parser.add_argument("--prefix", type=str, default='./')
    parser.add_argument("--wandb_ignore", action='store_true')
    return parser.parse_args()



def get_model_paths(model_type):
    model_paths = _load_json(f"./reeval_path_{model_type}_dist.json")[model_type]
    return model_paths


if __name__ == "__main__":
    args = _parse_args()
    model_type = args.model_type
    args.postprocessing = True

    wandb_project = "MINMAX_dist_stats_POSTPROCESSING_round"
    #######################################################
    ##################################################################
    dataset_info = _load_json("datasets_info.json")
    best_postprocessing_orderings = _load_json("best_postprocessing_orderings.json")
    model_paths = get_model_paths(model_type)

    results = []
    for use_case in model_paths.keys():
        if use_case == 'botnet':
            continue
        X_train = pd.read_csv(f"data/{use_case}/train_data.csv")
        if use_case == 'lcld':
            X_train = pd.read_csv(f"data/{use_case}/tiny/train_data.csv")

        original_X_train = X_train
        columns = X_train.columns
        cat_idx = dataset_info[use_case]["manual_inspection_categorical_cols_idx"]
        cont_idx = list(set(np.arange(0,len(X_train.columns))) - set(cat_idx))
        X_train = X_train.to_numpy()

        version = 'unconstrained'
        label_ordering = best_postprocessing_orderings[model_type][use_case]
        wds, jsds = [], []
        for k, seed_run in enumerate(model_paths[use_case][version]):
            try:
                if version == 'unconstrained' and args.postprocessing:
                    unconstrained_generated_data = pd.read_pickle(args.prefix + seed_run + "/unconstrained_generated_data.pkl")
                    # if model_type == 'tablegan':
                    #     unconstrained_generated_data = {"train": unconstrained_generated_data[0], "val": unconstrained_generated_data[1],
                    #                       "test": unconstrained_generated_data[2]}
                    if model_type == 'goggle':
                        unconstrained_generated_data = {
                            "train": [e.to_numpy() for e in unconstrained_generated_data['train']],
                            "val": [e.to_numpy() for e in unconstrained_generated_data['val']],
                            "test": [e.to_numpy() for e in unconstrained_generated_data['test']]}

                    constraints_file = f'./data/{use_case}/{use_case}_constraints.txt'
                    ordering, constraints = parse_constraints_file(constraints_file)

                    if '_out' in model_type:
                        post_model_type = model_type.lower()[:-4]
                    else:
                        post_model_type = model_type

                    print('MODEL TYPE FOR SETTING ORDERING', post_model_type)
                    ordering = set_ordering(use_case, ordering, label_ordering, post_model_type)
                    sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)

                    args.label_ordering = label_ordering
                    args.use_case = use_case
                    args.constraints_file = constraints_file
                    args.round_before_cons = False
                    args.version = version
                    args.round_after_cons = False
                    args.use_only_target_original_dtype = False
                    generated_data, _ = prepare_gen_data(args, unconstrained_generated_data,
                                                                                None, None, columns,
                                                                                original_X_train)

                else:
                    continue
                print("Seed loaded", args.prefix + seed_run)
            except Exception as e:
                # print(e)
                continue


            wd, jsd = calculate_distances(X_train, generated_data["train"], cat_idx, cont_idx, model_type, use_case, version,k, postprocessing=True)
            wds.append(wd)
            jsds.append(jsd)
        if wds:
            row = [use_case, version, np.nanmean(wds), np.nanstd(wds), np.nanmean(jsds), np.nanstd(jsds)]
            print("Row", row)
            results.append(row)
    if results and not args.wandb_ignore:
        #######################################################
        DATETIME = datetime.datetime.now()
        exp_id = f"{model_type}_{DATETIME:%d-%m-%y--%H-%M-%S}"
        wandb_run = wandb.init(project=wandb_project, id=exp_id, reinit=True)
        wandb_run.config["model_type"] = model_type
        ##################################################################
        results = pd.DataFrame(results, columns=[ "Dataset", "Version", "Mean WD", "STD WD", "Mean JSD", "STD JSD"])
        print(results)
        wandb.log({"Stats":results})

