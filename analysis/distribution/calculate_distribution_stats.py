import argparse

import pandas as pd
import numpy as np
import torch
import wandb
import datetime
from scipy.stats import wasserstein_distance 
from scipy.spatial.distance import jensenshannon
from utils import read_csv, set_seed, _load_json
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl

def _parse_args():
    parser = argparse.ArgumentParser(description=' Command Line Interface')
    parser.add_argument("model_type", type=str, choices=['wgan', 'tablegan', 'ctgan', 'octgan', "tvae", "goggle"])
    parser.add_argument("--prefix", type=str, default='')
    parser.add_argument("--wandb_ignore", action='store_true')
    return parser.parse_args()


def get_model_paths(model_type):
    model_paths = _load_json(f"./reeval_path_{model_type}_dist.json")[model_type]
    return model_paths

def calculate_distances(train_data, generated_data, cat_idx, cont_idx, model_type, use_case, version, seed_run, postprocessing):
    wd, full_wd, full_jsd, jsd = [], [], [], []
    for data in generated_data:
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        elif isinstance(data, torch.Tensor):
            data = data.detach().numpy()

        wd_distances = []
        for i in cont_idx:
            cont_idx_train_data = train_data[:, i].reshape(-1, 1)
            cont_idx_gen_data = data[:, i].reshape(-1, 1)

            # fit the minmax scaler on the cont feature train data
            scaler = MinMaxScaler(feature_range=(0, 1), copy=True, clip=True).fit(cont_idx_train_data)
            cont_idx_train_data = scaler.transform(cont_idx_train_data).reshape(-1)
            cont_idx_gen_data = scaler.transform(cont_idx_gen_data).reshape(-1)

            cont_idx_wasserstein_distance = wasserstein_distance(cont_idx_train_data, cont_idx_gen_data)
            wd_distances.append(cont_idx_wasserstein_distance)

        full_wd.append(wd_distances)
        wd.append(np.nanmean(wd_distances))

        real = np.round(train_data[:,cat_idx],2)
        gen = np.round(data[:, cat_idx],2)
        js_div = jensenshannon(real, gen, axis=0)
        full_jsd.append(js_div)
        jsd.append(np.nanmean(js_div))

    if version != 'unconstrained':
        version = 'constrained'
    pkl_path_name = f'{model_type}_{use_case}_{version}_{seed_run}'
    if postprocessing:
        pkl_path_name += '_postprocessing'
    with open(f'calculation_distributions/{pkl_path_name}_wd.pkl', 'wb') as f:
        pkl.dump(full_wd, f, -1)

    with open(f'calculation_distributions/{pkl_path_name}_jsd.pkl', 'wb') as f:
        pkl.dump(full_jsd, f, -1)

    return np.mean(wd), np.mean(jsd)

if __name__ == "__main__":
    args = _parse_args()
    model_type = args.model_type
    wandb_project = "MINMAX_dist_stats_round"

    dataset_info = _load_json("datasets_info.json")
    model_paths = get_model_paths(model_type)

    results = []
    # for use_case in model_paths.keys():  
        # if use_case not in ["url"]:
        #     continue      
    for use_case in model_paths.keys():
        if use_case == 'botnet':
            continue
        X_train = pd.read_csv(f"data/{use_case}/train_data.csv")
        if use_case == 'lcld':
            X_train = pd.read_csv(f"data/{use_case}/tiny/train_data.csv")

        cat_idx = dataset_info[use_case]["manual_inspection_categorical_cols_idx"]
        cont_idx = list(set(np.arange(0,len(X_train.columns))) - set(cat_idx))
        X_train = X_train.to_numpy()

        for version in model_paths[use_case].keys():
            wds, jsds = [], []
            for k,seed_run in enumerate(model_paths[use_case][version]):
                try:
                    #print(args.prefix + '/' + seed_run+"/generated_data.pkl")
                    #generated_data = pd.read_pickle(args.prefix + '/' + seed_run+"/generated_data.pkl")
                    if model_type == 'wgan' and version == 'unconstrained':
                        generated_data = pd.read_pickle(args.prefix + seed_run+"/unconstrained_generated_data.pkl")
                    else:
                        generated_data = pd.read_pickle(args.prefix + seed_run+"/generated_data.pkl")

                    print("Seed loaded", args.prefix + seed_run)
                except Exception as e:
                    #print(e)
                    continue

                if model_type == 'tablegan':
                    try: 
                        generated_data = {"train":generated_data[0], "val":generated_data[1], "test":generated_data[2]}
                    except Exception as e:
                        pass

                wd, jsd = calculate_distances(X_train, generated_data["train"], cat_idx, cont_idx, model_type, use_case, version,k, postprocessing=False)
                print(wd, jsd)
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

