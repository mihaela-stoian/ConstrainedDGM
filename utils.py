import wandb
import pandas as pd
import torch
import random
import os
import numpy as np
import logging
import sys 
import json




def all_div_gt_n(n, m):
    for i in range(n, m // 2):
         if m % i == 0:
            return i
    return 1

def wandb_raw_data(wandb_run):
    for partition in ['train', 'val', 'test']:
        filename = 'data/url/{:}_data.csv'.format(partition)

        raw_data = pd.read_csv(filename)
        raw_data = wandb.Table(dataframe=raw_data)
        artifact = wandb.Artifact("train_URL", type="dataset")
        artifact.add(raw_data, 'raw_data_artifact_{:}'.format(partition))
        artifact.add_file(filename)
        # Log the table to visualize with a run...
        wandb_run.log({"raw_data_{:}".format(partition): raw_data})
        wandb_run.log_artifact(artifact)
    return wandb_run

# seed everything https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")



def _load_json(path):
    with open(path) as json_file:
        return json.load(json_file)

def get_args():
    args = ArgumentParser()
    args.add_argument("expdir", type=str)
    args.add_argument("--log_wandb", action='store_true')
    args.add_argument("--eval_epoch", default=None, type=int)
    return args.parse_args()

def round_func_BPDA(input):
    # This is equivalent to replacing round function (non-differentiable) with
    # an identity function (differentiable) only when backward.
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out


def metrics_to_wandb(metrics):
    columns = []
    data = []
    for k_prim, v_prim in metrics.items():
        for k_sec, v_sec in v_prim.items():
            columns.append(f"{k_prim}.{k_sec}")
            data.append(v_sec)
    df = pd.DataFrame(data=[data], columns=columns, index=[0])
    return df


def log_dict(name, results, logger):
    logger.info(f'\n{name}')
    for key in results:
        log_result = f"{key}: {results[key][0]}"
        logger.info(log_result)


def get_roundable_data(df):
    _is_roundable = ((df%1)==0).all(axis=0)
    roundable_cols = df.columns[_is_roundable]
    roundable_idx = [df.columns.get_loc(c) for c in roundable_cols]
    round_digits = df.iloc[:,roundable_idx].apply(get_round_decimals)
    return roundable_idx, round_digits


def get_round_decimals(col):
    MAX_DECIMALS = sys.float_info.dig - 1
    if (col == col.round(MAX_DECIMALS)).all():
        for decimal in range(MAX_DECIMALS + 1):
         if (col == col.round(decimal)).all():
             return decimal


def single_value_cols(df):
    a = df.to_numpy()
    single_value = (a[0] == a).all(0)
    return df.columns[single_value].to_list()

def read_csv(csv_filename, use_case="", manual_inspection_cat_cols_idx=[]):
    """Read a csv file."""
    data = pd.read_csv(csv_filename)
    single_val_col = single_value_cols(data)
    roundable_idx, round_digits = get_roundable_data(data)

    # TODO: Create configuration files
    # TODO: Unify data loading for WGAN and the rest of the models
    cat_cols_names = data.columns[manual_inspection_cat_cols_idx].values.tolist()
    for col in single_val_col:
        try:
            cat_cols_names.remove(col)
        except Exception as e:
            pass
    bin_cols_idx = [data.columns.get_loc(c) for c in cat_cols_names if c in data]
    roundable_idx = [i for i in roundable_idx if i not in bin_cols_idx]
    round_digits = round_digits[data.columns[roundable_idx]]

    if len(bin_cols_idx) == 0:
        bin_cols_idx = None
        cat_cols_names = None
    return data, (cat_cols_names, bin_cols_idx), (roundable_idx, round_digits)