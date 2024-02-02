import argparse
import datetime
import pandas as pd
import wandb

from utils import set_seed, read_csv, _load_json
from evaluation.stasy_utility_detection import compute_utility_real
DATETIME = datetime.datetime.now()

def _parse_args():
    parser = argparse.ArgumentParser(description='Utility Real Command Line Interface')
    parser.add_argument("--use_case", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--seed", type=int)

    return parser.parse_args()

def main():
    """CLI."""
    args = _parse_args()   
    set_seed(args.seed)
    dataset_info = _load_json("datasets_info.json")[args.use_case]
    X_train, (cat_cols, cat_idx), (roundable_idx, round_digits) = read_csv(f"data/{args.use_case}/train_data.csv", args.use_case, dataset_info["manual_inspection_categorical_cols_idx"])
    X_test = pd.read_csv(f"data/{args.use_case}/test_data.csv")
    X_val = pd.read_csv(f"data/{args.use_case}/val_data.csv")
    columns = X_train.columns.values.tolist()



    if args.use_case == "botnet" or args.use_case == "lcld":
        X_train = pd.read_csv(f"data/{args.use_case}/tiny/train_data.csv")
        X_test = pd.read_csv(f"data/{args.use_case}/tiny/test_data.csv")
        X_val = pd.read_csv(f"data/{args.use_case}/tiny/val_data.csv")
    ##################################################################################
    exp_id = f"{args.use_case}_{args.seed}_{DATETIME:%d-%m-%y--%H-%M-%S}"
    wandb_run = wandb.init(project=args.wandb_project, id=exp_id, reinit=True)
    for k,v in args._get_kwargs():
        wandb_run.config[k] = v
    #################################################################################
    real_data = {"train": X_train, "val": X_val, "test": X_test}
    utility_real=compute_utility_real(real_data, None,  dataset_info["problem_type"], columns, dataset_info["target_col"], dataset_info["target_size"], True)

    utility_real[3].columns = ["Utility_real.binary_f1", "Utility_real.roc_auc", "Utility_real.weighted_f1", "Utility_real.accuracy"]
    wandb_run.log({f"INFERENCE/aggregated/utility_real": wandb.Table(dataframe=utility_real[3])})

if __name__ == '__main__':
    main()
