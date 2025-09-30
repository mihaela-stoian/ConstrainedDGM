import argparse

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from utils import _load_json


def _parse_args():
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    parser.add_argument("use_case", type=str, choices=['url', 'wids', 'lcld', 'botnet', 'heloc',' news', 'faults'])
    parser.add_argument("--num_samples", type=int, default=3000)
    return parser.parse_args()


def main():
    args = _parse_args()
    dataset_info = _load_json("datasets_info.json")[args.use_case]
    target_col = dataset_info["target_col"]

    train_data = pd.read_csv(f"./data/{args.use_case}/train_data.csv")
    df = pd.DataFrame(train_data)
    print('Length of the dataset is:', len(train_data))

    if args.num_samples > len(df):
        raise Exception("Num samples exceeds the dataset size!")
        exit()

    tiny_ratio = args.num_samples/len(df)
    tiny_ids = df.groupby(target_col, group_keys=False).apply(lambda x: x.sample(frac=tiny_ratio)).sample(frac=1).index #.reset_index(drop=True)
    tiny_train_data = df.iloc[tiny_ids].reset_index(drop=True)
    num_samples_actual = len(tiny_train_data)

    tiny_train_data.to_csv(f"data/{args.use_case}/train_data_{num_samples_actual}.csv", index=False)


if __name__ == '__main__':
    main()
