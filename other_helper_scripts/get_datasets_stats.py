import pandas as pd 
from utils import _load_json

use_cases = ["url", "wids", "lcld", "botnet", "heloc", "faults", "news"]
train_size = []
val_size = []
test_size = []
cat_cols = []
num_cols = []
tasks = []


for use_case in use_cases:
    X_train = pd.read_csv(f"data/{use_case}/train_data.csv")
    train_size.append(X_train.shape[0])

    X_test = pd.read_csv(f"data/{use_case}/test_data.csv")
    test_size.append(X_test.shape[0])

    X_val = pd.read_csv(f"data/{use_case}/val_data.csv")
    val_size.append(X_val.shape[0])

    dataset_info = _load_json("datasets_info.json")[use_case]

    cat_col = len(dataset_info["manual_inspection_categorical_cols_idx"])
    cat_cols.append(cat_col)

    num_col = X_train.shape[1] - cat_col
    num_cols.append(num_col)

    tasks.append(dataset_info["problem_type"])


df = pd.DataFrame([use_cases, train_size, val_size, test_size, cat_cols, num_cols, tasks]).transpose()
df.columns = ["Dataset", "train_size", "val_size", "test_size", "cat_cols", "num_cols", "tasks"]
df.to_csv("dataset_info.csv", index=False)