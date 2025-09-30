import pandas as pd

use_cases = ["lcld", "botnet"]
splits = ["train", "val", "test"]

for use_case in use_cases:
    for split in splits:
        if use_case == "lcld":
            target = 'charged_off'
        elif use_case == "botnet":
            target = "is_botnet"
        df = pd.read_csv(f"data/{use_case}/{split}_data.csv")
        df = df.groupby(target, group_keys=False).apply(lambda x: x.sample(frac=0.1)).sample(frac=1).reset_index(drop=True)
        df.to_csv(f"data/{use_case}/tiny_{split}_data.csv", index=False) 
