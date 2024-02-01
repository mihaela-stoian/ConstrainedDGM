import numpy as np
import pandas as pd
import joblib
from data_processors.wgan.tab_scaler import TabScaler


def prepare_data_torch_scaling(train_data, use_case, bin_cols_idx):
    train_data = train_data.to_numpy()
    scaler = TabScaler(one_hot_encode=True)
    scaler.fit(train_data, cat_idx = bin_cols_idx)
    #joblib.dump(scaler, f"WGAN_out/{use_case}/{use_case}_torch_scaler.joblib")
    train_data = scaler.transform(train_data)
    return train_data, scaler






