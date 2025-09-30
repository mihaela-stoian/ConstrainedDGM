import joblib
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing._data import _handle_zeros_in_scale

from cdgm.data_processors.wgan.tab_scaler import TabScaler, _handle_zeros_in_scale_torch, softargmax, get_cat_idx


def prepare_data_torch_scaling(train_data, use_case, bin_cols_idx):
    train_data = train_data.to_numpy()
    scaler = TabScaler(one_hot_encode=True)
    scaler.fit(train_data, cat_idx = bin_cols_idx)
    #joblib.dump(scaler, f"WGAN_out/{use_case}/{use_case}_torch_scaler.joblib")
    train_data = scaler.transform(train_data)
    return train_data, scaler


def prepare_data_torch_scaling_orderings(train_data, use_case, scaling,  bin_cols_idx, scale):
    train_data = train_data.to_numpy()
    if scaling:
        scaler = MinMaxScaler(*scale)
        scaler.fit(train_data, cat_idx = bin_cols_idx)
        joblib.dump(scaler, f"WGAN_out/{use_case}/{use_case}_torch_scaler.joblib")
        train_data = scaler.transform(train_data)
    return train_data


class MinMaxScaler:
    def __init__(self, out_min: float = 0.0, out_max: float = 1.0) -> None:

        self.out_min = out_min
        self.out_max = out_max

        if self.out_min >= self.out_max:
            raise ValueError("out_min must be smaller than out_max")

        self.x_min = None
        self.x_max = None
        self.categories = None
        self.cat_idx = None
        self.encoder = None
        self.num_idx = []

    def fit(self, x, cat_idx=None, x_type=None):
        if isinstance(x, np.ndarray):
            nb_features = x.shape[1]
            self.cat_idx = cat_idx

            if self.cat_idx is None:
                if x_type is not None:
                    self.cat_idx = self.cat_idx = get_cat_idx(x_type)

            self.num_idx = range(nb_features)
            if self.cat_idx is not None:
                self.encoder = OneHotEncoder(sparse=False)
                self.encoder.fit(x[:, self.cat_idx])
                self.num_idx = [
                    e for e in self.num_idx if e not in self.cat_idx
                ]

        else:
            raise NotImplementedError

        self.x_min = np.min(x[:, self.num_idx], axis=0)
        self.x_max = np.max(x[:, self.num_idx], axis=0)
        return self

    def transform(self, x, cat_values=None):
        nb_features = x.shape[1]

        if isinstance(x, torch.Tensor):
            scale = torch.Tensor(self.x_max) - torch.Tensor(self.x_min)
            scale = _handle_zeros_in_scale_torch(scale)
            x_out = (x[:, self.num_idx] - torch.tensor(self.x_min)) / scale
            x_out = x_out * (torch.tensor(self.out_max) - torch.tensor(self.out_min)) + torch.tensor(self.out_min)

        else:
            scale = self.x_max - self.x_min
            scale = _handle_zeros_in_scale(scale)
            x_out = (x[:, self.num_idx] - self.x_min) / scale
            x_out = x_out * (self.out_max - self.out_min) + self.out_min
            x_out = np.nan_to_num(x_out, nan=0)

            if self.cat_idx is not None:
                x_cat = self.encoder.transform(x[:, self.cat_idx])
                x_out = np.concatenate([x_cat, x_out, ], axis=1)

        return x_out

    def inverse_transform(self, x):

        nb_features = x.shape[1]
        x_cat = x[:, 0: nb_features - len(self.num_idx)]
        x_num = x[:, nb_features - len(self.num_idx): nb_features]

        if isinstance(x, torch.Tensor):
            # process as tensor to preserve gradient
            # scale = torch.Tensor(self.x_max) - torch.Tensor(self.x_min)
            # scale = _handle_zeros_in_scale_torch(scale)
            x_num = (x_num - torch.Tensor([self.out_min])) / (
                        torch.Tensor([self.out_max]) - torch.Tensor([self.out_min]))
            x_num = x_num * (torch.Tensor(self.x_max) - torch.Tensor(self.x_min)) + torch.Tensor(self.x_min)
            x_num = torch.clip(x_num, torch.Tensor(self.x_min), torch.Tensor(self.x_max))

            # Special case for binary features
            if self.cat_idx is not None:
                x_cat_encoded = torch.split(x_cat, [len(a) for a in self.encoder.categories_], 1)
                x_cat_softmax = [softargmax(a, 1) for a in x_cat_encoded]
                x_cat_unencoded = torch.stack(x_cat_softmax).swapaxes(0, 1)
                x_reversed = torch.zeros((x.shape[0], x_num.shape[1] + x_cat_unencoded.shape[1]))
            else:
                x_reversed = torch.zeros((x.shape[0], x_num.shape[1]))

        else:
            # process as numpy
            scale = self.x_max - self.x_min
            scale = _handle_zeros_in_scale(scale)
            x_num = (x_num - self.out_min) / (self.out_max - self.out_min)
            x_num = x_num * (self.x_max - self.x_min) + self.x_min
            x_num = np.clip(x_num, self.x_min, self.x_max)

            if self.cat_idx is not None:
                x_cat_unencoded = self.encoder.inverse_transform(x_cat)
                x_reversed = np.zeros((x.shape[0], x_num.shape[1] + x_cat_unencoded.shape[1]))
            else:
                x_reversed = np.zeros((x.shape[0], x_num.shape[1]))

        if self.cat_idx is not None:
            x_reversed[:, self.cat_idx] = x_cat_unencoded
        x_reversed[:, self.num_idx] = x_num
        return x_reversed



