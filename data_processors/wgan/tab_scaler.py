from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder

from data_processors.wgan.typing import NDNumber
from data_processors.wgan.utils import round_func_BPDA, to_numpy_number
import torch.nn as nn


def get_num_idx(x_type: Union[pd.Series, npt.NDArray[np.str_]]) -> List[int]:
    return np.where(x_type != "cat")[0].tolist()


def get_cat_idx(x_type: Union[pd.Series, npt.NDArray[np.str_]]) -> List[int]:
    return np.where(x_type == "cat")[0].tolist()


def copy_x(
    x: Union[npt.NDArray[Any], torch.Tensor]
) -> Union[npt.NDArray[Any], torch.Tensor]:
    if isinstance(x, np.ndarray):
        return x.copy()
    if isinstance(x, torch.Tensor):
        return torch.clone(x)
    raise NotImplementedError


def softargmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # crude: assumes max value is unique
    # Can cause rounding errors
    # beta = 100.0
    # xx = beta * x
    # sm = torch.nn.functional.softmax(xx, dim=dim)
    indices = torch.arange(x.shape[dim])
    y = torch.mul(indices, x)
    result = torch.sum(y, dim)

    return result


def _handle_zeros_in_scale_torch(
    scale: torch.Tensor,
    copy: bool = True,
    constant_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Set scales of near constant features to 1.
    The goal is to avoid division by very small or zero values.
    Near constant features are detected automatically by identifying
    scales close to machine precision unless they are precomputed by
    the caller and passed with the `constant_mask` kwarg.
    Typically for standard scaling, the scales are the standard
    deviation while near constant features are better detected on the
    computed variances which are closer to machine precision by
    construction.
    """
    # if we are fitting on 1D arrays, scale might be a scalar
    if scale.dim() == 0:
        if scale == 0.0:
            scale = torch.tensor(1.0)
        return scale
    else:
        if constant_mask is None:
            # Detect near constant values to avoid dividing by a very small
            # value that could lead to surprising results and numerical
            # stability issues.
            constant_mask = scale < 10 * torch.finfo(scale.dtype).eps

        if copy:
            # New array to avoid side-effects
            scale = scale.clone()
        scale[constant_mask] = 1.0
        return scale


def process_cat_idx_params(
    x: torch.Tensor,
    cat_idx: Optional[List[int]] = None,
    x_type: Optional[Union[pd.Series, npt.NDArray[np.str_]]] = None,
) -> Tuple[List[int], List[int]]:
    if cat_idx is None:
        if x_type is not None:
            cat_idx = get_cat_idx(x_type)

    nb_features = x.shape[1]
    num_idx = [*range(nb_features)]

    if cat_idx is None:
        cat_idx = []
    else:
        num_idx = [e for e in num_idx if e not in cat_idx]

    return num_idx, cat_idx

### If cat_idx and one_hot_encode it will skip the processing of categorical
class TabScaler:
    def __init__(
        self,
        num_scaler: str = "min_max",
        one_hot_encode: bool = True,
        out_min: float = 0.0,
        out_max: float = 1.0,
    ) -> None: 

        # Params
        self.num_scaler = num_scaler
        self.one_hot_encode = one_hot_encode
        self.out_min = out_min
        self.out_max = out_max

        # Internal
        self.fitted = False
        self.x_min: Optional[torch.Tensor] = None
        self.x_max: Optional[torch.Tensor] = None
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.categories: List[int] = []
        self.cat_idx: List[int] = []
        self.num_idx: List[int] = []

        # Check params
        if self.out_min >= self.out_max:
            raise ValueError("out_min must be smaller than out_max")

    def fit(
        self,
        x: Union[torch.Tensor, NDNumber],
        cat_idx: Optional[List[int]] = None,
        x_type: Union[pd.Series, npt.NDArray[np.str_]] = None,
    ) -> TabScaler:

        if isinstance(x, np.ndarray):
            return self.fit(torch.Tensor(x), cat_idx, x_type)

        # Process feature types
        self.num_idx, self.cat_idx = process_cat_idx_params(x, cat_idx, x_type)

        # Numerical features
        if self.num_scaler == "min_max":
            self.x_min = torch.min(x[:, self.num_idx], dim=0)[0]
            self.x_max = torch.max(x[:, self.num_idx], dim=0)[0]
            if self.x_min is None or self.x_max is None:
                raise ValueError("No numerical features to scale")
            self.min_max_scale = _handle_zeros_in_scale_torch(
                self.x_max - self.x_min
            )

        elif self.num_scaler == "standard":
            self.mean = torch.mean(x[:, self.num_idx], dim=0)
            self.std = torch.std(x[:, self.num_idx], dim=0)

        elif self.num_scaler == "none":
            pass

        else:
            raise NotImplementedError

        self.one_hot_encode = self.one_hot_encode and len(self.cat_idx) > 0

        # Categorical features
        if self.one_hot_encode:
            self.ohe = OneHotEncoder(sparse=False)
            self.ohe.fit(x.numpy()[:, self.cat_idx])

        self.fitted = True

        return self

    def transform(
        self, x_in: Union[torch.Tensor, NDNumber]
    ) -> Union[torch.Tensor, NDNumber]:

        if not self.fitted:
            raise ValueError("Must fit scaler before transforming data")

        if isinstance(x_in, np.ndarray):
            return to_numpy_number(self.transform(torch.Tensor(x_in)))

        # Numerical features
        x = x_in.clone()

        if self.num_scaler == "min_max":
            x[:, self.num_idx] = (
                (x[:, self.num_idx].clone() - self.x_min) / self.min_max_scale
            ) * (self.out_max - self.out_min) + self.out_min

        elif self.num_scaler == "standard":
            x[:, self.num_idx] = (x[:, self.num_idx] - self.mean) / self.std

        elif self.num_scaler == "none":
            pass

        else:
            raise NotImplementedError

        if self.one_hot_encode:

            list_encoded = []

            for i, idx in enumerate(self.cat_idx):

                categories = self.ohe.categories_[i]
                num_categories = len(categories)

                coder = torch.tensor(categories, dtype=torch.float32)
                codes = x[:, idx].clone()

                rows = x.shape[0]

                dummies = torch.broadcast_to(coder, (rows, num_categories))
                coded = codes.repeat_interleave(num_categories).reshape(
                    rows, num_categories
                )

                # diff = coded - dummies
                # diff[diff > 0] = -diff[diff > 0]
                # elu = torch.nn.ELU()
                # array_bef = elu(diff) + 0.55
                # encoded = round_func_BPDA(array_bef)
                # list_encoded.append(encoded)
                diff_pos = (coded - dummies)**2
                diff_neg = 1 - diff_pos
                encoded = torch.maximum(diff_neg, torch.zeros_like(diff_neg))
                list_encoded.append(encoded)
                
            x = torch.concat([x[:, self.num_idx]] + list_encoded, dim=1)

        return x

    def inverse_transform(
        self, x_in: Union[torch.Tensor, NDNumber]
    ) -> Union[torch.Tensor, NDNumber]:

        if not self.fitted:
            raise ValueError("Must fit scaler before transforming data")

        if isinstance(x_in, np.ndarray):
            return to_numpy_number(self.inverse_transform(torch.Tensor(x_in)))

        x = x_in.clone()

        # Categorical features
        if self.one_hot_encode:

            start_idx = len(self.num_idx)

            decoded_list = []
            for i, categories in enumerate(self.ohe.categories_):
                num_categories = len(categories)
                end_idx = start_idx + num_categories

                decoded = softargmax(x[:, start_idx:end_idx], 1)
                decoded_list.append(decoded)
                start_idx = end_idx

            out: List[torch.Tensor] = []

            # First numerical features
            out.append(x[:, : self.cat_idx[0]])
            last_num_used = self.cat_idx[0]

            for i, idx in enumerate(self.cat_idx):
                out.append(decoded_list[i].reshape(-1, 1))
                if i < len(self.cat_idx) - 1:
                    num_to_add = self.cat_idx[i + 1] - idx - 1
                else:
                    num_to_add = len(self.num_idx) - last_num_used
                end_idx = last_num_used + num_to_add
                out.append(x[:, last_num_used:end_idx])
                last_num_used = end_idx

            x = torch.concat(out, dim=1)

        # Numerical features
        if self.num_scaler == "min_max":
            x[:, self.num_idx] = (x[:, self.num_idx] - self.out_min) / (
                self.out_max - self.out_min
            )
            x[:, self.num_idx] = (
                x[:, self.num_idx] * self.min_max_scale + self.x_min
            )

        elif self.num_scaler == "standard":
            x[:, self.num_idx] = x[:, self.num_idx] * self.std + self.mean

        elif self.num_scaler == "none":
            pass

        else:
            raise NotImplementedError

        return x

    def get_transorm_nn(self):
        return Transform(self)

    def transform_mask(self, mask: torch.Tensor) -> torch.Tensor:
        mask_out = [mask[self.num_idx]]
        for i, e in self.cat_idx:
            mask_out.append(mask[e].repeat(len(self.ohe.categories_[i])))
        return torch.cat(mask_out, axis=0)


class Transform(nn.Module):
    def __init__(self, scaler: TabScaler):
        self.scaler = scaler
        super().__init__()

    def forward(self, x):
        return self.scaler.transform(x)
