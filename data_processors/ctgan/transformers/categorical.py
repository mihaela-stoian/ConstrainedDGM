import warnings
import torch
import numpy as np
import pandas as pd
import psutil
from scipy.stats import norm

from data_processors.ctgan.transformers.base import BaseTransformer
from rdt.errors import TransformerInputError
import sys
sys.path.append('../ctgan')
from utils import round_func_BPDA





class OneHotEncoder(BaseTransformer):
    """OneHotEncoding for categorical data.
    This transformer replaces a single vector with N unique categories in it
    with N vectors which have 1s on the rows where the corresponding category
    is found and 0s on the rest.
    Null values are considered just another category.
    """

    INPUT_SDTYPE = 'categorical'
    SUPPORTED_SDTYPES = ['categorical', 'boolean']
    dummies = None
    _dummy_na = None
    _num_dummies = None
    _dummy_encoded = False
    _indexer = None
    _uniques = None

    @staticmethod
    def _prepare_data(data):
        """Transform data to appropriate format.
        If data is a valid list or a list of lists, transforms it into an np.array,
        otherwise returns it.
        Args:
            data (pandas.Series or pandas.DataFrame):
                Data to prepare.
        Returns:
            pandas.Series or numpy.ndarray
        """
        if isinstance(data, list):
            data = np.array(data)

        if len(data.shape) > 2:
            raise ValueError('Unexpected format.')
        if len(data.shape) == 2:
            if data.shape[1] != 1:
                raise ValueError('Unexpected format.')
            if isinstance(data, (pd.Series, pd.DataFrame)):
                data = data.iloc[:, 0]
            else:
                data = data[:, 0]

        return data

    def _fit(self, data):
        """Fit the transformer to the data.
        Get the pandas `dummies` which will be used later on for OneHotEncoding.
        Args:
            data (pandas.Series or pandas.DataFrame):
                Data to fit the transformer to.
        """
        data = self._prepare_data(data)

        null = pd.isna(data).to_numpy()
        self._uniques = list(pd.unique(data[~null]))
        self._uniques.sort()
        self._dummy_na = null.any()
        self._num_dummies = len(self._uniques)
        self._indexer = list(range(self._num_dummies))
        self.dummies = self._uniques.copy()
        self.torch_dummies = torch.tensor(self._uniques.copy(), dtype=torch.float32)

        if not np.issubdtype(data.dtype, np.number):
            self._dummy_encoded = True

        if self._dummy_na:
            self.dummies.append(np.nan)

        self.output_properties = {
            f'value{i}': {'sdtype': 'float', 'next_transformer': None}
            for i in range(len(self.dummies))
        }

    def _transform_helper(self, data):
    
        if isinstance(data, torch.Tensor):
            #if self._dummy_encoded:
                ##
                #coder = self._indexer
                #codes = torch.tensor(pd.Categorical(data, categories=self._uniques).codes)
            coder = torch.tensor(self._uniques, dtype=torch.float32)
            codes = data.clone()

            rows = data.shape[0]
            dummies = torch.broadcast_to(coder,(rows, self._num_dummies))
            coded = codes.repeat_interleave(self._num_dummies).reshape(rows, self._num_dummies)

            # diff = coded - dummies
            # diff[diff>0]= - diff[diff>0]
            # elu = torch.nn.ELU()
            # array_bef = elu(diff) + 0.55
            # array = round_func_BPDA(array_bef)
            diff_pos = (coded - dummies)**2
            diff_neg = 1 - diff_pos
            array = torch.maximum(diff_neg, torch.zeros_like(diff_neg))

            if self._dummy_na:
                null = torch.zeros((rows, 1), dtype=torch.int)
                null[torch.isnan(codes)] = 1
                array = torch.cat((array, null), dim=1)

        else: 
            if self._dummy_encoded:
                    coder = self._indexer
                    codes = pd.Categorical(data, categories=self._uniques).codes
            else:
                coder = self._uniques
                codes = data

            rows = len(data)
            dummies = np.broadcast_to(coder, (rows, self._num_dummies))
            coded = np.broadcast_to(codes, (self._num_dummies, rows)).T
            array = (coded == dummies).astype(int)
            if self._dummy_na:
                null = np.zeros((rows, 1), dtype=int)
                null[pd.isna(data)] = 1
                array = np.append(array, null, axis=1)
        return array

    def _transform(self, data, probs, column_name):
        """Replace each category with the OneHot vectors.
        Args:
            data (pandas.Series, list or list of lists):
                Data to transform.
        Returns:
            numpy.ndarray
        """
        ## Need to update this func so it is torch compatible
        if isinstance(data, torch.Tensor):
            data = self._prepare_data(data)
            unique_data = {torch.nan if torch.isnan(x) else x for x in torch.unique(data)}
            # unseen_categories = unique_data - set(self.dummies)
            # if unseen_categories:
            #     # Select only the first 5 unseen categories to avoid flooding the console.
            #     examples_unseen_categories = set(list(unseen_categories)[:5])
            #     warnings.warn(
            #         f'The data contains {len(unseen_categories)} new categories that were not '
            #         f'seen in the original data (examples: {examples_unseen_categories}). Creating '
            #         'a vector of all 0s. If you want to model new categories, '
            #         'please fit the transformer again with the new data.'
            #     )

        else:
            data = self._prepare_data(data)
            unique_data = {np.nan if pd.isna(x) else x for x in pd.unique(data)}
            unseen_categories = unique_data - set(self.dummies)
            if unseen_categories:
                # Select only the first 5 unseen categories to avoid flooding the console.
                examples_unseen_categories = set(list(unseen_categories)[:5])
                warnings.warn(
                    f'The data contains {len(unseen_categories)} new categories that were not '
                    f'seen in the original data (examples: {examples_unseen_categories}). Creating '
                    'a vector of all 0s. If you want to model new categories, '
                    'please fit the transformer again with the new data.'
                )
        return self._transform_helper(data)
    
    def softargmax(self, x, dim=-1):
    # crude: assumes max value is unique
    # Can cause rounding errors
        # beta = 100.0
        # xx = beta * x
        # sm = torch.nn.functional.softmax(xx, dim=dim)
        # indices = torch.arange(x.shape[dim])
        indices = self.torch_dummies.clone()
        y = torch.mul(indices, x)
        result = torch.sum(y, dim)
        return result

    def _reverse_transform(self, data):
        """Convert float values back to the original categorical values.
        Args:
            data (pd.Series or numpy.ndarray):
                Data to revert.
        Returns:
            pandas.Series
        """
        if isinstance(data, (pd.Series, pd.DataFrame)):
            data = data.to_numpy()

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if isinstance(data, torch.Tensor):
            inv = self.softargmax(data, 1)
            return inv
        else:
            indices = np.argmax(data, axis=1)
            return pd.Series(indices).map(self.dummies.__getitem__)

            



