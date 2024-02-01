"""Transformers for numerical data."""
import copy
import sys
import warnings
import torch
import numpy as np
import pandas as pd
import scipy
from sklearn.mixture import BayesianGaussianMixture
#from data_processors.ctgan.mixtures._bayesian_mixture import BayesianGaussianMixture
sys.path.append('../ctgan')

from data_processors.ctgan.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer
from utils import round_func_BPDA
import wandb

EPSILON = np.finfo(np.float32).eps
MAX_DECIMALS = sys.float_info.dig - 1
INTEGER_BOUNDS = {
    'Int8': (-2**7, 2**7 - 1),
    'Int16': (-2**15, 2**15 - 1),
    'Int32': (-2**31, 2**31 - 1),
    'Int64': (-2**63, 2**63 - 1),
    'UInt8': (0, 2**8 - 1),
    'UInt16': (0, 2**16 - 1),
    'UInt32': (0, 2**32 - 1),
    'UInt64': (0, 2**64 - 1),
}


class FloatFormatter(BaseTransformer):
    """Transformer for numerical data.
    This transformer replaces integer values with their float equivalent.
    Non null float values are not modified.
    Null values are replaced using a ``NullTransformer``.
    Args:
        missing_value_replacement (object):
            Indicate what to replace the null values with. If an integer or float is given,
            replace them with the given value. If the strings ``'mean'`` or ``'mode'``
            are given, replace them with the corresponding aggregation.
            Defaults to ``mean``.
        model_missing_values (bool):
            Whether to create a new column to indicate which values were null or not. The column
            will be created only if there are null values. If ``True``, create the new column if
            there are null values. If ``False``, do not create the new column even if there
            are null values. Defaults to ``False``.
        learn_rounding_scheme (bool):
            Whether or not to learn what place to round to based on the data seen during ``fit``.
            If ``True``, the data returned by ``reverse_transform`` will be rounded to that place.
            Defaults to ``False``.
        enforce_min_max_values (bool):
            Whether or not to clip the data returned by ``reverse_transform`` to the min and
            max values seen during ``fit``. Defaults to ``False``.
        computer_representation (dtype):
            Accepts ``'Int8'``, ``'Int16'``, ``'Int32'``, ``'Int64'``, ``'UInt8'``, ``'UInt16'``,
            ``'UInt32'``, ``'UInt64'``, ``'Float'``.
            Defaults to ``'Float'``.
    """

    INPUT_SDTYPE = 'numerical'
    null_transformer = None
    missing_value_replacement = None
    _dtype = None
    _rounding_digits = None
    _min_value = None
    _max_value = None

    def __init__(self, missing_value_replacement='mean', model_missing_values=False,
                 learn_rounding_scheme=False, enforce_min_max_values=False,
                 computer_representation='Float'):
        super().__init__()
        self._set_missing_value_replacement('mean', missing_value_replacement)
        self.model_missing_values = model_missing_values
        self.learn_rounding_scheme = learn_rounding_scheme
        self.enforce_min_max_values = enforce_min_max_values
        self.computer_representation = computer_representation

    @staticmethod
    def _learn_rounding_digits(data):
        # check if data has any decimals
        name = data.name
        data = np.array(data)
        roundable_data = data[~(np.isinf(data) | pd.isna(data))]

        # Doesn't contain numbers
        if len(roundable_data) == 0:
            return None

        # Doesn't contain decimal digits
        if ((roundable_data % 1) == 0).all():
            return 0

        # Try to round to fewer digits
        if (roundable_data == roundable_data.round(MAX_DECIMALS)).all():
            for decimal in range(MAX_DECIMALS + 1):
                if (roundable_data == roundable_data.round(decimal)).all():
                    return decimal

        # Can't round, not equal after MAX_DECIMALS digits of precision
        warnings.warn(
            f"No rounding scheme detected for column '{name}'. Data will not be rounded.")
        return None

    def _raise_out_of_bounds_error(self, value, name, bound_type, min_bound, max_bound):
        raise ValueError(
            f"The {bound_type} value in column '{name}' is {value}."
            f" All values represented by '{self.computer_representation}'"
            f' must be in the range [{min_bound}, {max_bound}].'
        )


    def _validate_values_within_bounds(self, data, name=None):

        if isinstance(data, torch.Tensor):
            if self.computer_representation != 'Float':
                            fractions = data[~torch.isnan(data) & data % 1 != 0]
                            if fractions.shape[0] > 0:
                                raise ValueError(
                                    f"The column '{name}' contains float values {fractions.tolist()}. "
                                    f"All values represented by '{self.computer_representation}' must be integers."
                                )

                            min_value = torch.min(data)
                            max_value = torch.max(data)
                            min_bound, max_bound = INTEGER_BOUNDS[self.computer_representation]
                            if min_value < min_bound:
                                self._raise_out_of_bounds_error(
                                    min_value, name, 'minimum', min_bound, max_bound)

                            if max_value > max_bound:
                                self._raise_out_of_bounds_error(
                                    max_value, name, 'maximum', min_bound, max_bound)

        else: 

            if self.computer_representation != 'Float':
                fractions = data[~np.isnan(data) & data % 1 != 0]
                if fractions.shape[0] > 0:
                    raise ValueError(
                        f"The column '{name}' contains float values {fractions.tolist()}. "
                        f"All values represented by '{self.computer_representation}' must be integers."
                    )

                min_value = np.min(data)
                max_value = np.max(data)
                min_bound, max_bound = INTEGER_BOUNDS[self.computer_representation]
                if min_value < min_bound:
                    self._raise_out_of_bounds_error(
                        min_value, name, 'minimum', min_bound, max_bound)

                if max_value > max_bound:
                    self._raise_out_of_bounds_error(
                        max_value, name, 'maximum', min_bound, max_bound)
    

    def _fit(self, data):
        """Fit the transformer to the data.
        Args:
            data (pandas.Series):
                Data to fit.
        """
        self._validate_values_within_bounds(data)
        self._dtype = data.dtype

        if self.enforce_min_max_values:
            self._min_value = data.min()
            self._max_value = data.max()

        if self.learn_rounding_scheme:
            self._rounding_digits = self._learn_rounding_digits(data)

        self.null_transformer = NullTransformer(
            self.missing_value_replacement,
            self.model_missing_values
        )
        self.null_transformer.fit(data)
        if self.null_transformer.models_missing_values():
            self.output_properties['is_null'] = {'sdtype': 'float', 'next_transformer': None}

    def _transform(self, data,  probs=None, column_name=None):
        """Transform numerical data.
        Integer values are replaced by their float equivalent. Non null float values
        are left unmodified.
        Args:
            data (pandas.Series):
                Data to transform.
        Returns:
            numpy.ndarray
        """
        self._validate_values_within_bounds(data, column_name)

        #For the moment I am commenting this. Need to handle it later
        # self.null_transformer.transform(data)
        if isinstance (data, torch.Tensor):
            # data = torch.squeeze(data)
            pass

        elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
            data = np.squeeze(data.to_numpy())

        return data

    def _reverse_transform(self, data):
        """Convert data back into the original format.
        Args:
            data (pd.Series or numpy.ndarray):
                Data to transform.
        Returns:
            numpy.ndarray
        """
        #if not isinstance(data, np.ndarray):
            #data = data.to_numpy()

        #data = self.null_transformer.reverse_transform(data)
        if self.enforce_min_max_values:
            if isinstance(data, torch.Tensor):
                #data_bef = data.clone()
                data = data.clamp(self._min_value, self._max_value)
                #diff = np.count_nonzero((data-data_bef)!=0)/data.shape[0]
                #wandb.log({'Clip_diff/enforce_min_max': diff})
            else:
                data = data.clip(self._min_value, self._max_value)
        elif self.computer_representation != 'Float':
            min_bound, max_bound = INTEGER_BOUNDS[self.computer_representation]
            data = data.clip(min_bound, max_bound)

        is_integer = np.dtype(self._dtype).kind == 'i'

        ## Currently not entering this clause
        if self.learn_rounding_scheme and self._rounding_digits is not None:
            data = data.round(self._rounding_digits)
        elif is_integer:
            if isinstance(data, torch.Tensor):
                pass
                #data = round_func_BPDA(data)
            else:
                data = data.round(0)

        #if pd.isna(data).any() and is_integer:
            #return data

        #return data.astype(self._dtype)
        return data




class ClusterBasedNormalizer(FloatFormatter):
    """Transformer for numerical data using a Bayesian Gaussian Mixture Model.
    This transformation takes a numerical value and transforms it using a Bayesian GMM
    model. It generates two outputs, a discrete value which indicates the selected
    'component' of the GMM and a continuous value which represents the normalized value
    based on the mean and std of the selected component.
    Args:
        model_missing_values (bool):
            Whether to create a new column to indicate which values were null or not. The column
            will be created only if there are null values. If ``True``, create the new column if
            there are null values. If ``False``, do not create the new column even if there
            are null values. Defaults to ``False``.
        learn_rounding_scheme (bool):
            Whether or not to learn what place to round to based on the data seen during ``fit``.
            If ``True``, the data returned by ``reverse_transform`` will be rounded to that place.
            Defaults to ``False``.
        enforce_min_max_values (bool):
            Whether or not to clip the data returned by ``reverse_transform`` to the min and
            max values seen during ``fit``. Defaults to ``False``.
        max_clusters (int):
            The maximum number of mixture components. Depending on the data, the model may select
            fewer components (based on the ``weight_threshold``).
            Defaults to 10.
        weight_threshold (int, float):
            The minimum value a component weight can take to be considered a valid component.
            ``weights_`` under this value will be ignored.
            Defaults to 0.005.
    Attributes:
        _bgm_transformer:
            An instance of sklearn`s ``BayesianGaussianMixture`` class.
        valid_component_indicator:
            An array indicating the valid components. If the weight of a component is greater
            than the ``weight_threshold``, it's indicated with True, otherwise it's set to False.
    """

    STD_MULTIPLIER = 4
    _bgm_transformer = None
    valid_component_indicator = None

    def __init__(self, model_missing_values=False, learn_rounding_scheme=False,
                 enforce_min_max_values=False, max_clusters=10, weight_threshold=0.005):
        super().__init__(
            model_missing_values=model_missing_values,
            learn_rounding_scheme=learn_rounding_scheme,
            enforce_min_max_values=enforce_min_max_values
        )
        self.max_clusters = max_clusters
        self.weight_threshold = weight_threshold
        self.output_properties = {
            'normalized': {'sdtype': 'float', 'next_transformer': None},
            'component': {'sdtype': 'categorical', 'next_transformer': None},
        }
    def _get_seeds(self):
        return self.random_states["transform"].get_state()

    def _get_current_random_seed(self):
        if self.random_states:
            return self.random_states['fit'].get_state()[1][0]
        return 0

    def _fit(self, data):
        """Fit the transformer to the data.
        Args:
            data (pandas.Series):
                Data to fit to.
        """
        self._bgm_transformer = BayesianGaussianMixture(
            n_components=self.max_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            n_init=1,
            random_state=self._get_current_random_seed()
        )

        super()._fit(data)
        data = super()._transform(data, column_name=None)

        if data.ndim > 1:
            data = data[:, 0]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._bgm_transformer.fit(data.reshape(-1, 1))

        self.valid_component_indicator = self._bgm_transformer.weights_ > self.weight_threshold


    def _transform(self, data, probs, column_name):
        """Transform the numerical data.
        Args:
            data (pandas.Series):
                Data to transform.
        Returns:
            numpy.ndarray.
        """

        data = super()._transform(data, probs, column_name)
        if data.ndim > 1:
            data, model_missing_values = data[:, 0], data[:, 1]
        data = data.reshape((len(data), 1))
        means = self._bgm_transformer.means_.reshape((1, self.max_clusters))
        std_multiplier = torch.tensor(self.STD_MULTIPLIER, dtype=torch.float32)
        if isinstance(data, torch.Tensor):
            probs_detach = probs.clone().detach()
            stds_valid = np.sqrt(self._bgm_transformer.covariances_).reshape((1, self.max_clusters))[:,self.valid_component_indicator]
            means_valid = means[:,self.valid_component_indicator]
            means = torch.tensor(means_valid, dtype=torch.float32)
            stds = torch.tensor(stds_valid, dtype=torch.float32)
            normalized_values = (data - means) / (std_multiplier * stds)
            masked_normalized = torch.sum(normalized_values*probs_detach, dim=1).reshape([-1, 1])
            masked_normalized = torch.clamp(masked_normalized, -.99, .99)

            #masked_normalized = torch.clamp(masked_normalized_bef, -.99, .99)
            #diff = np.count_nonzero((masked_normalized-masked_normalized_bef)!=0)/masked_normalized.shape[0]
            #wandb.log({'Clip_diff/transform': diff})

            if self.null_transformer and self.null_transformer.models_missing_values():
                rows.append(model_missing_values)

            masked_output = torch.cat((masked_normalized, probs),dim=1)
            return masked_output



        else:
            stds = np.sqrt(self._bgm_transformer.covariances_).reshape((1, self.max_clusters))
            normalized_values = (data - means) / (self.STD_MULTIPLIER * stds)
            normalized_values = normalized_values[:, self.valid_component_indicator]
            component_probs = self._bgm_transformer.predict_proba(data)
            component_probs = component_probs[:, self.valid_component_indicator]

            selected_component = np.zeros(len(data), dtype='int')
            for i in range(len(data)):
                component_prob_t = component_probs[i] + 1e-6
                component_prob_t = component_prob_t / component_prob_t.sum()
                selected_component[i] = np.random.choice(
                    np.arange(self.valid_component_indicator.sum()),
                    p=component_prob_t
                )

            aranged = np.arange(len(data))
            normalized = normalized_values[aranged, selected_component].reshape([-1, 1])
            normalized = np.clip(normalized, -.99, .99)
            normalized = normalized[:, 0]
            rows = [normalized, selected_component]
            if self.null_transformer and self.null_transformer.models_missing_values():
                rows.append(model_missing_values)
            output = np.stack(rows, axis=1) 
        return output  # noqa: PD013


    def _reverse_transform_helper(self, data):
        if isinstance(data, torch.Tensor):
            normalized = torch.clip(data[:, 0], -1.0, 1.0)
            #diff = np.count_nonzero((normalized-data[:, 0])!=0)/data.shape[0]
            #wandb.log({'Clip_diff/inverse': diff})

            std_multiplier = torch.tensor(self.STD_MULTIPLIER, dtype=torch.float32)
            selected_component = data[:, 1:].clone().detach()

            means_valid = self._bgm_transformer.means_.reshape([-1])[self.valid_component_indicator]
            stds_valid = np.sqrt(self._bgm_transformer.covariances_.reshape([-1])[self.valid_component_indicator])
            means = torch.tensor(means_valid, dtype=torch.float32)
            stds = torch.tensor(stds_valid, dtype=torch.float32)
            std_t = torch.sum(stds*selected_component, dim=1)
            mean_t = torch.sum(means*selected_component, dim=1)
            reversed_data = normalized * std_multiplier * std_t + mean_t

        else:
            if not isinstance(data, np.ndarray):
                data = data.to_numpy()
            normalized = np.clip(data[:, 0], -1, 1)
            means = self._bgm_transformer.means_.reshape([-1])
            stds = np.sqrt(self._bgm_transformer.covariances_).reshape([-1])
            selected_component = data[:, 1].astype(int)
            selected_component = selected_component.clip(0, self.valid_component_indicator.sum() - 1)
            std_t = stds[self.valid_component_indicator][selected_component]
            mean_t = means[self.valid_component_indicator][selected_component]
            reversed_data = normalized * self.STD_MULTIPLIER* std_t + mean_t

        return reversed_data

    def _reverse_transform(self, data):
        """Convert data back into the original format.
        Args:
            data (pd.DataFrame or numpy.ndarray):
                Data to transform.
        Returns:
            pandas.Series.
        """

        data = self._reverse_transform_helper(data)
        if self.null_transformer and self.null_transformer.models_missing_values():
            if isinstance(data, torch.Tensor):
                data = torch.stack([data, data[:, -1]], axis=1)  # noqa: PD013
            else:
                data = np.stack([data, data[:, -1]], axis=1)  # noqa: PD013

        return super()._reverse_transform(data)