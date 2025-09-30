import numpy as np  


class BaseSynthesizer:
    """Base class for all default synthesizers of ``octgan``."""
    def __init__(self, dataset_name, args):
        self.dataset_name = dataset_name
        self.args = args
        
    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        pass

    def sample(self, samples):
        pass

    def fit_sample(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        
        self.fit(data, categorical_columns, ordinal_columns)
        num = data.shape[0]

        return self.sample(num)