import numpy as np
import torch 



class CustomTransformer():
    def __init__(self, means: np.ndarray, stds: np.ndarray, enforce_min_max_values=False, min_value=None, max_value=None, regression=False):
        self.regression = regression
        self.enforce_min_max_values = enforce_min_max_values
        self._min_value = min_value
        self._max_value = max_value
        self.set_means_and_stds(means, stds)

    def set_means_and_stds(self, means, stds):
        if self.regression:
            self.means = torch.tensor(means, dtype=torch.float32)
            self.stds = torch.tensor(stds, dtype=torch.float32)
        else:
            self.means = torch.cat([torch.tensor(means, dtype=torch.float32),
                                    torch.tensor([0.], dtype=torch.float32)])  # add 0 at the end for the target class
            self.stds = torch.cat([torch.tensor(stds, dtype=torch.float32),
                                   torch.tensor([1.], dtype=torch.float32)])  # add 1 at the end for the target class

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        transformed_x = (x - self.means) / self.stds
        return transformed_x

    def inverse_transform(self, z: torch.Tensor, inference_time_enforce_min_max_values=False) -> torch.Tensor:
        inverse_transformed_z = z * self.stds + self.means
        if self.enforce_min_max_values or inference_time_enforce_min_max_values:
            inverse_transformed_z = inverse_transformed_z.clamp(self._min_value, self._max_value)
        return inverse_transformed_z