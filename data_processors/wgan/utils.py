import json
import pathlib
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
from data_processors.wgan.typing import NDNumber


def cut_in_batch(
    arr: npt.NDArray[Any],
    n_desired_batch: int = 1,
    batch_size: Optional[int] = None,
) -> List[npt.NDArray[Any]]:

    if batch_size is None:
        n_batch = min(n_desired_batch, len(arr))
    else:
        n_batch = np.ceil(len(arr) / batch_size)
    batches_i = np.array_split(np.arange(arr.shape[0]), n_batch)

    return [arr[batch_i] for batch_i in batches_i]


def dict2obj(d: Dict[str, Any]) -> Any:
    dumped_data = json.dumps(d)
    result = json.loads(
        dumped_data, object_hook=lambda x: SimpleNamespace(**x)
    )
    return result


def parent_exists(path: str) -> str:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def round_func_BPDA(input):
    # This is equivalent to replacing round function (non-differentiable) with
    # an identity function (differentiable) only when backward.
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out


def to_numpy_number(tensor: Union[torch.Tensor, NDNumber]) -> NDNumber:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()

    elif isinstance(tensor, np.ndarray):
        return tensor

    raise NotImplementedError(f"Unsupported type: {type(tensor)}")


