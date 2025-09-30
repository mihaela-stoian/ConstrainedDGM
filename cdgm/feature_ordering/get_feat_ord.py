import json
import warnings
from pathlib import Path

import numpy as np
import torch

from constraints_code.feature_orderings import set_corr_ordering, set_kde_ordering, set_wasserstein_ordering, \
    set_jsd_ordering, set_causal_ordering

warnings.filterwarnings(action='ignore')
torch.set_printoptions(sci_mode=False)
np.set_printoptions(precision=3, suppress=True)
from argparse import ArgumentParser


def get_args():
    args = ArgumentParser()
    args.add_argument("use_case", type=str)
    args.add_argument("model_type", type=str, choices=['wgan', 'tablegan', 'ctgan', 'octgan', "tvae", "goggle"])
    args.add_argument("metric", type=str, choices=['kde', 'corr', 'wasserstein', 'jsd'])
    args.add_argument("--real_data_partition", default='test', type=str)
    return args.parse_args()


def main():
    args = get_args()
    label_ordering_choice = args.metric
    use_case = args.use_case
    model_type = args.model_type

    if label_ordering_choice == 'corr':
        ordering = set_corr_ordering(use_case, model_type)
    elif label_ordering_choice == 'kde':
        ordering = set_kde_ordering(use_case, model_type)
    elif label_ordering_choice == 'wasserstein':
        ordering = set_wasserstein_ordering(use_case, model_type)
    elif label_ordering_choice == 'jsd':
        ordering = set_jsd_ordering(use_case, model_type)
    elif label_ordering_choice == 'causal':
        ordering = set_causal_ordering(use_case, model_type)
    else:
        raise Exception(f'{label_ordering_choice} is not an option!')
    ordering = ordering[0]
    print(use_case, model_type, label_ordering_choice, ordering)

    json_filename = f'feature_ordering/feature_orderings.json'
    json_filename = Path(json_filename)
    if json_filename.exists():
        with open(json_filename, "r") as f:
            data = json.load(f)
        if len(data) == 0:
            data = {}
            data["feature_orderings"] = {}
    else:
        data = {}
        data["feature_orderings"] = {}

    if args.use_case in data["feature_orderings"]:
        pass
    else:
        data["feature_orderings"][args.use_case] = {}

    if args.model_type in data["feature_orderings"][args.use_case]:
        pass
    else:
        data["feature_orderings"][args.use_case][args.model_type] = {}

    if args.metric in data["feature_orderings"][args.use_case][args.model_type]:
        pass
    else:
        data["feature_orderings"][args.use_case][args.model_type][args.metric] = {}

    if args.real_data_partition in data["feature_orderings"][args.use_case][args.model_type][args.metric]:
        pass
    else:
        data["feature_orderings"][args.use_case][args.model_type][args.metric][args.real_data_partition] = ""

    data["feature_orderings"][args.use_case][args.model_type][args.metric][args.real_data_partition] = ordering
    with open(json_filename, "w") as f:
        json.dump(data, f, indent=2)
    # print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()