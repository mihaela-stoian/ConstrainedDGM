import json
import pickle as pkl
import warnings

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, wasserstein_distance
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


from synthetizers.CTGAN.ctgan import CTGAN
from utils import read_csv, _load_json

warnings.filterwarnings(action='ignore')
torch.set_printoptions(sci_mode=False)
np.set_printoptions(precision=3, suppress=True)
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_args():
    args = ArgumentParser()
    args.add_argument("use_case", type=str)
    args.add_argument("metric", type=str, choices=['kde', 'corr', 'wasserstein', 'jsd'])
    args.add_argument("real_data_partition", type=str)
    args.add_argument("model_type", type=str, choices=['wgan', 'tablegan', 'ctgan', 'octgan', "tvae", "goggle"])
    # args.add_argument("gen_data_path", type=str)
    args.add_argument("--gen_data_path", default=None, type=str)
    args.add_argument("--order", default="ascending", type=str, choices=['ascending', 'descending'])
    args.add_argument("--model_dir", default="plots_data", type=str)
    args.add_argument("--num_bins", default=200, type=int, help='only needed for kl metric')
    args.add_argument("--num_sampling_rounds", default=5, type=int)
    args.add_argument("--wasserstein_load", action="store_true", help="load pre-computed Wasserstein distances")
    args.add_argument("--quick", action="store_true", help="Analyse only the val dataset; useful for very large datasets")
    return args.parse_args()


def get_ranking(args, scores):
    print(f'\nRanking features by {args.metric} metric, in {args.order} order')
    ranked = np.array(scores).argsort()
    if args.order == 'descending':
        ranked = ranked[::-1]

    computed_ordering = []
    for i, feat_id in enumerate(ranked):
        print(f'Rank {i}: y_{feat_id} (column {args.feats[feat_id]}) with {args.metric} score {scores[feat_id]}') # TODO!!!!
        computed_ordering.append(f'y_{feat_id}')

    ordering = '"' + " ".join(computed_ordering)[:-1] + '"'
    print(f'\n ordering \n{ordering}')
    return computed_ordering


def normalise_probs(probs_dict):
    summed = np.array(list(probs_dict.values())).sum()
    for key in probs_dict:
        probs_dict[key] = probs_dict[key]/summed
    return probs_dict


def get_kl_ordering_botnet(args, real_data_, gen_data_, eps=1e-12):
    all_feats = real_data_.columns

    if args.model_type=='tablegan':
        if args.use_case == 'news':
            num_perm = 5
            num_partitions = 5
    elif args.model_type=='goggle':
        if args.use_case in ['url']:
            num_perm = 5
            num_partitions = 3
        elif args.use_case == 'faults':
            num_perm = 5
            num_partitions = 5
    elif args.model_type == 'tvae':
        if args.use_case == 'lcld':
            num_perm = 5
            num_partitions = 3
        if args.use_case == 'faults':
            num_perm = 5
            num_partitions = 5
    elif args.model_type!='octgan':
        if args.use_case == 'botnet':
            num_perm = 5
            num_partitions = 100
        elif args.use_case == 'faults':
            num_perm = 5
            num_partitions = 5
    else:
        if args.use_case in ['url']:
            num_perm = 5
            num_partitions = 3
        elif args.use_case == 'faults':
            num_perm = 5
            num_partitions = 5

    #     real_nan_cols = get_real_nan_columns(args, real_data_)
    #     real_data_ = real_data_.drop(columns=real_nan_cols)
    #     gen_data_ = gen_data_.drop(columns=real_nan_cols)

    kls = {feat:[] for feat in all_feats}
    partitions = list(map(int, np.linspace(0, len(all_feats), num_partitions).round()))
    print('Partitions:', partitions)

    for ii in range(num_perm):

        feats_perm = np.random.permutation(range(len(all_feats)))
        permuted_feats = all_feats[feats_perm]
        perm_real_data = real_data_[all_feats[feats_perm]].to_numpy()
        perm_gen_data = gen_data_[all_feats[feats_perm]].to_numpy()

        for partition_id in range(len(partitions[:-1])):
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@', partition_id, partitions[partition_id])
            real_data = perm_real_data.copy()[:,partitions[partition_id]:partitions[partition_id+1]]
            gen_data = perm_gen_data.copy()[:,partitions[partition_id]:partitions[partition_id+1]]
            feats = permuted_feats.copy()[partitions[partition_id]:partitions[partition_id+1]]
            print(feats)
            print('Using real and gen. data shapes and len feats:', real_data.shape, gen_data.shape, len(feats))

            # real_pdf = gaussian_kde(np.transpose(real_data))
            # gen_pdf = gaussian_kde(np.transpose(gen_data))

            real_model = KernelDensity(kernel='gaussian', bandwidth=1.).fit(real_data)
            print('Got KDE model')
            real_probabilities = real_model.score_samples(real_data)
            print('Scored real probs)')
            gen_probabilities = real_model.score_samples(gen_data)
            print('Scored gen probs)')

            real_probabilities = np.exp(real_probabilities)
            real_probabilities /= sum(real_probabilities)
            gen_probabilities = np.exp(gen_probabilities)
            gen_probabilities /= sum(gen_probabilities)

            print(gen_probabilities)
            marked= False
            for i,feat in tqdm(enumerate(feats), total=len(feats)):
                real_freq_x = real_data_[feat]
                real_unique_x = real_freq_x.unique()
                gen_freq_x = gen_data_[feat]
                gen_unique_x = gen_freq_x.unique()

                values_x = sorted(list(set(list(real_unique_x) + list(gen_unique_x))))
                # if len(gen_unique_x)/len(real_unique_x) < 0.5 or len(real_unique_x) < 0.5:
                #     print(f'Kept {len(values_x)} from {len(real_unique_x)} {len(gen_unique_x)}, feat {feat}')
                #     marked = True
                #     values_x = list(set(real_unique_x))
                # if len(real_unique_x)/len(gen_unique_x) < 0.5:
                #     print(f'Kept {len(values_x)} from {len(real_unique_x)} {len(gen_unique_x)}, feat {feat}')
                #     marked = True
                #     values_x = list(set(gen_unique_x))

                # values_x = sorted(real_unique_x)
                real_marginalised_x = {}
                for val in values_x:
                    if val not in real_unique_x:
                        real_marginalised_x[val] = 0.
                        continue
                    mask = real_freq_x == val
                    positions = real_freq_x[mask].index
                    real_marginalised_x[val] = sum(real_probabilities[positions])
                # real_marginalised_x = np.array(list(real_marginalised_x.values()))
                # real_marginalised_x /= real_marginalised_x.sum()
                real_marginalised_x_dict = normalise_probs(real_marginalised_x)

                # values_x = sorted(gen_unique_x)
                gen_marginalised_x = {}
                for val in values_x:
                    if val not in gen_unique_x:
                        gen_marginalised_x[val] = 0.
                        continue
                    mask = gen_freq_x == val
                    positions = gen_freq_x[mask].index
                    gen_marginalised_x[val] = sum(gen_probabilities[positions]) + eps
                # gen_marginalised_x = np.array(list(gen_marginalised_x.values()))
                # gen_marginalised_x /= gen_marginalised_x.sum()
                gen_marginalised_x_dict = normalise_probs(gen_marginalised_x)

                # for key in real_marginalised_x_dict:
                #     if key not in gen_marginalised_x_dict:
                #         gen_marginalised_x_dict[key] = 0.
                # for key in gen_marginalised_x_dict:
                #     if key not in real_marginalised_x_dict:
                #         real_marginalised_x_dict[key] = 0.

                real_marginalised_x = np.array(list(real_marginalised_x_dict.values()))
                gen_marginalised_x = np.array(list(gen_marginalised_x_dict.values()))



                # min_bound = (real_marginalised_x.min() + gen_marginalised_x.min())/2.
                # max_bound = (real_marginalised_x.max() + gen_marginalised_x.max())/2.
                #
                # min_bound = real_marginalised_x.min()
                # max_bound = real_marginalised_x.max()

                # min_bound = min(real_marginalised_x.min(), gen_marginalised_x.min())
                # max_bound = max(real_marginalised_x.max(), gen_marginalised_x.max())
                # range_bounds = (min_bound, max_bound)

                # # num_bins = args.num_bins
                # # num_bins = min(len(real_marginalised_x), len(gen_marginalised_x), 200)
                # num_bins = min(max(30,len(real_unique_x),len(gen_unique_x)), args.num_bins)
                # # if num_bins < 100:
                # #     print(num_bins)
                # real_marginalised_x_hist = np.histogram(list(real_marginalised_x_dict.values()), bins=num_bins, range=range_bounds)
                # gen_marginalised_x_hist = np.histogram(list(gen_marginalised_x_dict.values()), bins=num_bins, range=range_bounds)

                # real_marginalised_x_hist = np.histogram(real_marginalised_x, bins=num_bins, range=(real_marginalised_x.min(), real_marginalised_x.max()))
                # gen_marginalised_x_hist = np.histogram(gen_marginalised_x, bins=num_bins, range=(gen_marginalised_x.min(), gen_marginalised_x.max()))

                pk = real_marginalised_x + eps
                pk /= pk.sum()
                qk = gen_marginalised_x + eps
                qk /= qk.sum()


                if feat == ' ':
                    fig, axes = plt.subplots(2, 1)
                    # print(num_bins, feat, pk, qk)
                    print('\n\n ',feat)
                    print(real_marginalised_x)
                    print(gen_marginalised_x)
                    axes[0].set_title(feat)
                    axes[0].plot(real_marginalised_x_dict.keys(), pk)
                    plt.legend(['real'])
                    axes[1].plot(gen_marginalised_x_dict.keys(), qk)
                    plt.legend(['gen'])

                    # width = (real_marginalised_x_hist[1][1] - real_marginalised_x_hist[1][0])/1.2
                    # axes[0].bar(real_marginalised_x_hist[1][:-1], pk, width=width, align='center')
                    # plt.legend(['real'])
                    # width = (gen_marginalised_x_hist[1][1] - gen_marginalised_x_hist[1][0])/1.2
                    # axes[1].bar(gen_marginalised_x_hist[1][:-1], qk, width=width, align='center')
                    # plt.legend(['gen'])
                    plt.show()

                kl = entropy(pk=pk, qk=qk)
                # kl = entropy(pk=real_marginalised_x, qk=gen_marginalised_x)
                # print(f'KL-divergence for feature {feat} with index {i} is {kl}')
                kls[feat].append(kl)
                print(kl, feat)

                # if marked:
                #     print(kl, feat, real_unique_x, gen_unique_x)
                #     marked = False


    final_kls = []
    for feat in all_feats:
        if np.isnan(np.array(kls[feat])).all():
            final_kl = np.nan
        else:
            feat_kl_scores = np.array(kls[feat])
            nan_mask = np.isnan(feat_kl_scores)
            feat_kl_scores = feat_kl_scores[~nan_mask]
            final_kl = feat_kl_scores.mean()
        final_kls.append(final_kl)

    print('Final kls size:', len(final_kls))
    return final_kls # get_ranking(args, feats, kls)


def get_kl_ordering(args, real_data_, gen_data_, eps=1e-12):
    if args.use_case == 'botnet':
        real_nan_cols = get_real_nan_columns(args, real_data_)
        real_data_ = real_data_.drop(columns=real_nan_cols)
        gen_data_ = gen_data_.drop(columns=real_nan_cols)

    feats = real_data_.columns
    real_data = real_data_.to_numpy()
    gen_data = gen_data_.to_numpy()
    print('Using real and gen. data shapes:', real_data.shape, gen_data.shape)

    # real_pdf = gaussian_kde(np.transpose(real_data))
    # gen_pdf = gaussian_kde(np.transpose(gen_data))

    real_model = KernelDensity(kernel='gaussian', bandwidth=2).fit(real_data)
    print('Got KDE model')
    real_probabilities = real_model.score_samples(real_data)
    print('Scored real probs)')
    gen_probabilities = real_model.score_samples(gen_data)
    print('Scored gen probs)')

    real_probabilities = np.exp(real_probabilities)
    real_probabilities /= sum(real_probabilities)
    gen_probabilities = np.exp(gen_probabilities)
    gen_probabilities /= sum(gen_probabilities)

    kls = []
    marked= False
    for i,feat in tqdm(enumerate(feats), total=len(feats)):
        real_freq_x = real_data_[feat]
        real_unique_x = real_freq_x.unique()
        gen_freq_x = gen_data_[feat]
        gen_unique_x = gen_freq_x.unique()

        values_x = sorted(list(set(list(real_unique_x) + list(gen_unique_x))))
        # if len(gen_unique_x)/len(real_unique_x) < 0.5 or len(real_unique_x) < 0.5:
        #     print(f'Kept {len(values_x)} from {len(real_unique_x)} {len(gen_unique_x)}, feat {feat}')
        #     marked = True
        #     values_x = list(set(real_unique_x))
        # if len(real_unique_x)/len(gen_unique_x) < 0.5:
        #     print(f'Kept {len(values_x)} from {len(real_unique_x)} {len(gen_unique_x)}, feat {feat}')
        #     marked = True
        #     values_x = list(set(gen_unique_x))

        # values_x = sorted(real_unique_x)
        real_marginalised_x = {}
        for val in values_x:
            if val not in real_unique_x:
                real_marginalised_x[val] = 0.
                continue
            mask = real_freq_x == val
            positions = real_freq_x[mask].index
            real_marginalised_x[val] = sum(real_probabilities[positions])
        # real_marginalised_x = np.array(list(real_marginalised_x.values()))
        # real_marginalised_x /= real_marginalised_x.sum()
        real_marginalised_x_dict = normalise_probs(real_marginalised_x)

        # values_x = sorted(gen_unique_x)
        gen_marginalised_x = {}
        for val in values_x:
            if val not in gen_unique_x:
                gen_marginalised_x[val] = 0.
                continue
            mask = gen_freq_x == val
            positions = gen_freq_x[mask].index
            gen_marginalised_x[val] = sum(gen_probabilities[positions]) + eps
        # gen_marginalised_x = np.array(list(gen_marginalised_x.values()))
        # gen_marginalised_x /= gen_marginalised_x.sum()
        gen_marginalised_x_dict = normalise_probs(gen_marginalised_x)

        # for key in real_marginalised_x_dict:
        #     if key not in gen_marginalised_x_dict:
        #         gen_marginalised_x_dict[key] = 0.
        # for key in gen_marginalised_x_dict:
        #     if key not in real_marginalised_x_dict:
        #         real_marginalised_x_dict[key] = 0.

        real_marginalised_x = np.array(list(real_marginalised_x_dict.values()))
        gen_marginalised_x = np.array(list(gen_marginalised_x_dict.values()))



        # min_bound = (real_marginalised_x.min() + gen_marginalised_x.min())/2.
        # max_bound = (real_marginalised_x.max() + gen_marginalised_x.max())/2.
        #
        # min_bound = real_marginalised_x.min()
        # max_bound = real_marginalised_x.max()

        # min_bound = min(real_marginalised_x.min(), gen_marginalised_x.min())
        # max_bound = max(real_marginalised_x.max(), gen_marginalised_x.max())
        # range_bounds = (min_bound, max_bound)

        # # num_bins = args.num_bins
        # # num_bins = min(len(real_marginalised_x), len(gen_marginalised_x), 200)
        # num_bins = min(max(30,len(real_unique_x),len(gen_unique_x)), args.num_bins)
        # # if num_bins < 100:
        # #     print(num_bins)
        # real_marginalised_x_hist = np.histogram(list(real_marginalised_x_dict.values()), bins=num_bins, range=range_bounds)
        # gen_marginalised_x_hist = np.histogram(list(gen_marginalised_x_dict.values()), bins=num_bins, range=range_bounds)

        # real_marginalised_x_hist = np.histogram(real_marginalised_x, bins=num_bins, range=(real_marginalised_x.min(), real_marginalised_x.max()))
        # gen_marginalised_x_hist = np.histogram(gen_marginalised_x, bins=num_bins, range=(gen_marginalised_x.min(), gen_marginalised_x.max()))

        pk = real_marginalised_x + eps
        pk /= pk.sum()
        qk = gen_marginalised_x + eps
        qk /= qk.sum()


        if feat == ' ':
            fig, axes = plt.subplots(2, 1)
            # print(num_bins, feat, pk, qk)
            print('\n\n ',feat)
            print(real_marginalised_x)
            print(gen_marginalised_x)
            axes[0].set_title(feat)
            axes[0].plot(real_marginalised_x_dict.keys(), pk)
            plt.legend(['real'])
            axes[1].plot(gen_marginalised_x_dict.keys(), qk)
            plt.legend(['gen'])

            # width = (real_marginalised_x_hist[1][1] - real_marginalised_x_hist[1][0])/1.2
            # axes[0].bar(real_marginalised_x_hist[1][:-1], pk, width=width, align='center')
            # plt.legend(['real'])
            # width = (gen_marginalised_x_hist[1][1] - gen_marginalised_x_hist[1][0])/1.2
            # axes[1].bar(gen_marginalised_x_hist[1][:-1], qk, width=width, align='center')
            # plt.legend(['gen'])
            plt.show()

        kl = entropy(pk=pk, qk=qk)
        # kl = entropy(pk=real_marginalised_x, qk=gen_marginalised_x)
        # print(f'KL-divergence for feature {feat} with index {i} is {kl}')
        kls.append(kl)
        # print(kl)

        # if marked:
        #     print(kl, feat, real_unique_x, gen_unique_x)
        #     marked = False
        if i % 20 == 0:
            print(kls)
    return kls # get_ranking(args, feats, kls)

def get_wasserstein_ordering(args, real_data_, gen_data_, num_seeds=5):
    if args.wasserstein_load:
        return get_loaded_wasserstein_ordering(args, num_seeds)
    else:
        real_data_ = real_data_.to_numpy()
        if isinstance(gen_data_, pd.DataFrame):
            gen_data_ = gen_data_.to_numpy()
        elif isinstance(gen_data_, torch.Tensor):
            gen_data_ = gen_data_.detach().numpy()
        wd_distances = []
        # for i in cont_idx:
        for i in range(len(args.feats)):
            train_data = real_data_[:, i].reshape(-1, 1)
            gen_data = gen_data_[:, i].reshape(-1, 1)

            # fit the minmax scaler on the cont feature train data
            scaler = MinMaxScaler(feature_range=(0, 1), copy=True, clip=True).fit(train_data)
            train_data = scaler.transform(train_data).reshape(-1)
            gen_data = scaler.transform(gen_data).reshape(-1)

            wd = wasserstein_distance(train_data, gen_data)
            wd_distances.append(wd)

        return wd_distances



def get_jsd_ordering(args, real_data_, gen_data_, num_seeds=5):
    real_data_ = real_data_.to_numpy()
    if isinstance(gen_data_, pd.DataFrame):
        gen_data_ = gen_data_.to_numpy()
    elif isinstance(gen_data_, torch.Tensor):
        gen_data_ = gen_data_.detach().numpy()

    js_div = jensenshannon(real_data_, gen_data_, axis=0)
    return js_div

def get_loaded_wasserstein_ordering(args, num_seeds):
    dataset_info = _load_json("datasets_info.json")
    # model_paths = _load_json(f"../model_paths/reeval_path_{args.model_type}_dist.json")[args.model_type]

    cat_idx = dataset_info[args.use_case]["manual_inspection_categorical_cols_idx"]
    cont_idx = list(set(np.arange(0, len(args.feats))) - set(cat_idx))

    version = 'unconstrained'
    wd_seeds = []
    for seed_run in range(num_seeds):
        # load the pkl file containing the Wasserstein Distance between the real data and generated data, using the unconstrained model
        pkl_path_name = f'{args.model_type}_{args.use_case}_{version}_{seed_run}_wd.pkl'
        path_id = f'./calculation_distributions/{pkl_path_name}'

        wd = pkl.load(open(path_id, 'rb'))
        wd_seeds.append(wd)

    wd_all_feats = np.zeros(len(args.feats)) - np.infty
    wd_loaded_ = np.nanmean(np.concatenate(wd_seeds, axis=0), axis=0)
    wd_all_feats[cont_idx] = wd_loaded_

    wds = wd_all_feats
    return wds


def get_corr_ordering(args, real_data_: pd.DataFrame, gen_data_: pd.DataFrame):
    feats = real_data_.columns

    real_corr_matrix = real_data_.corr().to_numpy()
    gen_corr_matrix = gen_data_.corr().to_numpy()

    # for corr in [real_corr_matrix, gen_corr_matrix]:
    #     plt.matshow(corr, cmap='viridis', interpolation='nearest')
    #     plt.colorbar()
    #     plt.show()

    real_nan_mask = np.isnan(real_corr_matrix)
    real_corr_matrix[real_nan_mask] = 0.
    gen_corr_matrix[real_nan_mask] = 0.

    corr_diff = np.abs(real_corr_matrix - gen_corr_matrix)
    # plt.matshow(corr_diff, cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    # plt.show()

    scores = []
    for i, feat in enumerate(feats):
        nan_mask = np.isnan(corr_diff[i])
        if nan_mask.all():
            score = np.nan  # TODO: 0, nan, or inf?
        else:
            corr_diff[i][nan_mask] = np.abs(real_corr_matrix[i][nan_mask])
            score = corr_diff[i].mean()
        scores.append(score)

    # return get_ranking(args, feats, scores)
    return scores


def get_synthetic_data(args):
    X_train, (cat_cols, cat_idx), (roundable_idx, round_digits) = read_csv(f"data/{args.use_case}/train_data.csv", args.use_case)
    train_data_cols = X_train.columns

    if args.quick:
        subset = 'tiny'
    else:
        subset = ''
    real_data = pd.read_csv(f"data/{args.use_case}/{subset}/{args.real_data_partition}_data.csv")
    syn_data_shape = real_data.shape[0]

    text = f'{args.model_dir}/model.pt'
    generator = torch.load(text)

    model = CTGAN(real_data)
    model._generator = generator

    all_gen_data = []
    for i in range(args.num_sampling_rounds):

        gen_data = model.sample(syn_data_shape, None, None)
        gen_data = pd.DataFrame(gen_data, columns=train_data_cols)
        pkl.dump(gen_data, open(f'{args.model_dir}/syn_data_{args.real_data_partition}_{i}.pkl', 'wb'))
        all_gen_data.append(gen_data)
    return all_gen_data


def average_rankings(args, scores):
    avg_scores = []
    for i in range(len(args.feats)):
        avg_score = sum([score[i] for score in scores])/args.num_sampling_rounds
        avg_scores.append(avg_score)

    avg_scores = np.array(avg_scores)
    return get_ranking(args, avg_scores)


def get_real_nan_columns(args, real_data_):
    feats = real_data_.columns
    real_corr_matrix = real_data_.corr().to_numpy()
    real_nan_mask = np.isnan(real_corr_matrix).all(axis=0)
    nan_cols = feats[real_nan_mask]
    return nan_cols



# def sample_tablegan(generator, n, train_data, batch_size, random_dim, discrete_columns_idx):
#     from data_processors.wgan.tab_scaler import TabScaler
#     transformer = TabScaler(out_min=-1.0, out_max=1.0, one_hot_encode=False)
#     train_data = torch.from_numpy(train_data.values.astype('float32'))
#     transformer.fit(train_data, discrete_columns_idx)
#     sides = [4, 8, 16, 24, 32]
#     for i in sides:
#         if i * i >= train_data.shape[1]:
#             side = i
#             break
#     generator.eval()
#     steps = n // batch_size + 1
#     data = []
#     for i in range(steps):
#         noise = torch.randn(batch_size, random_dim, 1, 1)
#         fake = generator(noise)
#         fake_re = fake.reshape(-1, side * side)
#         data.append(fake_re.detach().cpu().numpy())
#
#     data = np.concatenate(data, axis=0)
#     data = data[:, : train_data.shape[1]]
#     return transformer.inverse_transform(data[:n])


# def get_synth_tablegan_samples(args, model, X_val, X_test, columns, cat_idx, num_sampling_rounds,  batch_size, random_dim, use_case, wandb_run, exp):
#     all_gen_data = []
#     X_train, (cat_cols, cat_idx), (roundable_idx, round_digits) = read_csv(f"data/{args.use_case}/train_data.csv", use_case)
#     train_data_cols = X_train.columns
#
#     if args.quick:
#         subset = 'tiny'
#     else:
#         subset = ''
#     real_data = pd.read_csv(f"data/{args.use_case}/{subset}/{args.real_data_partition}_data.csv")
#     syn_data_shape = real_data.shape[0]
#
#     text = f'{args.model_dir}/model.pt'
#     generator = torch.load(text)
#
#     for i in range(args.num_sampling_rounds):
#         gen_data = sample_tablegan(generator, syn_data_shape, X_train, batch_size, random_dim, cat_idx)
#         gen_data = pd.DataFrame(gen_data, columns=train_data_cols)
#         gen_data.iloc[:, roundable_idx] = gen_data.iloc[:, roundable_idx].round(round_digits)
#         gen_data = gen_data.astype(X_train.dtypes)
#
#         pkl.dump(gen_data, open(f'{args.model_dir}/syn_data_{args.real_data_partition}_{i}.pkl', 'wb'))
#         all_gen_data.append(gen_data)
#
#     return all_gen_data


def load_saved_gen_data(args):
    data = pkl.load(open(args.gen_data_path, 'rb'))
    if args.model_type == "tablegan":
        poss = {'train':0, 'val':1, 'test':2}
        loaded_data = data[poss[args.real_data_partition]]
        all_gen_data = []
        for elem in loaded_data:
            elem = pd.DataFrame(elem, columns=args.feats).astype(args.dtypes)
            all_gen_data.append(elem)
    elif args.model_type in ["goggle"]:
        poss = {'train':0, 'val':1, 'test':2}
        loaded_data = data[args.real_data_partition]
        all_gen_data = []
        for elem in loaded_data:
            elem = pd.DataFrame(elem, columns=args.feats).astype(args.dtypes)
            all_gen_data.append(elem)
    else:
        loaded_data = data[args.real_data_partition]
        all_gen_data = []
        for elem in loaded_data:
            elem = pd.DataFrame(elem.detach().numpy(), columns=args.feats).astype(args.dtypes)
            all_gen_data.append(elem)
    args.num_sampling_rounds = len(all_gen_data)
    return all_gen_data, args


if __name__ == "__main__":
    args = get_args()
    args.scale = (0,1)

    # load data
    if args.use_case in ['lcld', 'botnet']:
        args.quick = True
        subset = 'tiny'
    else:
        subset = ''
    real_data_ = pd.read_csv(f"data/{args.use_case}/{subset}/{args.real_data_partition}_data.csv")  # e.g. test
    # gen_data_ = pkl.load(open(args.gen_data_path, 'rb'))  # e.g. experiments/exp_id/gen_test_data.csv
    args.feats = real_data_.columns
    args.dtypes = real_data_.dtypes

    if args.model_type == "wgan":
        # all_gen_data = get_synthetic_data(args)
        all_gen_data, args = load_saved_gen_data(args)

    elif args.model_type in ["tablegan", "ctgan", 'octgan', "tvae", "goggle"]:
        # all_gen_data = get_synth_tablegan_samples(args)
        all_gen_data, args = load_saved_gen_data(args)

    scorings = []
    for gen_data_ in tqdm(all_gen_data):
        if args.metric == 'kde':
            if (args.model_type != 'octgan' and args.use_case in ('botnet', 'faults')) or (args.use_case in ['faults'] and args.model_type == 'octgan')\
                    or (args.model_type == 'tablegan' and args.use_case in ('news'))\
                    or (args.model_type == 'goggle' and args.use_case in ('url')): # or (args.model_type == 'tvae' and args.use_case in ('lcld')) :
                scoring = get_kl_ordering_botnet(args, real_data_, gen_data_)
            else:
                scoring = get_kl_ordering(args, real_data_, gen_data_)
        elif args.metric == 'corr':
            scoring = get_corr_ordering(args, real_data_, gen_data_)
        elif args.metric == 'wasserstein':
            scoring = get_wasserstein_ordering(args, real_data_, gen_data_)
        elif args.metric == 'jsd':
            scoring = get_jsd_ordering(args, real_data_, gen_data_)
        else:
            pass
        scorings.append(scoring)

    final_ranking = average_rankings(args, scorings)
    # print('Final ranking is:\n', final_ranking)