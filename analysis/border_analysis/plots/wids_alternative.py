import random

import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.ticker as plticker
import torch
import argparse
random.seed(0)
np.random.seed(0)

def get_args():
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    parser.add_argument("model", type=str, default='real', choices=['real','WGAN', "TableGAN", "CTGAN", "TVAE", "Goggle"])
    return parser.parse_args()



def real_plot(x_data, y_data, min_point, max_point, x_label, y_label, x_lim, y_lim, path):

    plt.figure(figsize=(5.5,4.5))
    sns.set(font_scale=1.9)
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'axes.grid.dash_capstyle': 'round'})
    x = np.linspace(min_point, max_point, 1000)
    y = x
    s1 = 40
    alpha1 = 0.4
    sns.scatterplot(x=x_data, y=y_data, label="Real", alpha=alpha1, color="green", s=s1)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xlim(x_lim[0],x_lim[1])
    plt.ylim(y_lim[0],y_lim[1])
    ax = plt.gca()  
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    # ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.6, dash_capstyle='round', dashes=(10, 10))  # Adjust the dash pattern as needed
    ax.grid(True, linestyle='-')
    ax.set_yticks(list(np.arange(y_lim[0], y_lim[1], 3, dtype=int)))
    ax.set_xticks(list(np.arange(y_lim[0], y_lim[1], 3, dtype=int)))
    plt.fill_between(x, y, color='lightcoral', alpha=0.2, edgecolor='red', linewidth=2)

    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    # ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.grid(alpha=0.15)
    lgnd = plt.legend(loc='lower right', handletextpad=-0.5)
    lgnd.legend_handles[0]._sizes = [100]
    plt.tight_layout()
    plt.savefig(path)
    plt.show()


def uncons_cons_plot(x_data_uncons, y_data_uncons, x_data_cons, y_data_cons, min_point, max_point, x_label, y_label, x_lim, y_lim, model, path, version):

    plt.figure(figsize=(5.5,4.5))
    sns.set(font_scale=1.9)
    sns.set_style("whitegrid", {'grid.linestyle': '--'})

    x = np.linspace(min_point, max_point, 1000)
    y = x

    s1 = 40
    alpha1 = 0.4
    if version=="both":
        sns.scatterplot(x=x_data_uncons, y=y_data_uncons, label=model, alpha=alpha1, s=s1)
        sns.scatterplot(x=x_data_cons, y=y_data_cons, label=f"C-{model}", alpha=alpha1, s=s1, color="darkgoldenrod")
    elif version=="uncons":
        sns.scatterplot(x=x_data_uncons, y=y_data_uncons, label=model, alpha=alpha1, s=s1)
    elif version=="cons":
        sns.scatterplot(x=x_data_cons, y=y_data_cons, label=f"C-{model}", alpha=alpha1, s=s1, color="darkgoldenrod")

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xlim(x_lim[0],x_lim[1])
    plt.ylim(y_lim[0],y_lim[1])
    ax = plt.gca()  
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    # ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.6, dash_capstyle='round', dashes=(10, 10))  # Adjust the dash pattern as needed
    ax.grid(True, linestyle='-')

    #ax.yaxis.label.set_color('white')

    ax.set_yticks(list(np.arange(y_lim[0], y_lim[1], 3, dtype=int)))
    ax.set_xticks(list(np.arange(y_lim[0], y_lim[1], 3, dtype=int)))

    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.fill_between(x, y, color='lightcoral', alpha=0.2, edgecolor='red', linewidth=2)

    #plt.yticks([]) 
    #[t.set_color('white') for t in ax.yaxis.get_ticklabels()]

    lgnd = plt.legend(loc='lower right', handletextpad=-0.5)
    lgnd.legend_handles[0]._sizes = [100]
    plt.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

def load_pickle(path):
    with open(path, 'rb') as f:
        data=pickle.load(f)
    return data

if __name__=="__main__":

    args = get_args()
    real_data = pd.read_csv("../data/wids/train_data.csv")

    # min_feat = 'd1_temp_min'
    # max_feat = 'd1_temp_max'
    # min_feat = 'h1_temp_min'
    # max_feat = 'h1_temp_max'
    # key_feat = 'temperature'

    max_feat = 'd1_hemaglobin_max'
    min_feat = 'd1_hemaglobin_min'
    key_feat = 'haemoglobin'

    min_feat_lb = real_data[min_feat].min()
    min_feat_ub = real_data[min_feat].max()
    max_feat_lb = real_data[max_feat].min()
    max_feat_ub = real_data[max_feat].max()

    # min_feat_lb = max_feat_lb = 0 #min(min_feat_lb, max_feat_lb)
    min_feat_lb = max_feat_lb = min(min_feat_lb, max_feat_lb)
    min_feat_ub = max_feat_ub = max(min_feat_ub, max_feat_ub) + 0.05 * max(min_feat_ub, max_feat_ub)

    y_min_id = list(real_data.columns.values).index(min_feat)
    y_max_id = list(real_data.columns.values).index(max_feat)

    real_data = real_data.to_numpy()
    real_data = real_data[real_data[:, y_min_id] <= real_data[:, y_max_id]]  # eliminate corrupted data
    idx = np.random.choice(np.arange(real_data.shape[0]), 2000, replace=False)
    if args.model == "real":
        #real_data = pd.read_csv("data/wids/train_data.csv")
        real_data = real_data.to_numpy()
        real_data = real_data[idx]
        real_plot(real_data[:,y_min_id], real_data[:,y_max_id],  0, max_feat_ub, f"Min {min_feat}", f"Max {max_feat}", (min_feat_lb,min_feat_ub),  (max_feat_lb,max_feat_ub),  "cons_plots/wids_temp_real.pdf")

    else:
        model = 'DGM'
        cons_data = load_pickle(f"plots_data/wids/{args.model}/constrained_generated_data.pkl")
        uncons_data = load_pickle(f"plots_data/wids/{args.model}/unconstrained_generated_data.pkl")
        cons_data = cons_data["train"][0].to_numpy()
        print(cons_data.shape)
        if args.model=="TableGAN":
            uncons_data = uncons_data[0][0]
        else:
            uncons_data = uncons_data["train"][0]

        if isinstance(uncons_data, pd.DataFrame):   
            uncons_data = uncons_data.to_numpy()
        elif isinstance(uncons_data, torch.Tensor):
            uncons_data = uncons_data.detach().numpy()
        cons_data = cons_data[idx]
        uncons_data = uncons_data[idx]
        uncons_cons_plot(uncons_data[:,y_min_id], uncons_data[:,y_max_id], cons_data[:,y_min_id], cons_data[:,y_max_id], 0, max_feat_ub, f"Min {key_feat}", f"Max {key_feat}", (min_feat_lb,min_feat_ub),  (max_feat_lb,max_feat_ub),
                          model, f"cons_plots/{args.model}_uncons_wids_{key_feat}.pdf", "uncons")
        uncons_cons_plot(uncons_data[:,y_min_id], uncons_data[:,y_max_id], cons_data[:,y_min_id], cons_data[:,y_max_id], 0, max_feat_ub, f"Min {key_feat}", f"Max {key_feat}", (min_feat_lb,min_feat_ub),  (max_feat_lb,max_feat_ub),
                         model, f"cons_plots/{args.model}_cons_wids_{key_feat}.pdf", "cons")

        # plot also real
        # real_data = real_data.to_numpy()
        real_data = real_data[idx]
        real_plot(real_data[:,y_min_id], real_data[:,y_max_id],  0, max_feat_ub, f"Min {key_feat}", f"Max {key_feat}", (min_feat_lb,min_feat_ub),  (max_feat_lb,max_feat_ub),  f"cons_plots/wids_{key_feat}_real.pdf")
