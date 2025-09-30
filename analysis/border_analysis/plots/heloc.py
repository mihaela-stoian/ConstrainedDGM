import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.ticker as plticker
import torch
import argparse


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
    plt.fill_between(x, y, color='lightcoral', alpha=0.7)
    sns.scatterplot(x=x_data, y=y_data, label="Real", alpha=0.4, color="green", s=90)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xlim(x_lim[0],x_lim[1])
    plt.ylim(y_lim[0],y_lim[1])
    ax = plt.gca()  
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.6, dash_capstyle='round', dashes=(10, 10))  # Adjust the dash pattern as needed
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    # ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.grid(alpha=0.3) 
    plt.legend(loc='lower right', handletextpad=0.05)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()


def uncons_cons_plot(x_data_uncons, y_data_uncons, x_data_cons, y_data_cons, min_point, max_point, x_label, y_label, x_lim, y_lim, model, path):

    plt.figure(figsize=(5.7,4.2))
    sns.set(font_scale=1.9)
    sns.set_style("whitegrid", {'grid.linestyle': '--'})

    x = np.linspace(min_point, max_point, 1000)
    y = x
    plt.fill_between(x, y, color='lightcoral', alpha=0.7)
    sns.scatterplot(x=x_data_uncons, y=y_data_uncons, label=model, alpha=0.4, s=90)
    sns.scatterplot(x=x_data_cons, y=y_data_cons, label=f"C-{model}", alpha=0.4, s=90, color="darkgoldenrod")

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xlim(x_lim[0],x_lim[1])
    plt.ylim(y_lim[0],y_lim[1])
    ax = plt.gca()  
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.6, dash_capstyle='round', dashes=(10, 10))  # Adjust the dash pattern as needed
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.legend(loc='lower right', handletextpad=0.05)
    plt.grid(alpha=0.5) 
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

def load_pickle(path):
    with open(path, 'rb') as f:
        data=pickle.load(f)
    return data

if __name__=="__main__":

    args = get_args()
    if args.model == "real":
        real_data = pd.read_csv("data/heloc/train_data.csv")
        real_data = real_data.to_numpy()
        real_plot(real_data[:,6], real_data[:,5],  -1, 11, "Insolvent >= 90 days", "Insolvent >= 60 days", (-0.3,10.2),  (-0.3,10.2),  "cons_plots/heloc_trades_real.pdf")

    else:
        cons_data = load_pickle(f"plots_data/faults/{args.model}/constrained_generated_data.pkl")
        uncons_data = load_pickle(f"plots_data/faults/{args.model}/unconstrained_generated_data.pkl")
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

        uncons_cons_plot(uncons_data[:,6].astype(int)+0.1, uncons_data[:,5].astype(int), cons_data[:,6].astype(int), cons_data[:,5].astype(int), -1, 11,
                          "Insolvent >= 90 days", "Insolvent >= 60 days", (0,10.2),  (0,10.2), args.model, f"cons_plots/{args.model}_uncons_cons.pdf")
