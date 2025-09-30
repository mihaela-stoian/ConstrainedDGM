import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.ticker as plticker
import torch



def load_pickle(path):
    with open(path, 'rb') as f:
        data=pickle.load(f)
    return data


def histplot(dfs, labels, column_idx):
    # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
    plt.figure(figsize=(5.5,4.5))
    sns.set(font_scale=1.9)
    sns.set_style("whitegrid", {'grid.linestyle': '--'})

    sns.kdeplot(dfs[0][:,column_idx], fill=True,  label=labels[0])
    sns.kdeplot(dfs[1][:,column_idx],  fill=True,  label=labels[1])
    sns.kdeplot(dfs[2][:,column_idx],  fill=True,  label=labels[2])
    sns.kdeplot(dfs[3][:,column_idx],  fill=True,   label=labels[3])
    plt.legend()
    plt.show()


def histplot_subplots(uncons_dfs, cons_dfs, labels, column_idx, xaxis_label, path):

    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].set_title('WGAN')
    axes[1].set_title('TableGAN')
    axes[2].set_title('CTGAN')
    axes[3].set_title('TVAE')

    for i in range(4):

        axes[i].ticklabel_format(style="scientific")
        sns.kdeplot(cons_dfs[0][:,column_idx], fill=True,   alpha=0.15, label=labels[0], ax=axes[i], )
        sns.kdeplot(uncons_dfs[i+1][:,column_idx], fill=True,   alpha=0.15, label=labels[i+1], ax=axes[i])
        sns.kdeplot(cons_dfs[i+1][:,column_idx], fill=True,   alpha=0.15, label='C-'+labels[i+1], ax=axes[i])
        axes[i].legend()
        axes[i].set_ylim(0,0.0016)
        axes[i].set_xlim(-490,2300)
        # axes[1].set_ylabel("")
        # axes[1].set_yticklabels([])
        axes[i].set_xlabel(xaxis_label)
        axes[i].grid(alpha=0.5) 

    plt.grid(alpha=0.5) 
    plt.tight_layout()
    plt.savefig(path)
    plt.show()


if __name__=="__main__":
    col_idx1 = 0
    col_idx2 = 1

    real_data = pd.read_csv("data/faults/train_data.csv")
    xaxis_label1 = real_data.columns[col_idx1]
    xaxis_label2 = real_data.columns[col_idx2]
    print(real_data[xaxis_label1])
    print(real_data[xaxis_label2])
    real_data = real_data.to_numpy()
    skip_border_calc = False

    # _________ Quantile __________
    abs_diff = np.abs(real_data[:,col_idx1] - real_data[:,col_idx2])
    q10 = np.quantile(abs_diff, [0.2])
    print(q10)

    # _________ Range __________
    range_col1 = np.abs(real_data[:,col_idx1].max() - real_data[:,col_idx1].min())
    range_col2 = np.abs(real_data[:,col_idx2].max() - real_data[:,col_idx2].min())
    print(real_data[:,col_idx1].min(), real_data[:,col_idx1].max())
    print(real_data[:,col_idx2].min(), real_data[:,col_idx2].max())
    print(range_col1)
    print(range_col2)

    percs = [0.05, 0.01, 0.5]
    for perc in percs:
        range_stripe = perc*np.sqrt((range_col1**2 + range_col1**2))
        print("Perc:", perc, "Stripe:", range_stripe)

   
        results = {"b_real":[], "b_wgan":[], "b_cwgan":[], "b_tablegan":[], "b_ctablegan":[], "b_ctgan":[], "b_cctgan":[], "b_tvae":[], "b_ctvae":[], "b_goggle":[], "b_cgoggle":[]} 
        results_quant = {"b_real":[], "b_wgan":[], "b_cwgan":[], "b_tablegan":[], "b_ctablegan":[], "b_ctgan":[], "b_cctgan":[], "b_tvae":[], "b_ctvae":[], "b_goggle":[], "b_cgoggle":[]} 
        results_range = {"b_real":[], "b_wgan":[], "b_cwgan":[], "b_tablegan":[], "b_ctablegan":[], "b_ctgan":[], "b_cctgan":[], "b_tvae":[], "b_ctvae":[], "b_goggle":[], "b_cgoggle":[]} 
     

        for i in range(5):
            models = ["WGAN", "TableGAN", "CTGAN", "TVAE", "GOGGLE"]
            data = {"real": real_data}
            for model in models:
                if model=="TableGAN":
                    uncons_data = load_pickle(f"plots_data/faults/{model}/unconstrained_generated_data.pkl")[0][i]
                    cons_data = load_pickle(f"plots_data/faults/{model}/constrained_generated_data.pkl")["train"][i].to_numpy()
                else:
                    uncons_data = load_pickle(f"plots_data/faults/{model}/unconstrained_generated_data.pkl")["train"][i].detach().numpy()
                    cons_data = load_pickle(f"plots_data/faults/{model}/constrained_generated_data.pkl")["train"][i].to_numpy()

            data[model] = [uncons_data, cons_data]



        # if i==0:
        #     # histplot([real_data, uncons_data_wgan, uncons_data_tablegan, uncons_data_ctgan], ["Real", "WGAN", "TableGAN", "CTGAN"], 0)
        #     # histplot([real_data, cons_data_wgan, cons_data_tablegan, cons_data_ctgan], ["Real", "C-WGAN", "C-TableGAN", "C-CTGAN"], 0)
        #     histplot_subplots([real_data, uncons_data_wgan, uncons_data_tablegan, uncons_data_ctgan, uncons_data_tvae], 
        #                     [real_data, cons_data_wgan, cons_data_tablegan, cons_data_ctgan, cons_data_tvae], 
        #                     ["Real", "WGAN", "TableGAN", "CTGAN", "TVAE"], 
        #                     col_idx1, 
        #                     xaxis_label1,
        #                     f"dist_plots/faults_feat{xaxis_label1}.pdf")

        #     histplot_subplots([real_data, uncons_data_wgan, uncons_data_tablegan, uncons_data_ctgan, uncons_data_tvae], 
        #                     [real_data, cons_data_wgan, cons_data_tablegan, cons_data_ctgan, cons_data_tvae], 
        #                     ["Real", "WGAN", "TableGAN", "CTGAN", "TVAE"], 
        #                     col_idx2, 
        #                     xaxis_label2,
        #                     f"dist_plots/faults_feat{xaxis_label1}.pdf")
        