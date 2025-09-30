import pandas as pd
import numpy as np
import pickle
import argparse


def load_pickle(path):
    with open(path, 'rb') as f:
        data=pickle.load(f)
    return data

def border_points_faults(data, col_idx1, col_idx2, quant, range_stripe):
    abs_diff = np.abs(data[:,col_idx2] - data[:,col_idx1])
    border_perc  = data[abs_diff<=0.1].shape[0]/data.shape[0]
    border_perc_quant  = data[abs_diff<=quant].shape[0]/data.shape[0]
    border_perc_range  = data[abs_diff<=range_stripe].shape[0]/data.shape[0]
    return border_perc*100, border_perc_quant*100, border_perc_range*100

def get_args():
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    parser.add_argument("dataset", type=str, default='wids', choices=['wids','faults', "heloc"])
    return parser.parse_args()

if __name__=="__main__":

    args = get_args()
    if args.dataset == "faults":
        col_idx1 = 0
        col_idx2 = 1
    elif args.dataset == "heloc":
        col_idx1 = 5
        col_idx2 = 6
    elif args.dataset == "wids":
        col_idx1 = 83
        col_idx2 = 84


    real_data = pd.read_csv(f"data/{args.dataset}/train_data.csv")
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


            if not skip_border_calc:
                j = 0
                for key in data.keys():
                    if key == "real":
                        perc_border, perc_border_quant, perc_border_range = border_points_faults(data[key], col_idx1, col_idx2, q10, range_stripe)
                        results[list(results.keys())[j]].append(perc_border)
                        results_quant[list(results_quant.keys())[j]].append(perc_border_quant)
                        results_range[list(results_range.keys())[j]].append(perc_border_range)
                        j += 1

                    else: 
                        for m in range(2):
                            perc_border, perc_border_quant, perc_border_range = border_points_faults(data[key][m], col_idx1, col_idx2, q10, range_stripe)
                            results[list(results.keys())[j]].append(perc_border)
                            results_quant[list(results_quant.keys())[j]].append(perc_border_quant)
                            results_range[list(results_range.keys())[j]].append(perc_border_range)
                            j += 1

            print("______________________TOL_______________________")
            for key in results.keys():
                print(key, np.mean(results[key]), np.std(results[key]))

            print("______________________QUANTILES_______________________")
            for key in results_quant.keys():
                print(key, np.mean(results_quant[key]), np.std(results_quant[key]))


            print("______________________RANGE_______________________")
            df_list = []
            for key in results_range.keys():
                df_list.append([key, np.mean(results_range[key]), np.std(results_range[key])])
                print(key, np.mean(results_range[key]), np.std(results_range[key]))
            df = pd.DataFrame(df_list, columns = ['Model', 'Mean', 'Std']) 
            df.to_csv(f"dist_plots/{args.dataset}_range_border_{perc}.csv", index=False)


