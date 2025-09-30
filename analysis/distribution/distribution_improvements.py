import pandas as pd
import numpy as np
from utils import  _load_json
import argparse



def _parse_args():
    parser = argparse.ArgumentParser(description=' Command Line Interface')
    parser.add_argument("--postprocessing", action='store_true')
    parser.add_argument("--wandb_ignore", action='store_true')
    return parser.parse_args()

def main(args):
    calc_dir = 'calculation_distributions'
    round_digits = 2
    all_results = []
    all_results_latex = []
    for model in models:
        model_results = [model]
        for use_case in use_cases:
            X_train = pd.read_csv(f"data/{use_case}/train_data.csv")
            if use_case == 'lcld':
                X_train = pd.read_csv(f"data/{use_case}/tiny/train_data.csv")
            cat_idx = dataset_info[use_case]["manual_inspection_categorical_cols_idx"]
            cont_idx = list(set(np.arange(0,len(X_train.columns))) - set(cat_idx))

            seed_wd, seed_jsd = [], []
            for seed in range(5):
                try:
                    if args.postprocessing:
                        uncons_wd = pd.read_pickle(f"{calc_dir}/{model}_{use_case}_unconstrained_{seed}_wd.pkl")
                        cons_wd = pd.read_pickle(f"{calc_dir}/{model}_{use_case}_unconstrained_{seed}_postprocessing_wd.pkl")

                        uncons_jsd = pd.read_pickle(f"{calc_dir}/{model}_{use_case}_unconstrained_{seed}_jsd.pkl")
                        cons_jsd = pd.read_pickle(f"{calc_dir}/{model}_{use_case}_unconstrained_{seed}_postprocessing_jsd.pkl")
                    else:
                        uncons_wd = pd.read_pickle(f"{calc_dir}/{model}_{use_case}_unconstrained_{seed}_wd.pkl")
                        cons_wd = pd.read_pickle(f"{calc_dir}/{model}_{use_case}_constrained_{seed}_wd.pkl")

                        uncons_jsd = pd.read_pickle(f"{calc_dir}/{model}_{use_case}_unconstrained_{seed}_jsd.pkl")
                        cons_jsd = pd.read_pickle(f"{calc_dir}/{model}_{use_case}_constrained_{seed}_jsd.pkl")
           
                except Exception as e:
                    print(e)
                    continue
                sample_wd, sample_jsd = [], []
                for sample_i in range(5):
                    uncons_wd_rouned = np.round(np.array(uncons_wd[sample_i]),round_digits)
                    cons_wd_rounded = np.round(np.array(cons_wd[sample_i]),round_digits)
                    uncons_jsd_rouned = np.round(np.array(uncons_jsd[sample_i]),round_digits)
                    cons_jsd_rounded = np.round(np.array(cons_jsd[sample_i]),round_digits)
                    num_improved_wd = (uncons_wd_rouned-cons_wd_rounded>=0).sum()
                    num_improved_jsd = (uncons_jsd_rouned-cons_jsd_rounded>=0).sum()
                    sample_wd.append(num_improved_wd*100/len(cont_idx))
                    #sample_jsd.append(num_improved_jsd*100/len(cat_idx))
                    sample_jsd.append(num_improved_jsd*100/np.count_nonzero(~np.isnan(cons_jsd_rounded)))

                seed_wd.append(np.nanmean(sample_wd))
                seed_jsd.append(np.nanmean(sample_jsd))
            all_results.append([model, use_case, np.nanmean(seed_wd), np.nanstd(seed_wd),np.nanmean(seed_jsd), np.nanstd(seed_jsd) ])
            model_results.extend([np.nanmean(seed_wd), np.nanstd(seed_wd),np.nanmean(seed_jsd), np.nanstd(seed_jsd)])
        all_results_latex.append(model_results)

    results_df = pd.DataFrame(all_results, columns=["Model", "Dataset", "WD Mean", "WD Std", "JSD Mean", "JSD Std"])
    if args.postprocessing:
        results_df.to_csv("distribution_improvement_percentage_postprocessing.csv", index=False)
    else:
        results_df.to_csv("distribution_improvement_percentage.csv", index=False)

    cols = ["Model"]
    cols.extend(["WD Mean", "WD Std", "JSD Mean", "JSD Std"]*6)
    results_df_latex = pd.DataFrame(all_results_latex, columns=cols)
    print("Post", args.postprocessing)
    if args.postprocessing:
        results_df_latex.to_csv("distribution_improvement_percentage_latex_postprocessing.csv", index=False)
    else:
        results_df_latex.to_csv("distribution_improvement_latex_percentage.csv", index=False)

if __name__ =="__main__":

    args = _parse_args()
    dataset_info = _load_json("datasets_info.json")
    use_cases = ["url", "wids", "lcld", "heloc", "faults", "news"]
    models = ["wgan", "tablegan","ctgan", "tvae", "goggle"]

    main(args)