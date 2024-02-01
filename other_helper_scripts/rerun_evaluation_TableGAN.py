import os
import numpy as np
import pandas as pd
import torch
import wandb 

from data_processors.wgan.tab_scaler import TabScaler
from synthetizers.tableGAN import TableGAN
from evaluation.eval import eval_synthetic_data
from utils import set_seed, read_csv, get_target_col


# TODO: Adapt  this function to WGAN sampling strategy
def sample(generator, n, train_data, batch_size, random_dim, discrete_columns_idx):
    transformer = TabScaler(out_min=-1.0, out_max=1.0, one_hot_encode=False)
    train_data = torch.from_numpy(train_data.values.astype('float32'))
    transformer.fit(train_data, discrete_columns_idx)
    sides = [4, 8, 16, 24, 32]
    for i in sides:
        if i * i >= train_data.shape[1]:
            side = i
            break
    generator.eval()
    steps = n // batch_size + 1
    data = []
    for i in range(steps):
        noise = torch.randn(batch_size, random_dim, 1, 1)
        fake = generator(noise)
        fake_re = fake.reshape(-1, side * side)
        data.append(fake_re.detach().cpu().numpy())
    
    data = np.concatenate(data, axis=0)
    data = data[:, : train_data.shape[1]]
    return transformer.inverse_transform(data[:n])



def perform_evaluation(model, X_train, X_val, X_test, columns, cat_idx, num_sampling_rounds,  batch_size, random_dim, use_case, wandb_run, exp):
    gen_data = [[], [], []]
    sizes = [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
    for r in range(num_sampling_rounds): 
        for i in range (len(sizes)):

            # TODO: Change this function call for WGAN
            sampled_data = sample(model, sizes[i], X_train, batch_size, random_dim, cat_idx)
            #sampled_data = model.sample(sizes[i])
            sampled_data = pd.DataFrame(sampled_data, columns=columns)
            sampled_data.iloc[:, roundable_idx] = sampled_data.iloc[:, roundable_idx].round(round_digits)
            sampled_data = sampled_data.astype(float)
            target_col = columns[-1]
            sampled_data[target_col] = sampled_data[target_col].astype(X_train.dtypes[-1])
            gen_data[i].append(sampled_data)

    real_data = {"train":X_train, "val":X_val, "test":X_test}
    generated_data = {"train":gen_data[0], "val":gen_data[1], "test":gen_data[2]}

    target_utility = get_target_col(use_case)

    from argparse import Namespace
    args = Namespace(exp_path=exp)
    eval_synthetic_data(args, use_case, real_data, generated_data, columns, problem_type='binary_classification', target_utility=target_utility, target_detection="", log_wandb=True, wandb_run=wandb_run)



if __name__ =="__main__":

    num_sampling_rounds = 5
    #use_cases = ["url", "lcld"]
    # use_cases = ["lcld"]
    use_cases = ["wids"]

    for use_case in use_cases:

        main_dir = f"TableGAN_out/{use_case}/unconstrained/"
        print(main_dir)
        print(os.listdir(main_dir))
        exp_dirs = [d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]
        X_train, (cat_cols, cat_idx), (roundable_idx, round_digits) = read_csv(f"data/{use_case}/train_data.csv", use_case)
        X_test = pd.read_csv(f"data/{use_case}/test_data.csv")
        X_val = pd.read_csv(f"data/{use_case}/val_data.csv")

        target_col = get_target_col(use_case)

        if use_case in ['lcld', 'botnet']:
            X_train = X_train.groupby(target_col, group_keys=False).apply(lambda x: x.sample(frac=0.1)).sample(
                frac=1).reset_index(drop=True)
            X_test = X_test.groupby(target_col, group_keys=False).apply(lambda x: x.sample(frac=0.1)).sample(
                frac=1).reset_index(drop=True)
            X_val = X_val.groupby(target_col, group_keys=False).apply(lambda x: x.sample(frac=0.1)).sample(
                frac=1).reset_index(drop=True)

        columns = X_train.columns.values.tolist()
        for exp in exp_dirs:

            # if exp in ["unconstrained_9_adam_20_512_100_128_01-08-23--16-25-50",
            #     "unconstrained_9_adam_20_128_100_64_27-07-23--23-19-23",
            #     "unconstrained_9_rmsprop_20_256_100_64_28-07-23--19-12-55",
            #     "unconstrained_9_rmsprop_20_128_100_64_28-07-23--07-18-36",
            #     "unconstrained_9_rmsprop_20_128_100_64_28-07-23--02-02-09",
            #     "unconstrained_9_sgd_20_256_100_64_29-07-23--00-20-55",
            #     "unconstrained_9_sgd_20_128_100_64_28-07-23--12-03-29",
            #     "unconstrained_9_sgd_20_512_100_64_29-07-23--13-35-00",
            #     "unconstrained_9_adam_20_128_100_64_27-07-23--18-43-23",
            #     ]:
            print(exp)
            if exp not in ['unconstrained_9_adam_50_256_100_64_10-08-23--08-30-26', 'unconstrained_9_adam_50_256_100_64_10-08-23--07-39-03', 'unconstrained_9_adam_50_256_100_64_10-08-23--06-48-17', 'unconstrained_9_sgd_50_128_100_64_10-08-23--05-56-03', 'unconstrained_9_sgd_50_128_100_64_10-08-23--05-03-18', 'unconstrained_9_rmsprop_50_128_100_64_10-08-23--04-10-32', 'unconstrained_9_rmsprop_50_128_100_64_10-08-23--03-07-28', 'unconstrained_9_rmsprop_50_128_100_64_10-08-23--02-02-02', 'unconstrained_9_adam_50_128_100_64_10-08-23--00-56-37', 'unconstrained_9_adam_50_128_100_64_09-08-23--23-49-40', 'unconstrained_9_adam_50_128_100_64_09-08-23--22-40-34']:
                continue
            wandb_run = wandb.init(project=f"TableGAN_rerun_evaluation_{use_case}_10perc", id=exp, reinit=True)

            # TODO: Update this so you can get the parameters needed for WGAN
            # For URL:   exp_id = f"{args.version}_{args.seed}_{args.epochs}_{args.batch_size}_{args.random_dim}_{args.num_channels}_{DATETIME:%d-%m-%y--%H-%M-%S}"
            # For LCLD I added  optimiser:  exp_id = f"{args.version}_{args.seed}_{args.optimiser}_{args.epochs}_{args.batch_size}_{args.random_dim}_{args.num_channels}_{DATETIME:%d-%m-%y--%H-%M-%S}"

            parameters = exp.split('_')
            if use_case == "url":
                batch_size = int(parameters[3])
                random_dim = int(parameters[4])
                seed = int(parameters[1])
            else:
                batch_size = int(parameters[4])
                random_dim = int(parameters[5])
                seed = int(parameters[1])

            set_seed(seed)
            model_path = main_dir + exp + "/model.pt"

            print(model_path)
            try:
                model = torch.load(model_path)
            except Exception as e:
                continue
            model.eval()
            perform_evaluation(model, X_train, X_val, X_test, columns, cat_idx, num_sampling_rounds,  batch_size, random_dim, use_case, wandb_run, main_dir + exp)
    