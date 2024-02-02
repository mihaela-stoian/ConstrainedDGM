"""CLI."""
import os
import argparse
import pickle
import datetime
import pandas as pd
import numpy as np
import wandb

from timeit import default_timer as timer

from synthetizers.TVAE.tvae import TVAE
from evaluation.eval import eval_synthetic_data, constraints_sat_check
from utils import set_seed, read_csv, _load_json
from evaluation.reeval_final import prepare_gen_data

# wandb.log({'accuracy': train_acc, 'loss': train_loss})
# wandb.config.dropout = 0.2
# wandb.alert(title="Low accuracy", text=f"Accuracy {acc} is below threshold {thresh}")
# https://docs.wandb.ai/guides/data-and-model-versioning/dataset-versioning?_gl=1*1la1mgf*_ga*MTMwNzYxOTUyOC4xNjU1MzA5NTE0*_ga_JH1SJHJQXJ*MTY3OTY3MTkyNC4xOC4xLjE2Nzk2NzI0MjguMTMuMC4w
# https://wandb.ai/dpaiton/splitting-tabular-data/reports/Tabular-Data-Versioning-and-Deduplication-with-Weights-Biases--VmlldzoxNDIzOTA1?_gl=1*1p4t0h4*_ga*MTMwNzYxOTUyOC4xNjU1MzA5NTE0*_ga_JH1SJHJQXJ*MTY3OTY3MTkyNC4xOC4xLjE2Nzk2NzI0MTUuMjYuMC4w
# https://docs.wandb.ai/guides/data-vis/tables-quickstart
DATETIME = datetime.datetime.now()


def _parse_args():
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--use_only_target_original_dtype", action='store_true')
    parser.add_argument("--wandb_project", default="tvae", type=str)
    parser.add_argument("--wandb_mode", default="online", type=str, choices=['online', 'disabled', 'offline'])
    parser.add_argument('-e', '--epochs', default=300, type=int,
                        help='Number of training epochs')
    parser.add_argument('-n', '--num-samples', type=int,
                        help='Number of rows to sample. Defaults to the training data size')
    parser.add_argument("--save_every_n_epochs", default=50, type=int)
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimension of input z to the encoder.')
    parser.add_argument('--compress_dims', type=str, default='128,128',
                        help='Dimension of each encoder layer. '
                        'Comma separated integers with no whitespaces.')
    parser.add_argument('--decompress_dims', type=str, default='128,128',
                        help='Dimension of each decoder layer. '
                        'Comma separated integers with no whitespaces.')
    parser.add_argument('--l2scale', type=float, default=1e-5,
                    help='')
    parser.add_argument('--loss_factor', type=float, default=2,
                help='Multiplier for the reconstruction error. Defaults to 2' )
    parser.add_argument("--label_ordering", default='random', choices=['random', 'corr', 'kde'])
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size. Must be an even number.')
    parser.add_argument('--save', default=None, type=str,
                        help='A filename to save the trained synthesizer.')
    parser.add_argument('--load', default=None, type=str,
                        help='A filename to load a trained synthesizer.')
    parser.add_argument("use_case", type=str, choices=["url","wids","botnet","lcld","heloc","news","faults"])
    parser.add_argument("--version", type=str, default='unconstrained', choices=['unconstrained','constrained', "postprocessing"],
                        help='Version of training. Correct values are unconstrained, constrained and postprocessing')
    parser.add_argument('--skip_evaluation', action='store_true')
    parser.add_argument('--runtime_evaluation_only', action='store_true')
    return parser.parse_args()


def main():
    """CLI."""
    args = _parse_args()
    set_seed(args.seed)
    exp_id = f"{args.version}_{args.label_ordering}_{args.seed}_{args.epochs}_{args.batch_size}_{DATETIME:%d-%m-%y--%H-%M-%S}"
    path = f"outputs/TVAE_out/{args.use_case}/{args.version}/{exp_id}"
    args.exp_path = path
    os.makedirs(path)


      ######################################################################
    wandb_run = wandb.init(project=args.wandb_project, id=exp_id, reinit=True, mode=args.wandb_mode)
    for k,v in args._get_kwargs():
        wandb_run.config[k] = v
    ######################################################################
    args.constraints_file = f'./data/{args.use_case}/{args.use_case}_constraints.txt'
    ######################################################################
    dataset_info = _load_json("datasets_info.json")[args.use_case]
    print(dataset_info)
    ######################################################################

    X_train, (cat_cols, cat_idx), (roundable_idx, round_digits) = read_csv(f"data/{args.use_case}/train_data.csv", args.use_case, dataset_info["manual_inspection_categorical_cols_idx"])
    X_test = pd.read_csv(f"data/{args.use_case}/test_data.csv")
    X_val = pd.read_csv(f"data/{args.use_case}/val_data.csv")
    columns = X_train.columns.values.tolist()
    args.train_data_cols = columns
    args.dtypes = X_train.dtypes

    if cat_cols == None:
        cat_cols = []
        cat_idx = []


    if args.load:
        model = TVAE.load(args.load)
    else:
        compress_dims = [int(x) for x in args.compress_dims.split(',')]
        decompress_dims = [int(x) for x in args.decompress_dims.split(',')]

        test_data = pd.read_csv(f"data/{args.use_case}/test_data.csv")

        model = TVAE(test_data,
            embedding_dim=args.embedding_dim, compress_dims=compress_dims,
            decompress_dims=decompress_dims, l2scale=args.l2scale, loss_factor=args.loss_factor,
            batch_size=args.batch_size, epochs=args.epochs, path=path, bin_cols_idx=cat_idx, version=args.version)

    model.set_random_state(args.seed)
    model.fit(args, X_train, cat_cols)

    if args.save is not None:
        model.save(args.save)

    if args.use_case == "botnet" or args.use_case == "lcld":
        X_train = pd.read_csv(f"data/{args.use_case}/tiny/train_data.csv")
        X_test = pd.read_csv(f"data/{args.use_case}/tiny/test_data.csv")
        X_val = pd.read_csv(f"data/{args.use_case}/tiny/val_data.csv")

    model.set_random_state(args.seed)
    num_sampling_rounds = 5

    if args.runtime_evaluation_only:
        size = 1000
        runs = []
        for i in range(num_sampling_rounds):
            start = timer()
            sampled_data, unconstrained_output = model.sample(size, args.sample_condition_column, args.sample_condition_column_value)
            end = timer()
            runtime = end - start
            runs.append(runtime)
        runtime_df = pd.DataFrame(list(zip([np.mean(runtime)],[np.std(runtime)])), columns=["Mean", "Std"])
        wandb.log({'Runtime/Sampling': runtime_df})
    
    else:

        gen_data = [[], [], []]
        unconstrained_gen_data = [[], [], []]
        constrained_unrounded_gen_data = [[], [], []]
        sizes = [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
        for r in range(num_sampling_rounds): 
            for i in range (len(sizes)):
                sampled_data, unconstrained_output = model.sample(sizes[i])
                unconstrained_gen_data[i].append(unconstrained_output)
                constrained_unrounded_output = sampled_data
                constrained_unrounded_output = pd.DataFrame(constrained_unrounded_output, columns=columns)
                constrained_unrounded_output = constrained_unrounded_output.astype(float)
                target_col = columns[-1]
                constrained_unrounded_output[target_col] = constrained_unrounded_output[target_col].astype(X_train.dtypes[-1])
                constrained_unrounded_gen_data[i].append(constrained_unrounded_output)

                # sampled_data = pd.DataFrame(sampled_data, columns=columns)
                # sampled_data.iloc[:, roundable_idx] = sampled_data.iloc[:, roundable_idx].round(round_digits)  # NOTE: this shouldn't be after the constraints have been applied! (fixed by removing constr correction from sample fc, and adding it below here)
                # sampled_data = sampled_data.astype(X_train.dtypes)

                gen_data[i].append(constrained_unrounded_output)


        generated_data = {"train":gen_data[0], "val":gen_data[1], "test":gen_data[2]}
        unconstrained_generated_data = {"train":unconstrained_gen_data[0], "val":unconstrained_gen_data[1], "test":unconstrained_gen_data[2]}
        constrained_unrounded_generated_data = {"train":constrained_unrounded_gen_data[0], "val":constrained_unrounded_gen_data[1], "test":constrained_unrounded_gen_data[2]}

        with open(f'{path}/generated_data.pkl', 'wb') as f:
            pickle.dump(generated_data, f)
        with open(f'{path}/unconstrained_generated_data.pkl', 'wb') as f:
            pickle.dump(unconstrained_generated_data, f)
        with open(f'{path}/constrained_unrounded_generated_data.pkl', 'wb') as f:
            pickle.dump(constrained_unrounded_generated_data, f)

        real_data = {"train": X_train, "val": X_val, "test": X_test}

       


        if not args.skip_evaluation: 

            wandb.finish()
            ######################################################################
            args.real_data_partition = 'test'
            args.model_type = 'tvae'

            wandb_project_name = f"evaluation_{args.model_type}_{args.use_case}_fixed_eval"
            if "hyperparam" in args.wandb_project:
                wandb_project_name += "_hyperparam_search"
            args.wandb_project = wandb_project_name

            wandb_run = wandb.init(project=args.wandb_project, id=exp_id)
            for k, v in args._get_kwargs():
                wandb_run.config[k] = v
            ######################################################################
            args.round_before_cons = False
            args.round_after_cons = False
            args.postprocessing = False
            if args.version != 'unconstrained':
                args.version = args.label_ordering

            generated_data, unrounded_generated_data = prepare_gen_data(args, unconstrained_generated_data, roundable_idx, round_digits, columns, X_train)
            for i in range (5):
                print(generated_data["train"][i].shape, generated_data["val"][i].shape, generated_data["test"][i].shape)
                print(unrounded_generated_data["train"][i].shape, generated_data["val"][i].shape, generated_data["test"][i].shape)

            # if args.seed < 3:
            constraints_sat_check(args, real_data, unrounded_generated_data, log_wandb=True)
            # sdv_eval_synthetic_data(args, args.use_case, real_data, generated_data, columns,
            #                         problem_type=dataset_info["problem_type"],
            #                         target_utility=dataset_info["target_col"], target_detection="", log_wandb=True,
            #                         wandb_run=wandb_run)
            print('Using evaluators with the following specs', dataset_info["problem_type"], dataset_info["target_size"],
                dataset_info["target_col"])
            eval_synthetic_data(args, args.use_case, real_data, generated_data, columns,
                                problem_type=dataset_info["problem_type"], target_utility=dataset_info["target_col"],
                                target_utility_size=dataset_info["target_size"], target_detection="", log_wandb=True,
                                wandb_run=wandb_run, unrounded_generated_data_for_cons_sat=unrounded_generated_data)


            # if args.seed < 3:
            #     constraints_sat_check(args, real_data, generated_data, log_wandb=True)
            #     sdv_eval_synthetic_data(args, args.use_case, real_data, generated_data, columns, problem_type=dataset_info["problem_type"], target_utility=dataset_info["target_col"], target_detection="", log_wandb=True, wandb_run=wandb_run)
            #     eval_synthetic_data(args, args.use_case, real_data, generated_data, columns, problem_type=dataset_info["problem_type"], target_utility=dataset_info["target_col"], target_utility_size=dataset_info["target_size"], target_detection="", log_wandb=True, wandb_run=wandb_run)

if __name__ == '__main__':

    main()
