import wandb
import pandas as pd
import numpy as np
import pickle
from timeit import default_timer as timer

from evaluation.eval import constraints_sat_check, eval_synthetic_data
from run_evaluation.reeval_final import prepare_gen_data
from utils import all_div_gt_n

def add_common_args(parser):
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("use_case", type=str, choices=["url","wids","lcld","heloc","news","faults"])
    parser.add_argument("--version", type=str, default='unconstrained', choices=['unconstrained','constrained', "postprocessing"],
                        help='Version of training. Correct values are unconstrained, constrained and postprocessing')
    parser.add_argument("--label_ordering", default='random', choices=['random', 'corr', 'kde', 'wasserstein', 'jsd', 'causal'])
    parser.add_argument("--wandb_project", default='c-dgm_test', type=str)
    parser.add_argument("--wandb_mode", default="online", type=str, choices=['online', 'disabled', 'offline'])
    parser.add_argument('-e', '--epochs', default=300, type=int,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size. Must be an even number.')
    parser.add_argument('-n', '--num-samples', type=int,
                        help='Number of rows to sample. Defaults to the training data size')
    parser.add_argument('--num_sampling_rounds', type=int, default=5)
    parser.add_argument("--save_every_n_epochs", default=50, type=int)
    parser.add_argument('--skip_evaluation', action='store_true')
    parser.add_argument('--runtime_evaluation_only', action='store_true')
    parser.add_argument('--performance_evaluation', action='store_true')
    parser.add_argument('--save_generated_data', action='store_true')
    parser.add_argument("--use_only_target_original_dtype", action='store_true')
    return parser

def set_pac(args):
  if args.pac != 1:
    if args.batch_size % args.pac != 0:
        original_pac = args.pac
        args.pac = all_div_gt_n(original_pac, args.batch_size)
        print(f'Changed pac {original_pac} to {args.pac}')

def generate_evaluation_data(args, model, real_data, columns):
    gen_data, unconstrained_gen_data, constrained_unrounded_gen_data = [[[] for _ in range(3)] for _ in range(3)]
    sizes = [real_data['train'].shape[0], real_data['val'].shape[0], real_data['test'].shape[0]]
    for r in range(args.num_sampling_rounds): 
        for i in range (len(sizes)):
            if args.model_type == "GOGGLE":
                X_partition = args.goggle_data[i]
                sampled_data, unconstrained_output = model.sample(X_partition, inverse_transformed=True)
                unconstrained_output = unconstrained_output.to_numpy()
                sampled_data = sampled_data.to_numpy()
            elif args.model_type == "CTGAN":
                sampled_data, unconstrained_output = model.sample(sizes[i], args.sample_condition_column, args.sample_condition_column_value)
            elif args.model_type == "TableGAN":
                sampled_data, unconstrained_output = model.sample(sizes[i], len(columns))
            elif args.model_type == "WGAN":
                sampled_data, unconstrained_output = model.sample(sizes[i], args.input_length)
            else:
                sampled_data, unconstrained_output = model.sample(sizes[i])
            unconstrained_gen_data[i].append(unconstrained_output)
            constrained_unrounded_output = pd.DataFrame(sampled_data, columns=columns).astype(float)
            constrained_unrounded_output[columns[-1]] = constrained_unrounded_output[columns[-1]].astype(real_data['train'].dtypes[-1])
            constrained_unrounded_gen_data[i].append(constrained_unrounded_output)

    unconstrained_generated_data = {"train":unconstrained_gen_data[0], "val":unconstrained_gen_data[1], "test":unconstrained_gen_data[2]}
    constrained_unrounded_generated_data = {"train":constrained_unrounded_gen_data[0], "val":constrained_unrounded_gen_data[1], "test":constrained_unrounded_gen_data[2]}

    if args.save_generated_data:
        with open(f'{args.path}/unconstrained_generated_data.pkl', 'wb') as f:
            pickle.dump(unconstrained_generated_data, f)
        with open(f'{args.path}/constrained_unrounded_generated_data.pkl', 'wb') as f:
            pickle.dump(constrained_unrounded_generated_data, f)
    return unconstrained_generated_data, constrained_unrounded_generated_data      



def evaluate_model_performance(args, unconstrained_generated_data, roundable_idx, round_digits, columns, real_data, dataset_info):
    ########################### WandB initialization ###################################
    args.wandb_project = f"evaluation_{args.model_type}_{args.use_case}_test"
    wandb_run = wandb.init(project=args.wandb_project, id=args.exp_id)
    for k, v in args._get_kwargs():
        wandb_run.config[k] = v
    ####################################################################################
    args.round_before_cons = False
    args.round_after_cons = False
    args.postprocessing = False
       
    if args.version != 'unconstrained':
        args.version = args.label_ordering

  
    generated_data, constrained_unrounded_generated_data = prepare_gen_data(args, unconstrained_generated_data, roundable_idx, round_digits, columns, real_data["train"])
    constraints_sat_check(args, real_data, constrained_unrounded_generated_data, log_wandb=True)
        
    print('Using evaluators with the following specs', dataset_info["problem_type"], dataset_info["target_size"], dataset_info["target_col"])
    eval_synthetic_data(args, args.use_case, real_data, generated_data, columns,
                            problem_type=dataset_info["problem_type"], target_utility=dataset_info["target_col"],
                            target_utility_size=dataset_info["target_size"], target_detection="", log_wandb=True,
                            wandb_run=wandb_run, unrounded_generated_data_for_cons_sat=constrained_unrounded_generated_data)

def runtime_evaluation(args, model, wandb_run):
    size = 1000
    runs = []
    for i in range(args.num_sampling_rounds):
        start = timer()
        if args.model_type == "GOGGLE":
            X_partition = args.goggle_data[i]
            sampled_data, unconstrained_output = model.sample(X_partition, inverse_transformed=True)
            unconstrained_output = unconstrained_output.to_numpy()
            sampled_data = sampled_data.to_numpy()
        elif args.model_type == "CTGAN":
            sampled_data, unconstrained_output = model.sample(size, args.sample_condition_column, args.sample_condition_column_value)
        elif args.model_type == "TableGAN":
            sampled_data, unconstrained_output = model.sample(size, len(args.train_data_cols))
        elif args.model_type == "WGAN":
            sampled_data, unconstrained_output = model.sample(size, args.input_length)
        else:
            sampled_data, unconstrained_output = model.sample(size)
        end = timer()
        runtime = end - start
        runs.append(runtime)
    runtime_df = pd.DataFrame(list(zip([np.mean(runs)],[np.std(runs)])), columns=["Mean", "Std"])
    wandb_run.log({'Runtime/Sampling': runtime_df})
    print("Runtime evaluation completed and logged to WandB.")

