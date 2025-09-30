import wandb
import pandas as pd
import numpy as np
import os
import pickle
from timeit import default_timer as timer
import torch

from evaluation.eval import constraints_sat_check, eval_synthetic_data, sdv_eval_synthetic_data
from run_evaluation.reeval_final import prepare_gen_data

def generate_evaluation_data(args, model, real_data, columns):
    gen_data, unconstrained_gen_data, constrained_unrounded_gen_data = [[[] for _ in range(3)] for _ in range(3)]
    sizes = [real_data['train'].shape[0], real_data['val'].shape[0], real_data['test'].shape[0]]
    for r in range(args.num_sampling_rounds): 
        for i in range (len(sizes)):
            sampled_data, unconstrained_output = model.sample(sizes[i], args.sample_condition_column, args.sample_condition_column_value)
            unconstrained_gen_data[i].append(unconstrained_output)
            constrained_unrounded_output = pd.DataFrame(sampled_data, columns=columns).astype(float)
            constrained_unrounded_output[columns[-1]] = constrained_unrounded_output[columns[-1]].astype(real_data['train'].dtypes[-1])
            constrained_unrounded_gen_data[i].append(constrained_unrounded_output)

    unconstrained_generated_data = {"train":unconstrained_gen_data[0], "val":unconstrained_gen_data[1], "test":unconstrained_gen_data[2]}
    constrained_unrounded_generated_data = {"train":constrained_unrounded_gen_data[0], "val":constrained_unrounded_gen_data[1], "test":constrained_unrounded_gen_data[2]}

    if args.save_generated_data:
        with open(f'{path}/unconstrained_generated_data.pkl', 'wb') as f:
            pickle.dump(unconstrained_generated_data, f)
        with open(f'{path}/constrained_unrounded_generated_data.pkl', 'wb') as f:
            pickle.dump(constrained_unrounded_generated_data, f)
    return unconstrained_generated_data, constrained_unrounded_generated_data      



def evaluate_model_performance(args, model, roundable_idx, round_digits, columns, real_data, dataset_info):
    ########################### WandB initialization ###################################
    args.wandb_project = f"evaluation_{args.model_type}_{args.use_case}"
    wandb_run = wandb.init(project=args.wandb_project, id=args.exp_id)
    for k, v in args._get_kwargs():
        wandb_run.config[k] = v
    ####################################################################################
    if args.version != 'unconstrained':
        args.version = args.label_ordering

    unconstrained_generated_data, constrained_unrounded_generated_data = generate_evaluation_data(args, model, real_data, columns)
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
        sampled_data = model.sample(size)
        end = timer()
        runtime = end - start
        runs.append(runtime)
    runtime_df = pd.DataFrame(list(zip([np.mean(runs)],[np.std(runs)])), columns=["Mean", "Std"])
    wandb_run.log({'Runtime/Sampling': runtime_df})
    print("Runtime evaluation completed and logged to WandB.")

