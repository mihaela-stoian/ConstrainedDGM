import pickle as pkl

import numpy as np
import pandas as pd
import torch
import wandb


from cdgm.constraints_code.parser import parse_constraints_file
from evaluation import stasy_utility_detection
from evaluation.visual import feature_correlation



def eval_synthetic_data(args, use_case, real_data, generated_data, columns, problem_type, target_utility, target_utility_size, target_detection, log_wandb, wandb_run, unrounded_generated_data_for_cons_sat=None, eval_real=False):

    feature_correlation(real_data["test"], log_wandb, wandb_run, "correlation_real")

    print('Get feat. corr. on unrounded_generated_data_for_cons_sat ')
    results_cons = []
    if unrounded_generated_data_for_cons_sat is None:
        unrounded_generated_data_for_cons_sat = generated_data
    for i in range(len(unrounded_generated_data_for_cons_sat["test"])):
        generated_data_i = unrounded_generated_data_for_cons_sat["test"][i]
        feature_correlation(generated_data_i, log_wandb, wandb_run, f"INFERENCE/correlation/syn_{i}")

    print('Compute the stasy utility scores')
    utility_real, utility_syn = stasy_utility_detection.compute_utility_scores(real_data, generated_data["train"],  problem_type, columns, target_utility, target_utility_size, eval_real=eval_real)

    metric_map = {
        "binary_classification": ["binary_f1", "roc_auc", "weighted_f1", "accuracy"],
        "multiclass_classification": ["macro_f1", "roc_auc", "weighted_f1", "accuracy"],
        "regression": ["r2", "explained_variance", "MAE", "RMSE"],
    }

    metrics = metric_map[problem_type]

    if eval_real:
        utility_real[3].columns = [f"Utility_real.{m}" for m in metrics]
    utility_syn[3].columns = [f"Utility_syn.{m}" for m in metrics]

    print('Compute the stasy detection scores')
    a_det, b_det, c_det, detection_avg = stasy_utility_detection.compute_detection_scores(real_data, generated_data, "binary_classification", columns, target_detection)
    detection_avg.columns = ["Detection.binary_f1", "Detection.roc_auc", "Detection.weighted_f1", "Detection.accuracy"]


    print('Finished utility+det. eval')

    # all_avg = pd.concat([quality_avg, utility_syn[3], detection_avg], axis=1)
    all_avg = pd.concat([utility_syn[3], detection_avg], axis=1)
    all_avg = all_avg.rename(index={"0": 'Mean', "1":"Std"})

    save_id = 'INFERENCE'
    pkl.dump(results_cons, open(f'{args.path}/{save_id}_results_cons.pkl', 'wb'), -1)
    pkl.dump(generated_data, open(f'{args.path}/{save_id}_generated_data.pkl', 'wb'), -1)
    pkl.dump(utility_syn, open(f'{args.path}/{save_id}_utility_scores.pkl', 'wb'), -1)
    pkl.dump((a_det, b_det, c_det, detection_avg), open(f'{args.path}/{save_id}_detection_scores.pkl', 'wb'), -1)
    pkl.dump(all_avg, open(f'{args.path}/{save_id}_all_avg_scores.pkl', 'wb'), -1)
    with open(f'{args.path}/final_{save_id}_all_avg_scores.csv', 'w') as f:
        a = ''
        b = ''
        for key in all_avg.columns:
            a += f'{key},'
            b += f'{all_avg[key][0]},'
        a += 'model_path\n'
        a += 'model_path\n'
        b += f'{args.path}/'
        f.write(a+b)


    if log_wandb:
        wandb_run.log({f"INFERENCE/aggregated/eval_metrics_syn": wandb.Table(dataframe=all_avg)})
        if eval_real:
            wandb_run.log({f"INFERENCE/aggregated/utility_real": wandb.Table(dataframe=utility_real[3])})
           

def constraints_sat_check(args, real_data, generated_data, log_wandb):
    # Note: the ordering of the labels does not matter here
    _, constraints = parse_constraints_file(args.constraints_file)
    gen_sat_check(args, generated_data, constraints, log_wandb)
    real_sat_check(args, real_data, constraints, log_wandb)

def gen_sat_check(args, generated_data, constraints, log_wandb):
    sat_rate_per_constr = {i:[] for i in range(len(constraints))}
    percentage_cons_sat_per_pred = []
    percentage_of_samples_sat_constraints = []
    percentage_of_constr_violated_at_least_once = []

    for _, gen_data in enumerate(generated_data["train"]):
        samples_sat_constr = torch.ones(gen_data.shape[0]) == 1.
        num_cons_sat_per_pred = torch.zeros(gen_data.shape[0])
        num_constr_violated_at_least_once = 0.
        # gen_data = gen_data.iloc[:, :-1].to_numpy()
        gen_data = torch.tensor(gen_data.to_numpy())
        for j, constr in enumerate(constraints):
            sat_per_datapoint = constr.single_inequality.check_satisfaction(gen_data)
            num_cons_sat_per_pred += sat_per_datapoint*1.
            num_constr_violated_at_least_once += 0. if sat_per_datapoint.all() else 1.
            sat_rate = sat_per_datapoint.sum()/len(sat_per_datapoint)
            # print('Synth sat_rate is', sat_rate, sat_per_datapoint.sum(), len(sat_per_datapoint), sat_per_datapoint)
            sat_rate_per_constr[j].append(sat_rate)
            samples_sat_constr = samples_sat_constr & sat_per_datapoint
            # print('samples_violating_constr:', samples_violating_constr.sum())
        percentage_cons_sat_per_pred.append(np.array(num_cons_sat_per_pred/len(constraints)).mean())
        percentage_of_samples_sat_constraints.append(sum(samples_sat_constr) / len(samples_sat_constr))
        percentage_of_constr_violated_at_least_once.append(num_constr_violated_at_least_once/len(constraints))
    sat_rate_per_constr = {i:[sum(sat_rate_per_constr[i])/len(sat_rate_per_constr[i]) * 100.0] for i in range(len(constraints))}
    percentage_cons_violations_per_pred = 100.0-sum(percentage_cons_sat_per_pred)/len(percentage_cons_sat_per_pred) * 100.0
    percentage_of_samples_violating_constraints = 100.0-sum(percentage_of_samples_sat_constraints)/len(percentage_of_samples_sat_constraints) * 100.0
    percentage_of_constr_violated_at_least_once = sum(percentage_of_constr_violated_at_least_once)/len(percentage_of_constr_violated_at_least_once) * 100.0
    print('SYNTH', 'sat_rate_per_constr', sat_rate_per_constr)
    print('SYNTH', 'percentage_of_samples_violating_at_least_one_constraint', percentage_of_samples_violating_constraints)
    print('SYNTH', 'percentage_cons_violations_per_pred', percentage_cons_violations_per_pred)
    print('SYNTH', 'percentage_of_constr_violated_at_least_once', percentage_of_constr_violated_at_least_once)

    sat_rate_per_constr = pd.DataFrame(sat_rate_per_constr, columns=list(range(len(constraints))))
    if log_wandb:
        wandb.log({f"INFERENCE/synth_constr_sat": wandb.Table(dataframe=sat_rate_per_constr)})

    synth_constr_eval_metrics = pd.DataFrame({'percentage_of_samples_violating_constraints': [percentage_of_samples_violating_constraints],
                                              'percentage_cons_violations_per_pred': percentage_cons_violations_per_pred,
                                              'percentage_of_constr_violated_at_least_once': percentage_of_constr_violated_at_least_once},
                                             columns=['percentage_of_samples_violating_constraints', 'percentage_cons_violations_per_pred', 'percentage_of_constr_violated_at_least_once'])
    if log_wandb:
        wandb.log({f"INFERENCE/synth_constr_eval_metrics": wandb.Table(dataframe=synth_constr_eval_metrics)})

    return sat_rate_per_constr, percentage_of_samples_violating_constraints, synth_constr_eval_metrics


def real_sat_check(args, real_data, constraints, log_wandb):
    sat_rate_per_constr = {i: [] for i in range(len(constraints))}
    percentage_of_samples_sat_constraints = []

    real_data = real_data["train"]
    samples_sat_constr = torch.ones(real_data.shape[0]) == 1.
    # real_data = real_data.iloc[:, :-1].to_numpy()
    real_data = torch.tensor(real_data.to_numpy())

    for j, constr in enumerate(constraints):
        sat_per_datapoint = constr.single_inequality.check_satisfaction(real_data)
        sat_rate = sat_per_datapoint.sum() / len(sat_per_datapoint)
        # print('Real sat_rate is', sat_rate, sat_per_datapoint.sum(), len(sat_per_datapoint), sat_per_datapoint)
        sat_rate_per_constr[j].append(sat_rate)
        samples_sat_constr = samples_sat_constr & sat_per_datapoint

    percentage_of_samples_sat_constraints.append(sum(samples_sat_constr)/len(samples_sat_constr))
    sat_rate_per_constr = {i: [sum(sat_rate_per_constr[i]) / len(sat_rate_per_constr[i]) * 100.0] for i in
                           range(len(constraints))}
    percentage_of_samples_violating_constraints = 100.0 - sum(percentage_of_samples_sat_constraints) / len(
        percentage_of_samples_sat_constraints) * 100.0
    print('REAL', 'sat_rate_per_constr', sat_rate_per_constr)
    print('REAL', 'percentage_of_samples_violating_constraints', percentage_of_samples_violating_constraints)

    sat_rate_per_constr = pd.DataFrame(sat_rate_per_constr, columns=list(range(len(constraints))))
    if log_wandb:
        wandb.log({f"INFERENCE/real_constr_sat": wandb.Table(dataframe=sat_rate_per_constr)})

    percentage_of_samples_violating_constraints = pd.DataFrame({'real_percentage_of_samples_violating_constraints': [percentage_of_samples_violating_constraints]}, columns=['real_percentage_of_samples_violating_constraints'])
    if log_wandb:
        wandb.log({f"INFERENCE/real_percentage_of_samples_violating_constr": wandb.Table(
        dataframe=percentage_of_samples_violating_constraints)})

    return sat_rate_per_constr, percentage_of_samples_violating_constraints