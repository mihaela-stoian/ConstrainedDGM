import torch
import wandb
import pandas as pd
import numpy as np
import pickle as pkl

from sdmetrics.single_table import MulticlassMLPClassifier, MulticlassDecisionTreeClassifier, LinearRegression, \
    MLPRegressor

#from evaluation.synthcity import eval_synthcity
#from evaluation.sdv import eval_sdv
from evaluation.constraints import constraint_satisfaction
from evaluation.visual import feature_correlation
from utils import metrics_to_wandb

#from synthcity.metrics import eval_detection, eval_performance, eval_statistical
from evaluation import stasy_utility_detection, synthcity_quality
from sdv.metrics.tabular import LogisticDetection, SVCDetection, BinaryAdaBoostClassifier, BinaryMLPClassifier, \
    BinaryDecisionTreeClassifier
from synthcity.plugins.core.dataloader import GenericDataLoader

from constraints_code.compute_sets_of_constraints import compute_sets_of_constraints
from constraints_code.correct_predictions import check_all_constraints_sat
from constraints_code.parser import parse_constraints_file
from constraints_code.feature_orderings import set_ordering


def eval_quality(real_data, generated_data_list, columns, target_column):

    real_data_loader = GenericDataLoader(real_data, target_column=target_column)
    #precision, coverage, authenticity = [], [], []
    results = []

    for generated_data in generated_data_list:
        #gen_data = pd.DataFrame(data=generated_data, columns=columns)
        #gen_data = pd.concat([gen_data, gen_data])
        gen_data_loader = GenericDataLoader(generated_data, target_column=target_column)

        quality_evaluator = synthcity_quality.AlphaPrecision()
        qual_res = quality_evaluator.evaluate(gen_data_loader, real_data_loader)
        qual_res = {
            k: v for (k, v) in qual_res.items() if "naive" in k
        }  # use the naive implementation of AlphaPrecision
        #qual_score = np.mean(list(qual_res.values()))
        results.append([qual_res["delta_precision_alpha_naive"], qual_res["delta_coverage_beta_naive"], qual_res["authenticity_naive"]])
    quality_scores = pd.DataFrame(results, columns=["Precision_alpha", "Coverage_beta", "Authenticity"])
    # quality_scores = pd.DataFrame([quality_scores.mean(axis=0), quality_scores.std(axis=0)])
    quality_scores = pd.DataFrame([quality_scores.mean(axis=0)])
    quality_avg = pd.DataFrame(quality_scores.mean(axis=1), columns = ['Quality'])
    return quality_scores, quality_avg





def eval_synthetic_data(args, use_case, real_data, generated_data, columns, problem_type, target_utility, target_utility_size, target_detection, log_wandb, wandb_run, unrounded_generated_data_for_cons_sat=None, eval_real=False):

    feature_correlation(real_data["test"], log_wandb, wandb_run, "correlation_real")

    print('Check (manual) constraint sat. and get feat. corr. on unrounded_generated_data_for_cons_sat ')
    results_cons = []
    if unrounded_generated_data_for_cons_sat is None:
        unrounded_generated_data_for_cons_sat = generated_data
    for i in range(len(unrounded_generated_data_for_cons_sat["test"])):
        generated_data_i = unrounded_generated_data_for_cons_sat["test"][i]
        features = generated_data_i.iloc[:, :-1].to_numpy()
        cons_rate, batch_rate, ind_score = constraint_satisfaction(features, use_case)
        results_cons.append([ind_score.mean(), batch_rate,  cons_rate])
        #gen_data_i = pd.DataFrame(generated_data_i, columns=columns)
        feature_correlation(generated_data_i, log_wandb, wandb_run, f"INFERENCE/correlation/syn_{i}")

    results_cons = pd.DataFrame(results_cons, columns=["mean_sat_score", "cons_sat_rate_all", "cons_sat_for_individual_example"])
    cons_avg = pd.DataFrame([results_cons.mean(axis=0), results_cons.std(axis=0)], index=["Mean", "Std"])

    # print('Compute the synthcity quality scores')
    # quality_scores, quality_avg= eval_quality(real_data["train"], generated_data["train"], columns, target_utility)

    # df_quality = metrics_to_wandb(quality_scores)
    # print(df_quality)

    print('Compute the stasy utility scores')
    utility_real, utility_syn = stasy_utility_detection.compute_utility_scores(real_data, generated_data["train"],  problem_type, columns, target_utility, target_utility_size, eval_real=eval_real)

    if problem_type=="binary_classification":
        if eval_real:
            utility_real[3].columns = ["Utility_real.binary_f1", "Utility_real.roc_auc", "Utility_real.weighted_f1", "Utility_real.accuracy"]
        utility_syn[3].columns = ["Utility_syn.binary_f1", "Utility_syn.roc_auc", "Utility_syn.weighted_f1", "Utility_syn.accuracy"]
    elif problem_type=="multiclass_classification":
        if eval_real:
            utility_real[3].columns = ["Utility_real.macro_f1", "Utility_real.roc_auc", "Utility_real.weighted_f1", "Utility_real.accuracy"]
        utility_syn[3].columns = ["Utility_syn.macro_f1", "Utility_syn.roc_auc", "Utility_syn.weighted_f1", "Utility_syn.accuracy"]
    elif problem_type=="regression":
        if eval_real:
            utility_real[3].columns = ["Utility_real.r2", "Utility_real.explained_variance", "Utility_real.MAE", "Utility_real.RMSE"]
        utility_syn[3].columns = ["Utility_syn.r2", "Utility_syn.explained_variance", "Utility_syn.MAE", "Utility_syn.RMSE"]

    print('Compute the stasy detection scores')
    a_det, b_det, c_det, detection_avg = stasy_utility_detection.compute_detection_scores(real_data, generated_data, "binary_classification", columns, target_detection)
    detection_avg.columns = ["Detection.binary_f1", "Detection.roc_auc", "Detection.weighted_f1", "Detection.accuracy"]

    # print('Finished quality+utility+det. eval using synthcity and stasy')
    print('Finished utility+det. eval using synthcity and stasy')

    # all_avg = pd.concat([quality_avg, utility_syn[3], detection_avg], axis=1)
    all_avg = pd.concat([utility_syn[3], detection_avg], axis=1)
    all_avg = all_avg.rename(index={"0": 'Mean', "1":"Std"})

    save_id = 'INFERENCE'
    pkl.dump(results_cons, open(f'{args.exp_path}/{save_id}_results_cons.pkl', 'wb'), -1)
    pkl.dump(cons_avg, open(f'{args.exp_path}/{save_id}_cons_avg.pkl', 'wb'), -1)
    pkl.dump(generated_data, open(f'{args.exp_path}/{save_id}_generated_data.pkl', 'wb'), -1)
    # pkl.dump((quality_scores, quality_avg), open(f'{args.exp_path}/{save_id}_quality_scores.pkl', 'wb'), -1)
    pkl.dump(utility_syn, open(f'{args.exp_path}/{save_id}_utility_scores.pkl', 'wb'), -1)
    pkl.dump((a_det, b_det, c_det, detection_avg), open(f'{args.exp_path}/{save_id}_detection_scores.pkl', 'wb'), -1)
    pkl.dump(all_avg, open(f'{args.exp_path}/{save_id}_all_avg_scores.pkl', 'wb'), -1)
    with open(f'{args.exp_path}/final_{save_id}_all_avg_scores.csv', 'w') as f:
        a = ''
        b = ''
        for key in all_avg.columns:
            a += f'{key},'
            b += f'{all_avg[key][0]},'
        a += 'model_path\n'
        a += 'model_path\n'
        b += f'{args.exp_path}/'
        f.write(a+b)


    if log_wandb:
        wandb_run.log({f"INFERENCE/constraints/all": wandb.Table(dataframe=results_cons)})
        wandb_run.log({f"INFERENCE/aggregated/constraints": wandb.Table(dataframe=cons_avg)})

        wandb_run.log({f"INFERENCE/aggregated/eval_metrics_syn": wandb.Table(dataframe=all_avg)})

        # wandb_run.log({f"INFERENCE/quality/quality": wandb.Table(dataframe=quality_scores)})
        wandb_run.log({f"INFERENCE/utility/synthetic/best_f1_scores": wandb.Table(dataframe=utility_syn[0])})
        wandb_run.log({f"INFERENCE/utility/synthetic/best_weighted_scores": wandb.Table(dataframe=utility_syn[1])})
        wandb_run.log({f"INFERENCE/utility/synthetic/best_auroc_scores": wandb.Table(dataframe=utility_syn[2])})

        wandb_run.log({f"INFERENCE/detection/best_f1_scores": wandb.Table(dataframe=a_det)})
        wandb_run.log({f"INFERENCE/detection/best_weighted_scores": wandb.Table(dataframe=b_det)})
        wandb_run.log({f"INFERENCE/detection/best_auroc_scores": wandb.Table(dataframe=c_det)})

        if eval_real:
            wandb_run.log({f"INFERENCE/aggregated/utility_real": wandb.Table(dataframe=utility_real[3])})
            wandb_run.log({f"INFERENCE/utility/real/best_f1_scores": wandb.Table(dataframe=utility_real[0])})
            wandb_run.log({f"INFERENCE/utility/real/best_weighted_scores": wandb.Table(dataframe=utility_real[1])})
            wandb_run.log({f"INFERENCE/utility/real/best_auroc_scores": wandb.Table(dataframe=utility_real[2])})
 

def ml_efficacy(train_data, test_data, target_col, ctype='binary'):
    # bt = BinaryDecisionTreeClassifier()
    # ab = BinaryAdaBoostClassifier()
    # mlp = BinaryMLPClassifier()
    if ctype == 'binary':
        classifiers = [BinaryAdaBoostClassifier(), BinaryMLPClassifier(), BinaryDecisionTreeClassifier()]
    elif ctype == 'multiclass':
        classifiers = [MulticlassMLPClassifier(), MulticlassDecisionTreeClassifier()]
    elif ctype == 'regression':
        classifiers = [LinearRegression(), MLPRegressor()]    # cls_names = ["BinaryAdaBoostClassifier()", "BinaryMLPClassifier()", "BinaryDecisionTreeClassifier"]
    scores = []
    for i in range(len(classifiers)):
        classifier = classifiers[i]
        avg_sc = 0
        for j in range(5):
            sc = classifier.compute(test_data, train_data, target=target_col)
            avg_sc += sc
        scores.append(avg_sc / 5)

    return scores

def ml_detection_metrics(real_data, syn_data):
    lr_model = LogisticDetection()
    svc_model = SVCDetection()
    lr = lr_model.compute(real_data, syn_data)
    svc = svc_model.compute(real_data, syn_data)

    print('Reporting detection scores as 1-|score-0.5|*2 at inference time, higher is better')
    lr = 1 - np.abs(lr-0.5)*2
    svc = 1 - np.abs(svc-0.5)*2
    return lr, svc


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


def sdv_eval_synthetic_data(args, use_case, real_data, generated_data, columns, problem_type, target_utility, target_detection, log_wandb, wandb_run):
    lr_scores, svc_scores, adas, mlps, dts = [], [], [], [], []
    ctype = 'binary'
    ctype_stasy = 'binary_classification'
    if args.use_case == 'faults':
        ctype = 'multiclass'
        ctype_stasy = 'multiclass_classification'
    elif args.use_case == 'news':
        ctype = 'regression'
        ctype_stasy = 'regression'
    for i in range(len(generated_data["test"])):
        # if args.quick_eval:
        #     syn_data_shape = args.num_quick_eval_samples
        # else:
        #     syn_data_shape = test_data.shape[0]
        syn_data_for_detection = generated_data['test'][i]
        syn_data_for_detection = pd.DataFrame(syn_data_for_detection, columns=args.train_data_cols).astype(float)
        syn_data_for_detection[target_utility] = syn_data_for_detection[target_utility].astype(args.dtypes[-1])
        real_data_test = pd.DataFrame(real_data['test'], columns=args.train_data_cols).astype(float)
        real_data_test[target_utility] = real_data_test[target_utility].astype(args.dtypes[-1])

        # lr_score, svc_score = ml_detection_metrics(real_data_test[:syn_data_for_detection.shape[0]], syn_data_for_detection)
        lr_score, svc_score = ml_detection_metrics(real_data_test, syn_data_for_detection)
        lr_scores.append(lr_score)
        svc_scores.append(svc_score)
        print(f'detection run {i}', lr_score, svc_score)


        print('Computing utility for synth data with same shape as train set, using test set to get generalisation performance')
        # if args.quick_eval:
        #     syn_data_shape = args.num_quick_eval_samples
        # else:
        #     syn_data_shape = X_train_.shape[0]

        syn_data_for_efficacy = generated_data['train'][i]
        syn_data_for_efficacy = pd.DataFrame(syn_data_for_efficacy, columns=args.train_data_cols).astype(float)
        syn_data_for_efficacy[target_utility] = syn_data_for_efficacy[target_utility].astype(args.dtypes[-1])

        ml_eff_score = ml_efficacy(syn_data_for_efficacy, real_data_test, target_utility, ctype=ctype)
        adas.append(ml_eff_score[0])
        mlps.append(ml_eff_score[1])
        lr_scores = [sum(lr_scores) / len(lr_scores)]
        svc_scores = [sum(svc_scores) / len(svc_scores)]
        adas = [sum(adas) / len(adas)]
        mlps = [sum(mlps) / len(mlps)]
        if len(ml_eff_score)>2:
            dts.append(ml_eff_score[2])
            dts = [sum(dts) / len(dts)]
            scores = {'LR': lr_scores, 'SVC': svc_scores, 'Ada': adas, 'MLP': mlps, 'DT': dts}

        else:
            scores = {'LR': lr_scores, 'SVC': svc_scores, 'Cls1': adas, 'Cls2': mlps}

    df = pd.DataFrame(data=scores)
    if log_wandb:
        wandb.log({"final_performance": wandb.Table(data=df)})  # https://docs.wandb.ai/ref/python/data-types/table

    df.to_csv(f'{args.exp_path}/final_sdv.csv', mode='a')
    for col in scores:
        values = scores[col]
        print(col)
        print('mean:', np.mean(values))
        print('std:', np.std(values))


