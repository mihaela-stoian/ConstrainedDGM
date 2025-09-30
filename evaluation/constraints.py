import numpy as np
from constraints.url import evaluate_numpy_url
from constraints.botnet import evaluate_numpy_botnet
from constraints.lcld import evaluate_numpy_lcld
from constraints.wids import evaluate_numpy_wids
from constraints.news import evaluate_numpy_news
from constraints.faults import evaluate_numpy_faults
from constraints.heloc import evaluate_numpy_heloc, INFINITY
import pandas as pd


def constraint_satisfaction(data, use_case):
    tol=1e-2
    if use_case=="url":
        cons = evaluate_numpy_url(data)

    if use_case=="botnet":
        cons = evaluate_numpy_botnet(data)
        # return -1, -1, -1 # TODO: a file is missing and causes an error in the function above
    if use_case=="lcld":
        cons = evaluate_numpy_lcld(data)

    if use_case=="wids":
        cons = evaluate_numpy_wids(data)

    if use_case=="heloc":
        cons = evaluate_numpy_heloc(data)
        # tol = 1  # TODO: why is this set to 1 in the unified_extended branch?

    if use_case=="news":
        cons = evaluate_numpy_news(data)

    if use_case=="faults":
        cons = evaluate_numpy_faults(data)

    # idx = np.squeeze(np.argwhere(cons[:,4]>0.0001))
    mask_non_missing = cons != -INFINITY
    count = np.count_nonzero((cons<=tol) & mask_non_missing, axis=1)
    num_constraints_non_missing_vals = mask_non_missing.sum(axis=1)
    # cons_rate = np.mean(count/cons.shape[1])*100
    cons_rate = np.mean(count/num_constraints_non_missing_vals)*100

    count_batch = np.count_nonzero((cons<=tol) & mask_non_missing, axis=0)
    num_datapoints_non_missing_vals = mask_non_missing.sum(axis=0)
    batch_rate = np.mean(count_batch/num_datapoints_non_missing_vals)*100

    cons[~mask_non_missing] = 0.
    # ind_score = np.mean(cons, axis=0)
    ind_score = np.sum(cons, axis=0)/num_datapoints_non_missing_vals

    return cons_rate, batch_rate, ind_score
