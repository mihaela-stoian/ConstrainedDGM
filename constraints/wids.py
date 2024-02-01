import numpy as np
import autograd.numpy as anp
import pickle
import joblib 
import os.path
from typing import List
import pandas as pd


def evaluate_numpy_wids(x):
    tol = 1e-2

    g_min_max = []
    for i in range(33, 94, 2):
        cons = x[:,i + 1] - x[:,i]
        g_min_max.append(cons)
    print(len(g_min_max))
    constraints = anp.column_stack(g_min_max)
    return constraints