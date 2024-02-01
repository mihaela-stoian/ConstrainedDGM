
import matplotlib.pyplot as plt
import pandas as pd
import wandb

def feature_correlation(data, log_wandb, wandb_run, path):
    fig = plt.figure()
    plt.matshow(data.corr())
    plt.colorbar()
    plt.tight_layout()
    if log_wandb:
        wandb_run.log({path: plt})
