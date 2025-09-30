from argparse import ArgumentParser
from pathlib import Path
from timeit import default_timer as timer
import pandas
import torch
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation, DAG
from castle.algorithms import PC, GAE, DAG_GNN, MCSL
import numpy as np
import pickle as pkl

def get_args():
    args = ArgumentParser()
    args.add_argument("use_case", default='url', type=str)
    args.add_argument("model", default='gae', type=str,
                      choices=['pc','gae','dag_gnn','mcsl'])
    args.add_argument("--epochs", default=1, type=int)
    return args.parse_args()


def model_selection(args):
    if args.model == 'pc':
        return PC()

    elif args.model == 'gae':
        return GAE(epochs=args.epochs)

    elif args.model == 'dag_gnn':
        return DAG_GNN(epochs=args.epochs)

    elif args.model == 'mcsl':
        return MCSL()


def main(args):
    start = timer()
    true_causal_matrix = np.zeros((2, 2))
    if args.use_case == "lcld":
        X = pandas.read_csv(f'../data/{args.use_case}/tiny/train_data.csv')
    else:
        X = pandas.read_csv(f'../data/{args.use_case}/train_data.csv')
    columns = X.columns
    X = X.to_numpy()

    # structure learning
    causal_discovery_model = model_selection(args)
    causal_discovery_model.learn(X, columns=columns)

    # plot predict_dag and true_dag
    GraphDAG(causal_discovery_model.causal_matrix, true_causal_matrix, 'result')

    pathname = Path('outputs')
    pathname.mkdir(parents=True, exist_ok=True)
    pathname = pathname / f'causal-matrix_{args.use_case}_{args.model}.pkl'
    print(causal_discovery_model.causal_matrix)
    with open(pathname, 'wb') as f:
        saved_data = torch.tensor(causal_discovery_model.causal_matrix)
        pkl.dump(saved_data, f, -1)
    end = timer()
    print("########### Runtime ###############")
    print(end-start)

if __name__ == '__main__':
    args = get_args()
    main(args)