import pandas as pd 
import numpy as np
import pickle

from constraints_code.classes import Constraint
from constraints_code.parser import parse_constraints_file

use_case = "botnet"
partition = "test"
data = pd.read_csv(f"../data/{use_case}/{partition}_data.csv")
constraints_file = f'../data/{use_case}/{use_case}_constraints.txt'

_, constraints = parse_constraints_file(constraints_file)


feat_dict = {}
cons_dict = {}
for i,c in enumerate(constraints):
    cons_dict[i] = {'num_f':0, 'num_f+':0, 'num_f-':0}
    c: Constraint

    body = c.get_body_atoms()
    cons_dict[i]['num_f'] += len(body)
    for atom in body:
        feat_id = atom.variable.id

        if atom.positive_sign:
            cons_dict[i]['num_f+'] += 1
        else:
            cons_dict[i]['num_f-'] += 1

        if feat_id not in feat_dict:
            feat_dict[feat_id] = {'in_cons':[], 'is_reducible': False, 'pos_occ_in_cons':[], 'neg_occ_in_cons':[], 'num_reducible_pairs':0}
        feat_dict[feat_id]['in_cons'].append(i)
        if atom.positive_sign:
            feat_dict[feat_id]['pos_occ_in_cons'].append(i)
        else:
            feat_dict[feat_id]['neg_occ_in_cons'].append(i)
        print(atom.readable())

# determine if feat is reducible
for feat_id in feat_dict:
    temp = feat_dict[feat_id]
    l1 = len(temp['pos_occ_in_cons'])
    l2 = len(temp['neg_occ_in_cons'])
    feat_dict[feat_id]['num_reducible_pairs'] = l1*l2
    if feat_dict[feat_id]['num_reducible_pairs']:
        feat_dict[feat_id]['is_reducible'] = True


s = ''
print("num constraints", len(constraints))
s += f'{len(constraints)} & '

print('num feats out of total num feats in dataset', f'{len(feat_dict)}/{data.shape[-1]}')
s += f'{len(feat_dict)}/{data.shape[-1]} & '

temp = np.array([cons_dict[i]['num_f'] for i in cons_dict])
print('avg. num feats per cons', temp, temp.mean())
s += f'{temp.mean():.2f} & '

temp = np.array([cons_dict[i]['num_f+'] for i in cons_dict])
print('avg num POS feats per cons', temp, temp.mean())
s += f'{temp.mean():.2f} & '

temp = np.array([cons_dict[i]['num_f-'] for i in cons_dict])
print('avg num NEG feats per cons', temp, temp.mean())
s += f'{temp.mean():.2f} & '

temp = np.array([len(feat_dict[feat_id]['in_cons']) for feat_id in feat_dict])
print('avg. num of constraints in which a feat appears', temp, temp.mean())
s += f'{temp.mean():.2f} & '

temp = np.array([feat_dict[feat_id]['is_reducible'] for feat_id in feat_dict])
print('num of reducible feats', temp, temp.sum())
s += f'{temp.sum()} & '

temp = np.array([feat_dict[feat_id]['num_reducible_pairs'] for feat_id in feat_dict if feat_dict[feat_id]['is_reducible']])
result = temp.mean() if len(temp) else '-'
print('avg. num of reducible pairs per feat', temp, result)
if len(temp):
    s += f'{temp.mean():.2f}  '
else:
    s += f'- '


print(s)