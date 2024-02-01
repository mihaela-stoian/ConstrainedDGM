import time
from typing import List

import torch

from constraints_code.classes import Variable, Constraint, eval_atoms_list, get_missing_mask
from constraints_code.compute_sets_of_constraints import compute_sets_of_constraints, get_pos_neg_x_constr
from constraints_code.parser import parse_constraints_file
from constraints_code.feature_orderings import set_random_ordering
import pandas as pd
INFINITY = torch.inf


def get_constr_at_level_x(x, sets_of_constr):
    for var in sets_of_constr:
        if var.id == x.id:
            return sets_of_constr[var]


def get_partial_x_correction(x: Variable, x_positive: bool, x_constraints: List[Constraint],
                             preds: torch.Tensor, epsilon=1e-12) -> torch.Tensor:
    # if x.id == 25:
    #     print('DEBUG!!!')
    if len(x_constraints) == 0:
        if x_positive:
            return -INFINITY
        else:
            return INFINITY

    # dependency_complements = [preds[:, x.id]] # Note: the original prediction cannot be here, as this function gets called twice: for pos and neg occurences of x!
    # so using the original value of x here can undo the partial correction for pos occurrences of x!
    dependency_complements = [] # size: num_atom_dependencies x B (i.e. B=batch size)
    mask_sat_batch_elem = None
    for constr in x_constraints:
        # print(constr.readable(), 'LLLLLLLLLLLLLLLL')
        mask_sat_batch_elem = None  # NOTE: the mask should be reset when a new constraint is considered, regardless of its type (ineq or disj of ineq!)
        # print(constr.single_inequality.readable(), 'AAAAAA')
        if len(constr.inequality_list) == 1:
            complement_body_atoms, x_atom, constr_constant, is_strict_inequality = constr.single_inequality.get_x_complement_body_atoms(x)
            # print('first branch', [e.readable() for e in complement_body_atoms], x_atom, constr_constant, is_strict_inequality)
        else:
            (complement_body_atoms, x_atom, constr_constant, is_strict_inequality), mask_sat_batch_elem = constr.single_inequality.get_x_complement_body_atoms(x, preds)
            # print('second branch', preds[:,2])
            if x_atom is None:
                # print('atom is none')
                # print(constr.readable(), 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAaaa!!!!!!!!!!!!!!1')
                continue
            # else:
                # print('first branch', [e.readable() for e in complement_body_atoms], x_atom, constr_constant, is_strict_inequality)

                # print(constr.readable(), 'MMMMMMMMMMMMMMMMMMMM!!!!!!!!!!!!!!1')

        # get the coefficient of variable x in constr
        x_coeff = x_atom.coefficient  # do not use self.get_signed_coefficient() here!
        # print(x_coeff,'x coeff')
        # print(constr_constant, '@')
        # if x is y1 and inequality is -y1>0, then add 0+bias to dependency_complements
        if len(complement_body_atoms) == 0:  # TODO: assumes that right hand side of ineq is 0, need to consider cases where the constant/bias is non-zero
            evaluated_complement_body = torch.zeros(preds.shape[0])  # shape (B,)
            # print('first br', evaluated_complement_body)
            # continue
        else:
            # evaluate the body of constr, after eliminating x occurrences from it
            evaluated_complement_body = eval_atoms_list(complement_body_atoms, preds)  # shape (B,)
            # print('second br', evaluated_complement_body)

        # print('x_coeff', x_coeff)
        # print('evaluated_complement_body', evaluated_complement_body)
        # weigh the evaluated complement body by the -1/(coefficient of x) if x occurs positively in constr
        # and by 1/(coefficient of x) if x occurs negatively in constr
        x_weight = -1. / x_coeff if x_positive else 1. / x_coeff
        evaluated_complement_body *= x_weight
        # print('evaluated_complement_body after weighing', evaluated_complement_body)

        # now add the weighted bias:
        evaluated_complement_body += constr_constant * (-x_weight)

        # then add/subtract epsilon if the ineq is strict: +epsilon if positive x, -epsilon otherwise
        if is_strict_inequality:
            # print(is_strict_inequality)
            evaluated_complement_body += epsilon if x_positive else -epsilon

        # if the constr is a disj of inqs, mask out the batch elements for which mask_sat_batch_elem is True
        if mask_sat_batch_elem is not None:
            evaluated_complement_body[mask_sat_batch_elem] += INFINITY * x_weight

        dependency_complements.append(evaluated_complement_body)

        # print('evaluated_complement_body after adding bias', evaluated_complement_body, dependency_complements)

    # if x.id == 25:
    #     print('END DEBUG!!!')

    if len(dependency_complements) > 1:
        dependency_complements = torch.stack(dependency_complements, dim=1)
    elif len(dependency_complements) == 1:
        dependency_complements = dependency_complements[0].unsqueeze(1)
    else:
        return -INFINITY if x_positive else INFINITY

    # print('@@@@@@@@@@@', complement_body_atoms, dependency_complements, len(dependency_complements), constr_constant)

    if x_positive:
        partially_corrected_val = dependency_complements.amax(dim=1)
        # print(partially_corrected_val, x.id, 'AAA')
    else:
        partially_corrected_val = dependency_complements.amin(dim=1)  # TODO: be careful here! the original value of h' to be corrected should not be added multiple times!! it shouldn't be added to the dependecy_complements
        # print(partially_corrected_val, 'BBB')

    # print('partially corrected val', partially_corrected_val)
    return partially_corrected_val


def get_final_x_correction(initial_x_val: torch.Tensor, pos_x_corrected: torch.Tensor,
                           neg_x_corrected: torch.Tensor) -> torch.Tensor:
    # print(initial_x_val, pos_x_corrected, neg_x_corrected, 'VAR25!!!')

    if type(pos_x_corrected) is not torch.Tensor:
        result_1 = initial_x_val
    else:
        # print(initial_x_val, pos_x_corrected)
        pos_x_corrected = pos_x_corrected.where(~(pos_x_corrected == INFINITY), initial_x_val)
        # keep_initial_pos_mask = pos_x_corrected.isinf()
        # pos_x_corrected1 = pos_x_corrected.clone()
        # pos_x_corrected2 = pos_x_corrected.clone()
        # pos_x_corrected3 = pos_x_corrected.clone()
        # pos_x_corrected1[keep_initial_pos_mask] = initial_x_val[keep_initial_pos_mask]
        # pos_x_corrected2 = torch.where(pos_x_corrected == INFINITY, initial_x_val, pos_x_corrected)
        # pos_x_corrected3 = pos_x_corrected.where(~(pos_x_corrected == INFINITY), initial_x_val)
        #
        # assert (pos_x_corrected1 == pos_x_corrected2).all(), (pos_x_corrected1, pos_x_corrected2, pos_x_corrected)
        # assert (pos_x_corrected1 == pos_x_corrected3).all(), (pos_x_corrected1, pos_x_corrected3, pos_x_corrected)
        result_1 = torch.cat([initial_x_val.unsqueeze(1), pos_x_corrected.unsqueeze(1)],dim=1).amax(dim=1)

    # print('result_1', result_1)
    if type(neg_x_corrected) is not torch.Tensor:
        result_2 = result_1
    else:
        # keep_initial_neg_mask = neg_x_corrected.isinf()
        # neg_x_corrected[keep_initial_neg_mask] = initial_x_val[keep_initial_neg_mask]
        neg_x_corrected = neg_x_corrected.where(~(neg_x_corrected == INFINITY), initial_x_val)
        result_2 = torch.cat([result_1.unsqueeze(1), neg_x_corrected.unsqueeze(1)],dim=1).amin(dim=1)
    # print('result_2', result_2)
    # print()
    # print(result_1, 'CCC')
    # print(result_2, 'CCC')

    return result_2


def correct_preds(preds: torch.Tensor, ordering: List[Variable], sets_of_constr: {Variable: List[Constraint]}):
    # given a NN's preds [h1, h2, .., hn],
    # an ordering of the n variables and
    # a set of constraints computed at each variable w.r.t. descending order of the provided ordering
    # correct the preds according to the constraints in ascending order w.r.t. provided ordering
    corrected_preds = preds.clone()

    for x in ordering:
        pos = x.id
        x_constr = get_constr_at_level_x(x, sets_of_constr)
        if len(x_constr) == 0:
            continue
        # print(x.id, [e.readable() for e in x_constr], 'KKKK')  # .readable()],'@@@')
        pos_x_constr, neg_x_constr = get_pos_neg_x_constr(x, x_constr)

        pos_x_corrected = get_partial_x_correction(x, True, pos_x_constr, preds)
        neg_x_corrected = get_partial_x_correction(x, False, neg_x_constr, preds)

        # print('pos', [e.readable() for e in pos_x_constr], preds[pos], pos_x_corrected)
        # print('neg', [e.readable() for e in neg_x_constr], preds[pos], neg_x_corrected)

        corrected_preds[:,pos] = get_final_x_correction(preds[:,pos], pos_x_corrected, neg_x_corrected)
        preds = corrected_preds.clone()
        corrected_preds = preds.clone()

    return corrected_preds


def example_predictions():
    # predictions = torch.tensor([-10.0, 5.0, -2.0, -9, 2, 20, -1]).unsqueeze(0)  # constraints_disj1: Corrected predictions tensor([-10.,  10.,  -2.,   3.,   1.,   2.,  -1.], grad_fn=<CopySlices>) True
    # y_38 y_37 y_21 y_3 y_31 y_26 y_28 y_2 y_19 y_25 y_23 y_13 y_20 y_1 y_4 y_5 y_6 y_7 y_8 y_9 y_10 y_11 y_12 y_14 y_15 y_16 y_17 y_0
    y_18 = y_22 = y_24 = y_27 = y_29 = y_30 = y_32 = y_33 = y_34 = y_35 = y_36 = -100
    y_38 = 0
    y_37 = 0
    y_21 = -1
    y_3 = 1
    y_31 = -3
    y_26 = 5
    y_28 = -3
    y_2 = 2
    y_19 = -4
    y_25 = -111 # change to -1 , value should be corrected to >0
    y_23 = 0
    y_13 = 0
    y_20 = 0
    y_1 = 0
    y_4 = 0
    y_5 = 0
    y_6 = 0
    y_7 = 0
    y_8 = 0
    y_9 = 0
    y_10 = 0
    y_11 = 0
    y_12 = 0
    y_14 = 0
    y_15 = 0
    y_16 = 0
    y_17 = 0
    y_0 = -14.0
    p1 = torch.tensor(eval('['+','.join(['y_'+str(i) for i in range(39)])+']')).unsqueeze(0)   # constraints_disj1: Corrected predictions tensor([-10.,  10.,  -2.,   3.,   1.,   2.,  -1.], grad_fn=<CopySlices>) True

    y_2 = -18
    p2 = torch.tensor(eval('['+','.join(['y_'+str(i) for i in range(39)])+']')).unsqueeze(0)
    predictions = torch.cat([p1,p1,p2,p1,p2],dim=0)
    return predictions


def example_predictions_lcld():
    # predictions = torch.tensor([-10.0, 5.0, -2.0, -9, 2, 20, -1]).unsqueeze(0)  # constraints_disj1: Corrected predictions tensor([-10.,  10.,  -2.,   3.,   1.,   2.,  -1.], grad_fn=<CopySlices>) True
    # y_38 y_37 y_21 y_3 y_31 y_26 y_28 y_2 y_19 y_25 y_23 y_13 y_20 y_1 y_4 y_5 y_6 y_7 y_8 y_9 y_10 y_11 y_12 y_14 y_15 y_16 y_17 y_0
    y_18 = y_22 = y_24 = y_27 = y_29 = y_30 = y_32 = y_33 = y_34 = y_35 = y_36 = -100
    y_38 = 0
    y_37 = 0
    y_21 = -1
    y_3 = 1
    y_31 = -3
    y_26 = 5
    y_28 = -3
    y_2 = 2
    y_19 = -4
    y_25 = -111 # change to -1 , value should be corrected to >0
    y_23 = 0
    y_13 = 0
    y_20 = 0
    y_1 = 0
    y_4 = 0
    y_5 = 0
    y_6 = 0
    y_7 = 0
    y_8 = 0
    y_9 = 0
    y_10 = 0
    y_11 = 0
    y_12 = 0
    y_14 = 0
    y_15 = 0
    y_16 = 0
    y_17 = 0
    y_0 = -14.0
    y_0= 0
    y_1 = 3
    p1 = torch.tensor(eval('['+','.join(['y_'+str(i) for i in range(2)])+']')).unsqueeze(0)   # constraints_disj1: Corrected predictions tensor([-10.,  10.,  -2.,   3.,   1.,   2.,  -1.], grad_fn=<CopySlices>) True

    y_1 = 34
    p2 = torch.tensor(eval('['+','.join(['y_'+str(i) for i in range(2)])+']')).unsqueeze(0)

    y_1 = 60.
    p3 = torch.tensor(eval('['+','.join(['y_'+str(i) for i in range(2)])+']')).unsqueeze(0)

    y_1 = 60.
    p4 = torch.tensor(eval('['+','.join(['y_'+str(i) for i in range(2)])+']')).unsqueeze(0)
    predictions = torch.cat([p1,p4,p2,p1,p3],dim=0)
    return predictions

def example_predictions_custom():
    y_0= -2
    y_1 = 2
    p1 = torch.tensor(eval('['+','.join(['y_'+str(i) for i in range(2)])+']')).unsqueeze(0)   # constraints_disj1: Corrected predictions tensor([-10.,  10.,  -2.,   3.,   1.,   2.,  -1.], grad_fn=<CopySlices>) True

    y_0 = 2
    y_1 = -2.
    p2 = torch.tensor(eval('['+','.join(['y_'+str(i) for i in range(2)])+']')).unsqueeze(0)

    predictions = torch.cat([p1,p2],dim=0)
    return predictions

def example_predictions_heloc():
    data = pd.read_csv(f"../data/heloc/test_data.csv")
    data = data.to_numpy().astype(float)
    return torch.tensor(data)


def compute_sat_stats(real_data, constraints, mask_out_missing_values=False):
    real_data = pd.DataFrame(real_data.detach().numpy())
    sat_rate_per_constr = {i: [] for i in range(len(constraints))}
    percentage_of_samples_sat_constraints = []

    samples_sat_constr = torch.ones(real_data.shape[0]) == 1.
    # real_data = real_data.iloc[:, :-1].to_numpy()
    real_data = torch.tensor(real_data.to_numpy())

    for j, constr in enumerate(constraints):
        sat_per_datapoint = constr.single_inequality.check_satisfaction(real_data)
        if mask_out_missing_values:
            missing_values_mask = get_missing_mask(constr.single_inequality.body, real_data)
        else:
            missing_values_mask = torch.ones(real_data.shape[0]) == 0.
        sat_per_datapoint[missing_values_mask] = True
        sat_rate = sat_per_datapoint[~missing_values_mask].sum() / (~missing_values_mask).sum()
        # print('Real sat_rate is', sat_rate, sat_per_datapoint.sum(), len(sat_per_datapoint), sat_per_datapoint)
        sat_rate_per_constr[j].append(sat_rate)
        # sat_rate_per_constr[j].append(sat_per_datapoint.sum() / len(sat_per_datapoint))
        samples_sat_constr = samples_sat_constr & sat_per_datapoint

    percentage_of_samples_sat_constraints.append(sum(samples_sat_constr) / len(samples_sat_constr))
    sat_rate_per_constr = {i: [sum(sat_rate_per_constr[i]) / len(sat_rate_per_constr[i]) * 100.0] for i in
                           range(len(constraints))}
    percentage_of_samples_violating_constraints = 100.0 - sum(percentage_of_samples_sat_constraints) / len(
        percentage_of_samples_sat_constraints) * 100.0
    print('REAL', 'sat_rate_per_constr', sat_rate_per_constr)
    print('REAL', 'percentage_of_samples_violating_constraints', percentage_of_samples_violating_constraints)

    sat_rate_per_constr = pd.DataFrame(sat_rate_per_constr, columns=list(range(len(constraints))))

    percentage_of_samples_violating_constraints = pd.DataFrame(
        {'real_percentage_of_samples_violating_constraints': [percentage_of_samples_violating_constraints]},
        columns=['real_percentage_of_samples_violating_constraints'])

    return sat_rate_per_constr, percentage_of_samples_violating_constraints


def main():
    # ordering, constraints = parse_constraints_file('../data/constraints_disj1.txt')
    # ordering, constraints = parse_constraints_file('../data/url_constraints.txt')
    # ordering, constraints = parse_constraints_file('../data/lcld/lcld_constraints.txt')
    ordering, constraints = parse_constraints_file('../custom_constr.txt')

    # set ordering to random
    ordering = set_random_ordering(ordering)

    print('verbose constr')
    for constr in constraints:
        print(constr.verbose_readable())

    print('compute sets of constraints')
    sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)

    # preds = example_predictions()
    # preds = example_predictions_lcld()
    preds = example_predictions_custom()
    print('\n\nPreds', preds)

    preds.requires_grad = True
    t1 = time.time()
    corrected_preds = correct_preds(preds, ordering, sets_of_constr)
    print('Original predictions', preds[0])

    print('Corrected predictions', corrected_preds[0], corrected_preds.requires_grad)

    check_all_constraints_are_sat(constraints, preds, corrected_preds)

    print('Time to correct preds', time.time() - t1)

    compute_sat_stats(preds, constraints, mask_out_missing_values=True)
    print(corrected_preds, 'corrected')


def check_all_constraints_are_sat(constraints, preds, corrected_preds):
    # print('sat req?:')
    for constr in constraints:
        sat = constr.check_satisfaction(preds)
        if not sat:
            print('Not satisfied!', constr.readable())

    # print('*' * 80)
    all_sat_after_correction = True
    for constr in constraints:
        sat = constr.check_satisfaction(corrected_preds)
        if not sat:
            all_sat_after_correction = False
            print('Not satisfied!', constr.readable())
    if all_sat_after_correction:
        print('All constraints are satisfied after correction!')
    else:
        print('There are still constraint violations!!!')
    return all_sat_after_correction


def check_all_constraints_sat(corrected_preds, constraints, error_raise=True):
    all_sat_after_correction = True
    for constr in constraints:
        sat = constr.check_satisfaction(corrected_preds)
        if not sat:
            all_sat_after_correction = False
            if error_raise:
                raise Exception('Not satisfied!', constr.readable())
    return all_sat_after_correction


if __name__ == '__main__':
    main()
