import numpy as np
from typing import List

from constraints_code.classes import Variable, Constraint, Atom, Inequality
from constraints_code.parser import parse_constraints_file


def collapse_atoms(atom_list):
    # merge any duplicated atoms in a atom list
    merged_atoms = {}
    merged_atoms: {int: Atom}
    for atom in atom_list:
        var = atom.variable.id
        if var not in merged_atoms.keys():
            merged_atoms[var] = atom
        else:
            variable, coefficient, positive_sign = merged_atoms[var].get_atom_attributes()
            existing_coeff = coefficient if positive_sign else -coefficient
            current_coeff = atom.coefficient if atom.positive_sign else -atom.coefficient
            new_coefficient = existing_coeff + current_coeff
            if new_coefficient != 0:
                new_atom = Atom(variable, float(np.abs(new_coefficient)), True if new_coefficient > 0 else False)
                merged_atoms[var] = new_atom
    return list(merged_atoms.values())


def split_constr_atoms(y: Variable, constr: Constraint):
    complementary_atoms = []
    for atom in constr.get_body_atoms():
        if atom.variable.id == y.id:
            red_coefficient = atom.coefficient  # note this is a positive real constant
        else:
            complementary_atoms.append(atom)
    return red_coefficient, complementary_atoms


def multiply_coefficients_of_atoms(atoms: List[Atom], coeff: float):
    new_atoms = []
    for atom in atoms:
        variable, coefficient, positive_sign = atom.get_atom_attributes()
        new_atom = Atom(variable, coefficient*coeff, positive_sign)
        new_atoms.append(new_atom)
    return new_atoms


def create_constr_by_reduction(y: Variable, constraints_with_y: List[Constraint]):
    red_constr = []
    # separate the constraints in two sets by the sign of y (pos or neg)
    pos_constr, neg_constr = get_pos_neg_x_constr(y, constraints_with_y)

    # then obtain new constr by reduction on y from pairs of constr (p,q)
    # where p is from pos_constr and q is from neg_constr
    for p in pos_constr:
        for q in neg_constr:
            p_coeff, p_complementary_body = split_constr_atoms(y, p)
            q_coeff, q_complementary_body = split_constr_atoms(y, q)
            # if p_complementary_body == []:
            #     break
            # if q_complementary_body == []:
            #     continue

            # multiply all coeff in p by q_coeff
            p_complementary_body = multiply_coefficients_of_atoms(p_complementary_body, q_coeff/p_coeff)

            # take the union of the p and q lists of atoms,
            # excluding the atom corresponding to y (whose coefficient is now 0)
            p_complementary_body.extend(q_complementary_body)

            # merge any atom duplicates
            p_complementary_body = collapse_atoms(p_complementary_body)

            _, ineq_sign_p, constant_p = p.single_inequality.get_ineq_attributes()
            _, ineq_sign_q, constant_q = q.single_inequality.get_ineq_attributes()
            # TODO: what happens if ineq_sign_p is >= and ineq_sign_q is > ? priority >?
            new_ineq_sign = ineq_sign_p
            new_constant = constant_p + constant_q
            if p_complementary_body != []:
                new_inequality = Inequality(p_complementary_body, new_ineq_sign, new_constant)
                red_constr.append(Constraint([new_inequality]))

    return red_constr


def get_pos_neg_x_constr(y, constraints_with_y):
    pos_constr, neg_constr = [], []
    for constr in constraints_with_y:
        for atom in constr.get_body_atoms():
            if atom.variable.id == y.id:
                if atom.positive_sign:
                    pos_constr.append(constr)
                else:
                    neg_constr.append(constr)
                break
    return pos_constr, neg_constr


def compute_set_of_constraints_for_variable(x: Variable, prev_x: Variable, constraints_at_previous_level: List[Constraint], verbose):
    # create two sets starting from constraints_at_previous_level:
    # one containing only the constraints which variable prev_x appears in
    # and its complement
    constraints_without_prev = []
    constraints_with_prev = []

    for constr in constraints_at_previous_level:
        if constr.contains_variable(prev_x):
            constraints_with_prev.append(constr)
        else:
            constraints_without_prev.append(constr)

    # then compute a new set of constraints derived by algebraic manipulation on constraints containing prev_x
    # note that this new set of constraints will not have occurrences of prev_x by construction
    reduced_constr = create_constr_by_reduction(prev_x, constraints_with_prev)

    # finally, get the union of constraints which do not contain y (constraints_without_prev U reduced_constr)
    constraints_without_prev.extend(reduced_constr)

    # if verbose:
    #     print('\nLEVEL', x.readable())
    #     print('-------------------Constraints at this level:')
    #     for constr in constraints_without_prev:
    #         print(constr.readable())

    return constraints_without_prev


def compute_sets_of_constraints(ordering: List[Variable], constraints: List[Constraint], verbose) -> {Variable: List[Constraint]}:
    # reverse the ordering:
    ordering = list(reversed(ordering))
    prev_x = ordering[0]

    # add all constraints for the highest ranking variable w.r.t. ordering
    ordered_constraints = {prev_x: constraints}
    print('All constraints')
    for constr in constraints:
        print(constr.readable())

    for x in ordering[1:]:
        constraints_at_previous_level = ordered_constraints[prev_x]
        set_of_constraints = compute_set_of_constraints_for_variable(x, prev_x, constraints_at_previous_level, verbose)
        ordered_constraints[x] = set_of_constraints
        prev_x = x

    print('-'*80)
    for x in ordering:
        print(f' *** Constraints for {x.readable()} ***')
        for i,constr in enumerate(ordered_constraints[x]):
            print(f'constr number {i}')
            print(constr.readable())
        if len(ordered_constraints[x]) == 0:
            print('empty set')
        print('***\n')
    print('-'*80)

    return ordered_constraints


def main():
    # ordering, constraints = parser.parse_constraints_file('../data/tiny_constraints.txt')
    ordering, constraints = parse_constraints_file('../data/heloc/heloc_constraints.txt')
    # for constr in constraints:
    #     print(constr.readable())
    #     # for elem in constr.inequality_list[-1].body:
    #     #     print('id', elem.get_variable_id())

    print('verbose constr')
    for constr in constraints:
        print(constr.verbose_readable())
    # print(constraints)

    print('compute sets of constraints')
    sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)
    # for var in ordering:
    #     print(var.readable())
    #     set_of_constr = sets_of_constr[var]
    #     for constr in set_of_constr:
    #         print(constr.verbose_readable())
    #     print()

if __name__ == '__main__':
    main()
