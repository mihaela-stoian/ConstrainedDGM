from typing import List

from constraints_code.classes import Variable, Atom, Inequality, Constraint

ALLOWED_BOOL_OPS = ['or', 'neg']
ALLOWED_OPS = ['+', '-', '*', '/']
ALLOWED_INEQ_SIGNS = ['>=', '>']  # TODO: support for '='?


def neg_postprocess_ineq(ineq: Inequality) -> Inequality:
    # equivalent to negating the sign of the inequality and the multiplying by -1
    # e.g. neg x1+x2-x3 >=0 becomes -x1-x2+x3 > 0
    atoms_list, ineq_sign, constant = ineq.get_ineq_attributes()

    # negate all atoms in the inequality
    neg_atoms_list = []
    for atom in atoms_list:
        variable, coefficient, positive_sign = atom.get_atom_attributes()
        neg_atom = Atom(variable, coefficient, not positive_sign)
        neg_atoms_list.append(neg_atom)

    # change the sign of the inequality
    neg_ineq_sign = '>' if ineq_sign == '>=' else '>='
    neg_ineq = Inequality(neg_atoms_list, neg_ineq_sign, constant)
    return neg_ineq


def parse_atom(x):
    """Example: y1, -2y2, -y23, 6y3."""
    x = x.strip().replace(" ", "")
    if x == '':
        return None
    positive_sign = False if '-' in x else True

    if '-' in x or '+' in x:
        x = x[1:]

    coefficient, var_id = x.split('y')
    coefficient = float(coefficient) if coefficient != '' else 1
    var = Variable('y' + var_id)

    atom = Atom(var, coefficient, positive_sign)
    return atom


def parse_inequality(inequality):
    """Example: y1-2y2>0, y1+y2>=0."""
    ineq_sign = None
    inequality = inequality.strip()
    for sign in ALLOWED_INEQ_SIGNS:
        if sign in inequality:
            ineq_sign = sign
            break
    body, constant = inequality.split(ineq_sign)
    constant = float(constant)

    atoms_list = []
    constructed_atom = ''
    for elem in body:
        if elem in ALLOWED_OPS:
            if constructed_atom == '':
                constructed_atom += elem
            else:
                atom = parse_atom(constructed_atom)
                if atom is not None:
                    atoms_list.append(atom)
                constructed_atom = elem
        else:
            constructed_atom += elem
    atom = parse_atom(constructed_atom)
    if atom is not None:
        atoms_list.append(atom)

    ineq = Inequality(atoms_list, ineq_sign, constant)
    return ineq


def parse_constraint(constr) -> Constraint:
    ineqs = []
    neg_postprocess = False

    disjunct_inequalities = constr.split('or')
    for ineq in disjunct_inequalities:
        if 'neg' in ineq:
            ineq = ineq.split('neg')[-1]
            neg_postprocess = True
        ineq = parse_inequality(ineq)
        if neg_postprocess:
            ineq = neg_postprocess_ineq(ineq)
            neg_postprocess = False
        ineqs.append(ineq)

    constr = Constraint(ineqs)
    return constr


def parse_constraints_file(file: str) -> (List[Variable], List[Constraint]):
    f = open(file, 'r')
    constraints = []
    ordering = []
    for line in f:
        line = line.strip()
        if 'ordering' in line:
            str_ordering = line.split('ordering ')[-1].split()
            for elem in str_ordering:
                ordering.append(Variable(elem.replace(" ", ""))) # remove all whitespace from variable string
            continue
        constraints.append(parse_constraint(line))
    if len(ordering) == 0:
        raise Exception('An ordering of the variables must be provided!')
    return ordering, constraints


def main():
    ordering, constraints = parse_constraints_file('../data/constraints.txt')
    for constr in constraints:
        print(constr.readable())
        # for elem in constr.inequality_list[0].body:
        #     print('id', elem.get_variable_id())

    print('verbose constr')
    for constr in constraints:
        print(constr.verbose_readable())
    print('ordering', ordering)


if __name__ == '__main__':
    main()
