from typing import List
import torch

TOLERANCE=1e-2

class Variable():
    def __init__(self, variable: str):
        super().__init__()
        self.variable = variable
        self.id = self.get_variable_id()

    def readable(self):
        return self.variable

    def get_variable_id(self):
        id = int(self.variable.split('_')[-1])
        return id


class Atom():
    def __init__(self, variable: Variable, coefficient: float, positive_sign: bool):
        super().__init__()
        self.variable = variable
        self.coefficient = coefficient
        self.positive_sign = positive_sign

    def get_variable_id(self):
        return self.variable.get_variable_id()

    def eval(self, x_value):
        return x_value * self.get_signed_coefficient()

    def get_signed_coefficient(self):
        return self.coefficient if self.positive_sign else -1 * self.coefficient

    def readable(self):
        readable = ' + ' if self.positive_sign else ' - '
        readable += (f'{self.coefficient:.2f}' if self.coefficient != int(
            self.coefficient) else f'{self.coefficient:.0f}') if self.coefficient != 1 else ''
        readable += self.variable.readable()
        return readable

    def get_atom_attributes(self):
        return self.variable, self.coefficient, self.positive_sign


class Inequality():
    def __init__(self, body: List[Atom], ineq_sign: str, constant: float):
        super().__init__()
        self.ineq_sign = ineq_sign
        # assert constant == 0, 'Inequalities need to be provided in the format body>0 or body>=0'
        self.constant = constant
        self.body = body

    def readable(self):
        readable_ineq = ''
        for elem in self.body:
            readable_ineq += elem.readable()
        readable_ineq += ' ' + self.ineq_sign + ' ' + str(self.constant)
        return readable_ineq

    def get_body_variables(self):
        var_list = []
        for atom in self.body:
            var_list.append(atom.variable)
        return var_list

    def get_body_atoms(self):
        atom_list = []
        for atom in self.body:
            atom_list.append(atom)
        return atom_list

    def get_x_complement_body_atoms(self, x: Variable) -> (List[Atom], Atom, bool):
        # TODO: get_x_complement_body_atoms(self, x) implementation for DisjunctIneq class
        # given a constraint constr in which variable x appears,
        # return the body of the constraint (i.e. the left-hand side of the inequality)
        # from which x occurrences have been removed
        complementary_atom_list = []
        x_atom_occurrences = []
        for atom in self.body:
            if atom.variable.id != x.id:
                complementary_atom_list.append(atom)
            else:
                x_atom_occurrences.append(atom)
        assert len(x_atom_occurrences) <= 1, "variable {x.id} appears more than one time, function collapse_atoms() from compute_sets_of_constraints should be applied"
        if len(x_atom_occurrences) == 1:
            x_atom_occurrences = x_atom_occurrences[0]
        is_strict_inequality = True if self.ineq_sign == '>' else False
        return complementary_atom_list, x_atom_occurrences, self.constant, is_strict_inequality

    def get_ineq_attributes(self):
        return self.body, self.ineq_sign, self.constant

    def contains_variable(self, x: Variable):
        body_variables = [elem.id for elem in self.get_body_variables()]
        return x.id in body_variables

    def check_satisfaction(self, preds: torch.Tensor) -> torch.Tensor:
        eval_body_value = eval_atoms_list(self.body, preds)
        if self.ineq_sign == '>':
            results = eval('(eval_body_value > self.constant - TOLERANCE) | (eval_body_value > self.constant + TOLERANCE)')
        elif self.ineq_sign == '>=':
            results = eval('(eval_body_value >= self.constant - TOLERANCE) | (eval_body_value >= self.constant + TOLERANCE)')
        return results #.all()

    def detailed_sat_check(self, preds: torch.Tensor) -> torch.Tensor:
        eval_body_value = eval_atoms_list(self.body, preds)
        if self.ineq_sign == '>':
            results = eval('(eval_body_value > self.constant - TOLERANCE) | (eval_body_value > self.constant + TOLERANCE)')
        elif self.ineq_sign == '>=':
            results = eval('(eval_body_value >= self.constant - TOLERANCE) | (eval_body_value >= self.constant + TOLERANCE)')
        return results, eval_body_value, self.constant, self.ineq_sign


    # def check_satisfaction(self, preds: torch.Tensor, tolerance=1e-4) -> torch.Tensor:
    #     eval_body_value = eval_atoms_list(self.body, preds)
    #     eval_body_value -= self.constant
    #     if self.ineq_sign == '>':
    #         results = eval('eval_body_value + tolerance > 0') | eval('eval_body_value - tolerance > 0')
    #     elif self.ineq_sign == '>=':
    #         results =  eval('eval_body_value + tolerance >= 0') | eval('eval_body_value - tolerance >= 0')
    #     return results #.all()


class DisjunctInequality(Inequality):
    def __init__(self, atoms_list: List[List], ineq_sign: str, constant: float, inequality_list: List[Inequality]):
        super().__init__(atoms_list, ineq_sign, constant)
        self.disjunct_op = 'max'
        self.subinequalities = inequality_list

    def readable(self):
        readable_ineq = ''
        for i,ineq in enumerate(self.subinequalities):
            readable_ineq += f'disjunct {i}: {ineq.readable()}\n'
        # readable_ineq = 'max('
        # for body_elem in self.body:
        #     for atom in body_elem:
        #         readable_ineq += atom.readable()
        #     readable_ineq += ', '
        # readable_ineq = readable_ineq[:-2] + ')'
        # readable_ineq += ' ' + self.ineq_sign + ' ' + str(self.constant)
        return readable_ineq

    def get_body_variables(self):
        var_list = []
        for body_elem in self.body:
            for atom in body_elem:
                var_list.append(atom.variable)
        return var_list

    def get_body_atoms(self):
        atom_list = []
        for body_elem in self.body:
            for atom in body_elem:
                atom_list.append(atom)
        return atom_list

    def split_ineqs_with_and_without_x(self, x: Variable) -> (List[Inequality], List[Inequality]):
        # separate ineqs that contain x from those that do not contain x
        ineqs_with_x = []
        ineqs_without_x = []
        for ineq in self.subinequalities:
            if ineq.contains_variable(x):
                ineqs_with_x.append(ineq)
            else:
                ineqs_without_x.append(ineq)

        return ineqs_with_x, ineqs_without_x

    def get_x_complement_body_atoms(self, x: Variable, preds: torch.Tensor) -> (List[Atom], Atom):
        # just select one constraint in which x appears, but only if all the inequalities that do not contain x are not satisfied
        complementary_atom_list = []
        # x_atom_occurrences = []

        # separate ineqs that contain x from those that do not contain x
        ineqs_with_x, ineqs_without_x = self.split_ineqs_with_and_without_x(x)
        ineqs_with_x: List[Inequality]

        # if x.id == 25:
        #     print(preds[:, 25], preds[:, 2], 'check preds0', [e.readable() for e in ineqs_with_x], [e.readable() for e in ineqs_without_x])

        # first, check whether any of the ineqs in ineqs_without_x holds
        # if it does, it's not necessary to correct the value of x
        mask_sat_batch_elem = torch.ones(preds.shape[0]) == 0.
        # print(mask_sat_batch_elem)
        for ineq in ineqs_without_x:
            # if x.id == 25:
            #     print(preds[:,25], preds[:,2], 'check preds')
            #     print(ineq.readable())
            # mask out the batch elements for which an ineq without x is satisfied, so that we do not correct the value of these batch elements
            mask_sat_batch_elem = torch.logical_or(mask_sat_batch_elem, ineq.check_satisfaction(preds))
            #if all batch elems sat at least one of the ineqs that do not contain x
            # if x.id == 25:
            #     print(preds[:,25], preds[:,2], 'check preds2')
            #     print(ineq.readable())
            #     print(mask_sat_batch_elem, mask_sat_batch_elem.all())
            if mask_sat_batch_elem.all(): # TODO: fix this when using batches
                return (None, None, None, None), None  # TODO: should this depend on the ordering of the vars, and if an ineq here is satisfied, check whether it contains any var of higher order, and if so, correct x

        # if there's no ineq in ineqs_without_x that is satisfied
        # select a ineq from ineqs_with_x which will be used to correct the value of x
        # TODO: use a function on the ordering of x relative to the vars in the ineqs from ineqs_without_x
        selected_ineq_with_x = ineqs_with_x[0]

        # then simply get the complement body of this selected ineq, where x apprears in
        return selected_ineq_with_x.get_x_complement_body_atoms(x), mask_sat_batch_elem

    def check_satisfaction(self, preds: torch.Tensor) -> bool:
        disj_results = []
        for ineq in self.subinequalities:
            disj_results.append(ineq.check_satisfaction(preds).unsqueeze(1))
        # disj res: B x num_ineqs
        disj_results = torch.cat(disj_results, dim=1)

        # disj_results: B
        disj_results = disj_results.any(dim=1)
        return disj_results


class Constraint():
    def __init__(self, inequality_list: List[Inequality]):
        super().__init__()
        self.inequality_list = inequality_list
        self.single_inequality = self.get_single_inequality()

    def rewrite_disjunct_ineq(self) -> Inequality:
        atoms_list = [ineq.body for ineq in self.inequality_list]
        ineq_sign = self.inequality_list[0].ineq_sign
        # TODO: what if signs differ in the disjunction of ineq (e.g. x1>=0 or x2>0);
        #  then need min(max(x1,0), max(x2,epsilon))
        constant = self.inequality_list[0].constant
        new_ineq = DisjunctInequality(atoms_list, ineq_sign, constant, self.inequality_list)
        return new_ineq

    def get_single_inequality(self):
        if len(self.inequality_list) >= 2:
            single_constr = self.rewrite_disjunct_ineq()
        else:
            single_constr = self.inequality_list[0]
        return single_constr

    def readable(self):
        readable_constr = self.single_inequality.readable()
        return readable_constr

    def verbose_readable(self):
        readable_constr = self.inequality_list[0].readable()
        for ineq in self.inequality_list[1:]:
            readable_constr += ' or ' + ineq.readable()
        return readable_constr

    def contains_variable(self, x: Variable):
        return self.single_inequality.contains_variable(x)

    def get_body_atoms(self):
        return self.single_inequality.get_body_atoms()

    def get_body_atoms_per_disjunct(self):
        atom_bodies = []
        for disjunct in self.inequality_list:
            atom_bodies.append(disjunct.get_body_atoms())
        return atom_bodies

    def get_x_complement_body_atoms(self, x: Variable):
        atom_bodies = []
        for disjunct in self.inequality_list:
            atom_bodies.append(disjunct.get_x_complement_body_atoms(x)[0])
        return atom_bodies


    def check_satisfaction(self, preds):
        return self.single_inequality.check_satisfaction(preds).all()  # for the whole batch

    def check_satisfaction_per_sample(self, preds):
        return self.single_inequality.check_satisfaction(preds)

    def detailed_sample_sat_check(self, preds):
        return self.single_inequality.detailed_sat_check(preds)


def eval_atoms_list(atoms_list: List[Atom], preds: torch.Tensor, reduction='sum'):
    # TODO: eval_atoms_list() implementation for DisjunctIneq class
    evaluated_atoms = []
    for atom in atoms_list:
        atom_value = preds[:, atom.variable.id]
        evaluated_atoms.append(atom.eval(atom_value))

    if evaluated_atoms == []:
        return torch.zeros(preds.shape[0])

    evaluated_atoms = torch.stack(evaluated_atoms, dim=1)
    if reduction == 'sum':
        result = evaluated_atoms.sum(dim=1)
    else:
        raise Exception(f'{reduction} reduction not implemented!')
    return result


def get_missing_mask(ineq_atoms: List[Atom], preds: torch.Tensor):
    raw_variable_values = []
    if type(ineq_atoms[0]) != list:
        ineq_atoms = [ineq_atoms]
    for atoms_list in ineq_atoms:
        for atom in atoms_list:
            raw_variable_values.append(preds[:, atom.variable.id])

    raw_variable_values = torch.stack(raw_variable_values, dim=1)
    missing_values_mask = raw_variable_values.prod(dim=1) < -TOLERANCE
    return missing_values_mask
