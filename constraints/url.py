
import numpy as np

#from constraints.constraints_operator import apply_or
#from constraints.file_constraints import FileConstraints


def apply_or(operands) -> np.ndarray:
    return np.min(operands, axis=0)



def evaluate_numpy_url(x) -> np.ndarray:
    tau = 0.000001
    tolerance = 1e-2
    # if a>0, then b, 0
    def apply_if_a_supp_zero_than_b_supp_zero(a, b):
        return apply_or([x[:, a], (-x[:, b] + tau)])

    g1 = x[:, 1] - x[:, 0]
    g2 = np.sum(x[:, 3:18], axis=1) + 3 * x[:, 19] - x[:, 0]

    # g3: if x[:, 21] > 0 then x[:,3] > 0
    # g3 = apply_if_a_supp_zero_than_b_supp_zero(21, 3)

    # g4: if x[:, 23] > 0 then x[:,13] > 0
    # g4 = apply_if_a_supp_zero_than_b_supp_zero(23, 13)

    g5 = 3 * x[:, 20] + 4 * x[:, 21] + 4 * x[:, 22] + 2 * x[:, 23] - x[:, 0]

    # g6: if x[:, 19] > 0 then x[:,25] > 0
    # g6 = apply_if_a_supp_zero_than_b_supp_zero(19, 25)

    # g7: if x[:, 19] > 0 then x[:,26] > 0
    # g7 = apply_if_a_supp_zero_than_b_supp_zero(19, 26)

    # g8: if x[:, 2] > 0 then x[:,25] > 0
    # g8 = apply_if_a_supp_zero_than_b_supp_zero(2, 25)

    # g9: if x[:, 2] > 0 then x[:,26] > 0
    # g9 = apply_if_a_supp_zero_than_b_supp_zero(2, 26)

    # g10: if x[:, 28] > 0 then x[:,25] > 0
    # g10 = apply_if_a_supp_zero_than_b_supp_zero(28, 25)

    # g11: if x[:, 31] > 0 then x[:,26] > 0
    # g11 = apply_if_a_supp_zero_than_b_supp_zero(31, 26)

    ### x[:,38] <= x[:,37]
    g12 = x[:, 38] - x[:, 37]

    g13 = 3 * x[:, 20] - x[:, 0] + 1
    g14 = 4 * x[:, 21] - x[:, 0] + 1
    g15 = 4 * x[:, 22] - x[:, 0] + 1
    g16 = 2 * x[:, 23] - x[:, 0] + 1

    constraints = np.column_stack(
        [g1, g2, g5, g12, g13, g14, g15, g16]
    )
    constraints[constraints <= tolerance] = 0.0
    return constraints



