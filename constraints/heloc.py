
import numpy as np
#from constraints.constraints_operator import apply_or
#from constraints.file_constraints import FileConstraints
INFINITY = np.int32(1e16)


def apply_or(operands) -> np.ndarray:
    return np.min(operands, axis=0)

def missing_cond(prod, values):
    mask = prod < 0
    # print(values[prod<0])
    values[mask] = -INFINITY
    # print(values[prod<0])

    return values


def evaluate_numpy_heloc(x) -> np.ndarray:
    tau = 0.000001
    tolerance = 1e-2
    # if a>0, then b, 0
    def apply_if_a_supp_zero_than_b_supp_zero(a, b):
        return apply_or([x[:, a], (-x[:, b] + tau)])

    g1 = missing_cond(x[:,2]*x[:,1], x[:,2] - x[:,1])
    g2 = missing_cond(x[:,6]*x[:,5], x[:,6] - x[:,5])
    # g3 = missing_cond(x[:,4]*x[:,7], apply_if_a_supp_zero_than_b_supp_zero(4, 7))
    # g4 = missing_cond(x[:,8]*x[:,10], apply_if_a_supp_zero_than_b_supp_zero(8, 10))
    # g5 = missing_cond(x[:,15]*x[:,14], apply_if_a_supp_zero_than_b_supp_zero(15, 14))

    g6 = missing_cond(x[:,4]*x[:,11], x[:,4] - x[:,11])
    g7 = missing_cond(x[:,12]*x[:,11], x[:,12] - x[:,11])
    #g8 = x[:,4] - x[:,11]*x[:,7]


    g9 = missing_cond(x[:,19]*x[:,11], x[:,19] - x[:,11])
    g10 = missing_cond(x[:,20]*x[:,11], x[:,20] - x[:,11])
    #g11 = x[:,20] - x[:,11]*x[:,13]
    g12 = missing_cond(x[:,21]*x[:,11], x[:,21] - x[:,11])
    #g13 =   x[:,19] + x[:,20] - x[:,22]*x[:,11]

    constraints = np.column_stack(
        [g1, g2, g6, g7, g9, g10, g12]
        #[g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13]

        )
    constraints[(constraints <= tolerance) & (constraints != -INFINITY)] = 0.0
    constraints[(constraints > tolerance)] = 1.0
    return constraints


