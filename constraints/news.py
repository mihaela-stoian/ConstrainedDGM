
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


def evaluate_numpy_news(x) -> np.ndarray:
    tolerance = 1e-2

    g1 = x[:,1] - x[:,2]
    g2 = x[:,5] - x[:,3]
    g3 = x[:,7] - x[:,6]
    g4 = x[:,27] - x[:,28]
    g5 = x[:,29] - x[:,28]

    constraints = np.column_stack(
        [g1, g2, g3, g4, g5]
        )
    constraints[constraints <= tolerance] = 0.0
    constraints[constraints > tolerance] = 1.0
    return constraints


