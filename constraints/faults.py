
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


def evaluate_numpy_faults(x) -> np.ndarray:
    tolerance = 1e-2

    g1 = x[:,0] - x[:,1]
    g2 = x[:,2] - x[:,3]
    g3 = x[:,8] - x[:,9]
    g4 = x[:,7] - x[:,9]

    constraints = np.column_stack(
        [g1, g2, g3, g4]
        )
    constraints[constraints <= tolerance] = 0.0
    constraints[constraints > tolerance] = 1.0
    return constraints

