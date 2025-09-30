import numpy as np
import autograd.numpy as anp


def evaluate_numpy_lcld(x: np.ndarray) -> np.ndarray:
    # ----- PARAMETERS

    tol = 1e-2

    # installment = loan_amount * int_rate (1 + int_rate) ^ term / ((1+int_rate) ^ term - 1)
    calculated_installment = (
                                     x[:, 0] * (x[:, 2] / 1200) * (1 + x[:, 2] / 1200) ** x[:, 1]
                             ) / ((1 + x[:, 2] / 1200) ** x[:, 1] - 1)
    g1 = np.absolute(x[:, 3] - calculated_installment) - 0.099999

    # open_acc <= total_acc
    # g42 = x[:, 10] - x[:, 14]
    g2 = x[:, 11] - x[:, 15]

    # pub_rec_bankruptcies <= pub_rec

    # g43 = x[:, 16] - x[:, 11]
    g3 = x[:, 19] - x[:, 12]

    # term = 36 or term = 60
    # g4 = np.absolute((36 - x[:, 1]) * (60 - x[:, 1]))
    g4a = 36 - x[:, 1]
    g4b = x[:, 1] - 60

    # ratio_loan_amnt_annual_inc
    # g45 = np.absolute(x[:, 20] - x[:, 0] / x[:, 6])
    g5 = np.absolute(x[:, 22] - x[:, 0] / x[:, 7])

    # ratio_open_acc_total_acc
    # g46 = np.absolute(x[:, 21] - x[:, 10] / x[:, 14])
    g6 = np.absolute(x[:, 23] - x[:, 11] / x[:, 15])

    # diff_issue_d_earliest_cr_line
    # g7 was diff_issue_d_earliest_cr_line
    # g7 is not necessary in v2
    # issue_d and d_earliest cr_line are replaced
    # by month_since_earliest_cr_line
    # g47 = np.absolute(
    #     x[:, 22]
    #     - (
    #         self._date_feature_to_month(x[:, 7])
    #         - self._date_feature_to_month(x[:, 9])
    #     )
    # )

    # ratio_pub_rec_diff_issue_d_earliest_cr_line
    # g48 = np.absolute(x[:, 23] - x[:, 11] / x[:, 22])
    g8 = np.absolute(x[:, 25] - x[:, 12] / x[:, 24])

    # ratio_pub_rec_bankruptcies_pub_rec
    # g49 = np.absolute(x[:, 24] - x[:, 16] / x[:, 22])
    g9 = np.absolute(x[:, 26] - x[:, 19] / x[:, 24])

    # ratio_pub_rec_bankruptcies_pub_rec
    # ratio_mask = x[:, 11] == 0
    ratio_mask = x[:, 12] == 0
    ratio = np.empty(x.shape[0])
    ratio = np.ma.masked_array(ratio, mask=ratio_mask, fill_value=-1).filled()
    # ratio[~ratio_mask] = x[~ratio_mask, 16] / x[~ratio_mask, 11]
    ratio[~ratio_mask] = x[~ratio_mask, 19] / x[~ratio_mask, 12]
    ratio[ratio == np.inf] = -1
    ratio[np.isnan(ratio)] = -1
    # g10 = np.absolute(x[:, 25] - ratio)
    g10 = np.absolute(x[:, 27] - ratio)

    constraints = anp.column_stack(
        [g2, g3, g4a, g4b]
    )
    constraints[constraints <= tol] = 0.0

    return constraints