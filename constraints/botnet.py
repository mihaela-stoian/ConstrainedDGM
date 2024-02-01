import numpy as np
import autograd.numpy as anp
import pickle
import joblib 
import os.path
from typing import List
import pandas as pd


def evaluate_numpy_botnet(x):
        tol = 1e-2
        # x, _ = self.get_x_y()
        # with open("data/botnet/feat_dict.pkl", "rb") as f:
        #     feat_idx = pickle.load(f)
        with open('data/botnet/feat_dict.pkl', 'rb') as handle:
            feat_idx = pickle.load(handle)
        features = list(feat_idx.keys())

        def get_feature_family(features_l: List[str], family: str) -> List[str]:
            return list(filter(lambda el: el.startswith(family), features_l))

        def sum_list_feature(features_l: List[str]):
            out = x[:, feat_idx[features_l[0]]]
            for el in features_l[1:]:
                out = out + x[:, feat_idx[el]]
            return out

        def sum_feature_family(features_l: List[str], family: str):
            return sum_list_feature(get_feature_family(features_l, family))

        g1 = np.absolute(
             ( sum_feature_family(features, "icmp_sum_s_")
                + sum_feature_family(features, "udp_sum_s_")
                + sum_feature_family(features, "tcp_sum_s_")) - 
                
            (sum_feature_family(features, "bytes_in_sum_s_")
            + sum_feature_family(features, "bytes_out_sum_s_"))
        )
        g2 = np.absolute((sum_feature_family(features, "icmp_sum_d_")
            + sum_feature_family(features, "udp_sum_d_")
            + sum_feature_family(features, "tcp_sum_d_")) - 
            ( sum_feature_family(features, "bytes_in_sum_d_")
            + sum_feature_family(features, "bytes_out_sum_d_"))
        )
        g_packet_size = []
        for e in ["s", "d"]:
            # -1 cause ignore last OTHER features
            bytes_outs = get_feature_family(features, f"bytes_out_sum_{e}_")[:-1]
            pkts_outs = get_feature_family(features, f"pkts_out_sum_{e}_")[:-1]
            if len(bytes_outs) != len(pkts_outs):
                raise Exception("len(bytes_out) != len(pkts_out)")

            # Tuple of list to list of tuples
            for byte_out, pkts_out in list(zip(bytes_outs, pkts_outs)):
                g = np.divide(x[:, feat_idx[byte_out]], x[:, feat_idx[pkts_out]], out=np.zeros_like(x[:, feat_idx[byte_out]]), where=x[:, feat_idx[pkts_out]]!=0) - 1500
                g_packet_size.append(g)

        g_min_max_sum = []
        for e_1 in ["bytes_out", "pkts_out", "duration"]:
            for port in [
                "1",
                "3",
                "8",
                "10",
                "21",
                "22",
                "25",
                "53",
                "80",
                "110",
                "123",
                "135",
                "138",
                "161",
                "443",
                "445",
                "993",
                "OTHER",
            ]:
                for e_2 in ["d", "s"]:
                    g_min_max_sum.extend(
                        [
                            x[:, feat_idx[f"{e_1}_max_{e_2}_{port}"]] - x[:,feat_idx[f"{e_1}_sum_{e_2}_{port}"]],
                            x[:, feat_idx[f"{e_1}_min_{e_2}_{port}"]] - x[:,feat_idx[f"{e_1}_sum_{e_2}_{port}"]],
                            x[:, feat_idx[f"{e_1}_min_{e_2}_{port}"]] - x[:,feat_idx[f"{e_1}_max_{e_2}_{port}"]],
                        ]
                    )

        constraints = [g1, g2] + g_packet_size + g_min_max_sum
        constraints = anp.column_stack(constraints)

        return constraints