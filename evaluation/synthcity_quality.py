# stdlib
import platform
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# synthcity absolute
from synthcity.metrics.core import MetricEvaluator
from synthcity.plugins.core.dataloader import DataLoader

from synthcity.utils.reproducibility import clear_cache
from synthcity.utils.serialization import load_from_file, save_to_file


class StatisticalEvaluator(MetricEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_statistical.StatisticalEvaluator
        :parts: 1

    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def type() -> str:
        return "stats"

    @abstractmethod
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        ...

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        cache_file = (
            self._workspace
            / f"sc_metric_cache_{self.type()}_{self.name()}_{X_gt.hash()}_{X_syn.hash()}_{self._reduction}_{platform.python_version()}.bkp"
        )
        if self.use_cache(cache_file):
            return load_from_file(cache_file)

        clear_cache()
        results = self._evaluate(X_gt, X_syn)
        save_to_file(cache_file, results)
        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate_default(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> float:
        return self.evaluate(X_gt, X_syn)[self._default_metric]


class AlphaPrecision(StatisticalEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_statistical.AlphaPrecision
        :parts: 1

    Evaluates the alpha-precision, beta-recall, and authenticity scores.

    The class evaluates the synthetic data using a tuple of three metrics:
    alpha-precision, beta-recall, and authenticity.
    Note that these metrics can be evaluated for each synthetic data point (which are useful for auditing and
    post-processing). Here we average the scores to reflect the overall quality of the data.
    The formal definitions can be found in the reference below:

    Alaa, Ahmed, Boris Van Breugel, Evgeny S. Saveliev, and Mihaela van der Schaar. "How faithful is your synthetic
    data? sample-level metrics for evaluating and auditing generative models."
    In International Conference on Machine Learning, pp. 290-306. PMLR, 2022.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="authenticity_OC", **kwargs)

    @staticmethod
    def name() -> str:
        return "alpha_precision"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def metrics(
        self,
        X: np.ndarray,
        X_syn: np.ndarray,
        emb_center: Optional[np.ndarray] = None,
    ) -> Tuple:
        # if len(X) != len(X_syn):
        #     raise RuntimeError("The real and synthetic data must have the same length")

        if emb_center is None:
            emb_center = np.mean(X, axis=0)

        n_steps = 30
        alphas = np.linspace(0, 1, n_steps)

        Radii = np.quantile(np.sqrt(np.sum((X - emb_center) ** 2, axis=1)), alphas)

        synth_center = np.mean(X_syn, axis=0)

        alpha_precision_curve = []
        beta_coverage_curve = []

        synth_to_center = np.sqrt(np.sum((X_syn - emb_center) ** 2, axis=1))

        nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(X)
        real_to_real, _ = nbrs_real.kneighbors(X)

        nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(X_syn)
        real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(X)

        # Let us find closest real point to any real point, excluding itself (therefore 1 instead of 0)
        real_to_real = real_to_real[:, 1].squeeze()
        real_to_synth = real_to_synth.squeeze()
        real_to_synth_args = real_to_synth_args.squeeze()

        real_synth_closest = X_syn[real_to_synth_args]

        real_synth_closest_d = np.sqrt(
            np.sum((real_synth_closest - synth_center) ** 2, axis=1)
        )
        closest_synth_Radii = np.quantile(real_synth_closest_d, alphas)

        for k in range(len(Radii)):
            precision_audit_mask = synth_to_center <= Radii[k]
            alpha_precision = np.mean(precision_audit_mask)

            beta_coverage = np.mean(
                (
                    (real_to_synth <= real_to_real)
                    * (real_synth_closest_d <= closest_synth_Radii[k])
                )
            )

            alpha_precision_curve.append(alpha_precision)
            beta_coverage_curve.append(beta_coverage)

        # See which one is bigger

        authen = real_to_real[real_to_synth_args] < real_to_synth
        authenticity = np.mean(authen)

        Delta_precision_alpha = 1 - np.sum(
            np.abs(np.array(alphas) - np.array(alpha_precision_curve))
        ) / np.sum(alphas)

        if Delta_precision_alpha < 0:
            raise RuntimeError("negative value detected for Delta_precision_alpha")

        Delta_coverage_beta = 1 - np.sum(
            np.abs(np.array(alphas) - np.array(beta_coverage_curve))
        ) / np.sum(alphas)

        if Delta_coverage_beta < 0:
            raise RuntimeError("negative value detected for Delta_coverage_beta")

        return (
            alphas,
            alpha_precision_curve,
            beta_coverage_curve,
            Delta_precision_alpha,
            Delta_coverage_beta,
            authenticity,
        )

    def _normalize_covariates(
        self,
        X: DataLoader,
        X_syn: DataLoader,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """_normalize_covariates
        This is an internal method to replicate the old, naive method for evaluating
        AlphaPrecision.

        Args:
            X (DataLoader): The ground truth dataset.
            X_syn (DataLoader): The synthetic dataset.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: normalised version of the datasets
        """
        X_gt_norm = X.dataframe().copy()
        X_syn_norm = X_syn.dataframe().copy()
        if self._task_type != "survival_analysis":
            if hasattr(X, "target_column"):
                X_gt_norm = X_gt_norm.drop(columns=[X.target_column])
            if hasattr(X_syn, "target_column"):
                X_syn_norm = X_syn_norm.drop(columns=[X_syn.target_column])
        scaler = MinMaxScaler().fit(X_gt_norm)
        if hasattr(X, "target_column"):
            X_gt_norm_df = pd.DataFrame(
                scaler.transform(X_gt_norm),
                columns=[
                    col
                    for col in X.train().dataframe().columns
                    if col != X.target_column
                ],
            )
        else:
            X_gt_norm_df = pd.DataFrame(
                scaler.transform(X_gt_norm), columns=X.train().dataframe().columns
            )

        if hasattr(X_syn, "target_column"):
            X_syn_norm_df = pd.DataFrame(
                scaler.transform(X_syn_norm),
                columns=[
                    col
                    for col in X_syn.dataframe().columns
                    if col != X_syn.target_column
                ],
            )
        else:
            X_syn_norm_df = pd.DataFrame(
                scaler.transform(X_syn_norm), columns=X_syn.dataframe().columns
            )

        return (X_gt_norm_df, X_syn_norm_df)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:

        results = {}

        X_ = X.numpy().reshape(len(X), -1)
        X_syn_ = X_syn.numpy().reshape(len(X_syn), -1)

        # OneClass representation
        emb = "_OC"
        oneclass_model = self._get_oneclass_model(X_)
        X_ = self._oneclass_predict(oneclass_model, X_)
        X_syn_ = self._oneclass_predict(oneclass_model, X_syn_)
        emb_center = oneclass_model.c.detach().cpu().numpy()

        (
            alphas,
            alpha_precision_curve,
            beta_coverage_curve,
            Delta_precision_alpha,
            Delta_coverage_beta,
            authenticity,
        ) = self.metrics(X_, X_syn_, emb_center=emb_center)

        results[f"delta_precision_alpha{emb}"] = Delta_precision_alpha
        results[f"delta_coverage_beta{emb}"] = Delta_coverage_beta
        results[f"authenticity{emb}"] = authenticity

        X_df, X_syn_df = self._normalize_covariates(X, X_syn)
        (
            alphas_naive,
            alpha_precision_curve_naive,
            beta_coverage_curve_naive,
            Delta_precision_alpha_naive,
            Delta_coverage_beta_naive,
            authenticity_naive,
        ) = self.metrics(X_df.to_numpy(), X_syn_df.to_numpy(), emb_center=None)

        results["delta_precision_alpha_naive"] = Delta_precision_alpha_naive
        results["delta_coverage_beta_naive"] = Delta_coverage_beta_naive
        results["authenticity_naive"] = authenticity_naive

        return results


