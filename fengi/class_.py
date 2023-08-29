"""Module Name: data_processing_module."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from tqdm import tqdm
from tqdm.notebook import tqdm

tqdm.pandas()

import os

import scipy.stats as stats
from pandarallel import pandarallel
from sklearn.decomposition import PCA

from .utils_feat import gauss
from .utils_feat.handling_outliers import detect_outliers_quantile, detect_outliers_zscore
from .utils_feat.quantization import hard_quantize


class Data:
    """Data processing and analysis class."""

    def __init__(self, f_matrix, b_matrix):
        """Initialize the Data class.

        :param f_matrix: pd.DataFrame
            Features matrix.
        :param b_matrix: pd.DataFrame
            Factor matrix.
        """
        self.f_matrix = f_matrix
        self.b_matrix = b_matrix

    def exposure(self):
        """Calculate the cross-sectional measure of orthogonality of features to given subspace.

        Returns:
        np.ndarray: Factor exposure matrix.
        """

        def loc_exposure(f_mat_temp):
            features = f_mat_temp.columns[1:]
            b_mat_temp = self.b_matrix.loc[
                self.b_matrix["date"].isin(f_mat_temp["date"]),
                self.b_matrix.columns[1:],
            ]
            fact_exp_lis = []
            for feature in features:
                M = np.array(f_mat_temp[feature])
                factor_exposure = np.dot(b_mat_temp.to_numpy().T, M)  # vector
                fact_exp_lis.append(factor_exposure)
            fact_exp_matrix = np.array(fact_exp_lis)
            return fact_exp_matrix

        f_exp_matrix = self.f_matrix.groupby("date", group_keys=False).apply(lambda x: loc_exposure(x))
        return f_exp_matrix

    def plot_corr(self, data, fig_name):
        """Plot correlation heatmaps.

        :param data: pd.DataFrame
            The data to be used for plotting the correlation heatmap.
        :param fig_name: str
            Name for saving the figures.

        :return: int
            Always returns 0.
        """
        Path("./paper/figures/").mkdir(parents=True, exist_ok=True)

        spear_corr = scipy.stats.spearmanr(data.drop(["date"], axis=1))[0]
        plt.figure(figsize=(15, 10))
        sns.heatmap(spear_corr)
        plt.savefig(f"./paper/figures/{fig_name}_spearman.png")

        plt.figure(figsize=(15, 10))
        sns.heatmap(data.drop(["date"], axis=1).corr())
        plt.savefig(f"./paper/figures/{fig_name}_pearson.png")
        return 0

    def plot_dist(self, data, fig_name, ndist=600, gbell=True):
        """Plot distribution histograms.

        :param data: pd.DataFrame
            Data matrix for plotting the distribution histograms.
        :param fig_name: str
            Name for saving the figures.
        :param ndist: int, optional
            Number of bins for histograms. Default is 600.
        :param gbell: bool, optional
            Plot Gaussian bell curve. Default is True.

        :return: int
            Always returns 0.
        """
        Path("./paper/figures/").mkdir(parents=True, exist_ok=True)

        for col in data.columns[1:]:
            mu = data[col].mean()
            sigma = data[col].std()
            x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

            plt.figure(figsize=(15, 10))
            data[col].hist(bins=ndist, density=True)
            if gbell:
                plt.plot(x, scipy.stats.norm.pdf(x, mu, sigma))
            plt.title(f"{col}")
            plt.grid("on")
            plt.savefig(f"./paper/figures/{fig_name}_{col}.png")
        return 0

    def standardize(self):
        """Standardize the feature matrix.

        Returns:
        int: Always returns 0.
        """

        def preprocess(local_f_matrix):
            """Get sigma --> median of the std dev list."""
            epochs = local_f_matrix[local_f_matrix.columns[0]].unique()
            for col in tqdm(local_f_matrix.columns[1:]):
                sigma_list = []
                for epoch in epochs:
                    M = local_f_matrix.loc[local_f_matrix["date"] == epoch, col]
                    sigma_list.append(M.std())
                sigma = np.median(sigma_list)
                for epoch in epochs:
                    local_f_matrix.loc[local_f_matrix["date"] == epoch, col] /= sigma

            return local_f_matrix

        self.f_matrix = preprocess(self.f_matrix)
        return 0

    def orthogonalize(self, beta_coeff=0.0, nb_workers=None):
        """Cross-sectionally project f to be orthogonal to the subspace spanned by B, with respect to the dot product, in a least square sense.

        :param beta_coeff: float
            Coefficient for adjusting projection.

        :return: int
            Always returns 0.
        """

        def loc_orthogonalize(f_mat_temp):
            print(f'Epoch: {f_mat_temp["date"].iloc[0]}')
            features = f_mat_temp.columns[1:]
            b_mat_temp = self.b_matrix.loc[
                self.b_matrix["date"].isin(f_mat_temp["date"]),
                self.b_matrix.columns[1:],
            ]

            b_pinv_temp = np.linalg.pinv(b_mat_temp)
            for feature in features:
                m = np.array(f_mat_temp[feature])
                m_parallel = np.dot(b_mat_temp.to_numpy(), np.dot(b_pinv_temp, m))
                if beta_coeff > 0:
                    mean_bf = np.mean(np.dot(b_mat_temp.to_numpy().T, m))
                    m -= (1 - beta_coeff / mean_bf) * m_parallel
                else:
                    m -= m_parallel
                f_mat_temp[feature] = m
            return f_mat_temp

        if nb_workers is None:
            available_cores = os.cpu_count()
            if available_cores is None:
                cores_to_use = 1
            else:
                cores_to_use = int(available_cores * 0.95)
            pandarallel.initialize(progress_bar=True, nb_workers=cores_to_use)
        else:
            pandarallel.initialize(progress_bar=True, nb_workers=nb_workers)

        self.f_matrix = self.f_matrix.groupby("date", group_keys=False).parallel_apply(
            lambda x: loc_orthogonalize(x)
        )
        return 0

    def gaussianize(self):
        """Apply Gaussianization to the features in the feature matrix.

        For each column in the feature matrix,
        this method fits a Gaussianization transformation and applies it to the column.

        :return: int
            Always returns 0.
        """
        f_local = self.f_matrix.copy()
        for col in tqdm(f_local.columns[1:]):

            def apply_kernel(x, col):
                gauss_kernel = gauss.Gaussianize()
                gauss_kernel.fit(x[col], y=None)
                y = gauss_kernel.transform(x[col])
                x.loc[:, col] = np.squeeze(y)
                return x

            self.f_matrix.loc[:, col] = f_local.groupby("date", group_keys=False).apply(
                lambda x: apply_kernel(x, col)
            )
        return 0

    def detect_outlier_moons(self, norm_method="frobenius", out_method="zscore"):
        """Detect outlier epochs using specified normalization and outlier detection methods.

        This method computes a metric for each epoch using the chosen normalization method, and then detects outliers based on the specified outlier detection method.

        :param norm_method: str, optional
            Normalization method to compute the metric. Options: "frobenius", "determinant", "cond_n". Default is "frobenius".
        :param out_method: str, optional
            Outlier detection method. Currently supporting only "zscore". Default is "zscore".

        :return: tuple
            A tuple containing the detected outlier dates and a flag indicating if outliers were detected.
        """
        local_f_matrix = self.f_matrix

        if norm_method == "frobenius":

            def frob(x):
                cov = x.corr(numeric_only=True)
                norm = scipy.linalg.norm(cov)
                return norm

            metric = local_f_matrix.groupby("date").apply(lambda x: frob(x))

        elif norm_method == "determinant":

            def det(x):
                cov = x.corr(numeric_only=True)
                det_ = np.linalg.det(cov)
                return det_

            metric = local_f_matrix.groupby("date").apply(lambda x: det(x))
        elif norm_method == "cond_n":

            def cond_n(x):
                cov = x.corr(numeric_only=True)
                cond = np.linalg.cond(cov)
                return cond

            metric = local_f_matrix.groupby("date").apply(lambda x: cond_n(x))

        metric = pd.DataFrame(metric).reset_index()

        # select outlier epochs looking at metric dataframe:
        if out_method == "zscore":
            idx = detect_outliers_zscore(metric)
        # Currently supporting only zscore.

        dates = metric.loc[idx, "date"]
        if idx.sum() == 0:
            outliers_flag = False
        else:
            outliers_flag = True
        return dates, outliers_flag

    def pca(self, n_components=0.9):
        """Perform Principal Component Analysis (PCA) on the feature matrix.

        This method applies PCA to the feature matrix, reducing its dimensionality by retaining the specified proportion of explained variance.

        :param n_components: float, optional
            The proportion of variance to be retained. Default is 0.9.

        :return: int
            Always returns 0.
        """
        epochs = self.f_matrix[self.f_matrix.columns[0]].unique()

        pca = PCA(n_components=n_components)
        pca.fit(self.f_matrix[self.f_matrix.columns[1:]])

        f_pca = pd.DataFrame()
        f_pca["date"] = self.f_matrix["date"]
        f_pca[self.f_matrix.columns[1 : len(pca.explained_variance_ratio_) + 1]] = np.nan

        for epoch in tqdm(epochs):
            daily = self.f_matrix[self.f_matrix["date"] == epoch][self.f_matrix.columns[1:]]
            daily_pca = pca.transform(daily)
            f_pca.loc[
                f_pca["date"] == epoch,
                self.f_matrix.columns[1 : len(pca.explained_variance_ratio_) + 1],
            ] = daily_pca

        self.f_matrix = f_pca

        return 0

    def quantizer(self, rank=False, bins=[0.0325, 0.1465, 0.365, 0.635, 0.8535, 0.9675, 1]):
        """Perform quantization on the feature matrix.

        This method applies quantization to the feature matrix based on specified quantization bins. If 'rank' is True, it performs quantization on ranked data.

        :param rank: bool, optional
            If True, perform quantization on ranked data. Default is False.
        :param bins: list, optional
            The bin edges for quantization. Default is [0.0325, 0.1465, 0.365, 0.635, 0.8535, 0.9675, 1].

        :return: int
            Always returns 0.
        """
        if rank:
            quant = self.f_matrix.groupby("date", group_keys=False).transform(
                lambda x: hard_quantize(x.rank(pct=True, method="first"), bins)
            )
        else:
            quant = self.f_matrix.groupby("date", group_keys=False).transform(
                lambda x: hard_quantize(x, bins)
            )

        self.f_matrix = pd.concat([self.f_matrix["date"], quant], axis=1)
        return 0

    def cross_sectional_moments(self, to_moon=False):
        """Calculate cross-sectional moments of the feature matrix.

        This method calculates various statistical moments (standard deviation, skewness, and kurtosis) of the features in the feature matrix.

        :param to_moon: bool, optional
            If True, calculate moments based on moons. If False, calculate moments based on dates. Default is False.

        :return: pd.DataFrame
            DataFrame containing calculated cross-sectional moments.
        """
        df = self.f_matrix.copy()
        if to_moon:
            df["moon"] = df.date.astype("category").cat.codes
            df = df.drop(columns=["date"])
            grouper = "moon"
        else:
            grouper = "date"

        col = df.columns

        def moonwise_moments(df):
            results = pd.DataFrame()
            for col in df.columns[:-1]:  # loop only over features.
                col_names = [f"{col}_std", f"{col}_skew", f"{col}_kurt"]
                std = df[col].std()
                skew = stats.skew(df[col])
                kurt = stats.kurtosis(df[col])

                moments = {
                    f"{col_names[0]}": [std],
                    f"{col_names[1]}": [skew],
                    f"{col_names[2]}": [kurt],
                }
                results = pd.concat([results, pd.DataFrame(moments)], axis=1)

            return results

        df_moments = df.groupby(grouper).progress_apply(
            lambda x: moonwise_moments(x)
        )  # progress_apply adds progress bar
        df_moments = df_moments.reset_index().drop("level_1", axis=1)

        self.df_moments = df_moments
        return self.df_moments
