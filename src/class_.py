from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from tqdm import tqdm
from tqdm.notebook import tqdm

tqdm.pandas()
import pdb as pdb

import scipy.stats as stats
from sklearn.decomposition import PCA

from utils import gauss
from utils.handling_outliers import detect_outliers_quantile, detect_outliers_zscore
from utils.quantization import hard_quantize


class Data:
    def __init__(self, f_matrix, b_matrix):
        """
        f_matrix:   features.parquet
        b_matrix:   factor_matrix.parquet
        """
        self.f_matrix = f_matrix
        self.b_matrix = b_matrix

    def exposure(self):
        """
        Returns: the f_exposure (measure of orthogonailty of f_matrix(master) to b_matrix(factors))
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

        f_exp_matrix = self.f_matrix.groupby("date", group_keys=False).apply(
            lambda x: loc_exposure(x)
        )
        return f_exp_matrix

    def plot_corr(self, data, fig_name):
        """
        data: Matrix

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
        """
        return: updates the self.f_matrix to standardised version
        """

        def preprocess(local_f_matrix):
            """
            get sigma --> median of the std dev list

            """
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

    def orthogonalize(self):
        """ """

        def loc_orthogonalize(f_mat_temp):
            features = f_mat_temp.columns[1:]
            b_mat_temp = self.b_matrix.loc[
                self.b_matrix["date"].isin(f_mat_temp["date"]),
                self.b_matrix.columns[1:],
            ]

            for feature in features:
                m = np.array(f_mat_temp[feature])
                m_parallel = np.dot(
                    b_mat_temp.to_numpy(), np.dot(np.linalg.pinv(b_mat_temp), m)
                )
                m -= m_parallel
                f_mat_temp[feature] = m
            return f_mat_temp

        self.f_matrix = self.f_matrix.groupby("date", group_keys=False).progress_apply(
            lambda x: loc_orthogonalize(x)
        )
        return 0

    def gaussianize(self):

        f_local = self.f_matrix.copy()
        for col in tqdm(f_local.columns[1:]):
            gauss_kernel = gauss.Gaussianize(tol=1e-10, max_iter=2000)

            def moments(x, col):
                f_local_epoch = x[col]
                std = f_local_epoch.std()
                kurt = scipy.stats.kurtosis(f_local_epoch)
                moments = {"std": [std], "kurt": [kurt]}
                return pd.DataFrame(moments)

            moments = f_local.groupby("date").apply(lambda x: moments(x, col))
            moments_array = np.array(moments)
            median_moments = np.median(moments_array, axis=0)
            distance_moments = []
            for i in range(moments_array.shape[0]):
                moments_array[i, :] -= median_moments
                distance_moments.append(np.linalg.norm(moments_array[i, :]))
            distance_moments_array = np.array(distance_moments)
            train_sample = f_local[
                f_local["date"]
                == f_local["date"].unique()[distance_moments_array.argmin()]
            ][col].to_numpy()

            # Gaussianize
            gauss_kernel.fit(train_sample, y=None)

            def apply_kernel(x, col):
                y = gauss_kernel.transform(x[col])
                x.loc[:, col] = np.squeeze(y)
                return x

            self.f_matrix.loc[:, col] = f_local.groupby("date", group_keys=False).apply(
                lambda x: apply_kernel(x, col)
            )
        return 0

    def detect_outlier_moons(self, norm_method="frobenius", out_method="zscore"):
        """Returns
        here f_matrix is post PCA.
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

    def pca(self):
        """
        returns : updates f_matrix(nd) --> f_matrix(nd)
        """

        epochs = self.f_matrix[self.f_matrix.columns[0]].unique()

        pca = PCA(n_components=0.9)
        pca.fit(self.f_matrix[self.f_matrix.columns[1:]])

        f_pca = pd.DataFrame()
        f_pca["date"] = self.f_matrix["date"]
        f_pca[
            self.f_matrix.columns[1 : len(pca.explained_variance_ratio_) + 1]
        ] = np.nan

        for epoch in tqdm(epochs):
            daily = self.f_matrix[self.f_matrix["date"] == epoch][
                self.f_matrix.columns[1:]
            ]
            daily_pca = pca.transform(daily)
            f_pca.loc[
                f_pca["date"] == epoch,
                self.f_matrix.columns[1 : len(pca.explained_variance_ratio_) + 1],
            ] = daily_pca

        self.f_matrix = f_pca

        return 0

    def quantizer(self, rank=False):
        """
        Returns: updated f_matrix after quantization

        """
        bins = [0.0325, 0.1465, 0.365, 0.635, 0.8535, 0.9675, 1]
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
        """Return DataFrame col: df.columns"""

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
