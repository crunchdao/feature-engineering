import os
import pdb as pdb

import numpy as np
import pandas as pd
from class_ import Data
from targets import tg_process

from .utils_feat.handling_outliers import (
    detect_outliers_quantile,
    detect_outliers_zscore,
    detect_outliers_zscore_multivariate,
)
from .utils_feat.quantization import quantize


def get_data():
    """ """
    os.system("./data/get_data.sh")
    return 0


def main(gd=True):
    """
    get_data: Boolean

    """

    if gd:
        get_data()

    print("----------Targets quantization start --------------------------------")
    targets = pd.read_parquet("./data/target.parquet")
    targets = tg_process(targets)
    print(targets.head())
    print(targets.describe())
    print("----------Targets quantization done--------------------------------")

    f_matrix = pd.read_parquet("./data/f_matrix.parquet")
    b_matrix = pd.read_parquet("./data/b_matrix.parquet")

    data = Data(f_matrix=f_matrix, b_matrix=b_matrix)
    data.plot_dist(targets, fig_name="check_tg_dist", ndist=100, gbell=False)

    print("----------Data reading done --------------------------------")
    print(data.f_matrix.head(), data.exposure().head())
    print("----------Data orthogonalization start --------------------------------")
    data.orthogonalize()
    print("----------Data orthogonalization done --------------------------------")

    # Plot corr
    data.plot_corr(data.f_matrix, fig_name="check1_corr")
    # Plot dist
    data.plot_dist(data.f_matrix, fig_name="check1_dist")

    f_matrix_o = data.f_matrix.copy()
    outliers_flag = True
    while outliers_flag:
        print(data.f_matrix.head(), data.exposure().head())
        print("----------Data Gaussianize start --------------------------------")
        data.gaussianize()
        print("----------Data Gaussianize done --------------------------------")

        # Plot corr
        data.plot_corr(data.f_matrix, fig_name="check2_corr")
        # Plot dist
        data.plot_dist(data.f_matrix, fig_name="check2_dist")

        print(data.f_matrix.head(), data.exposure().head())
        print("----------Data Orthogonalization start --------------------------------")
        data.orthogonalize()
        print("----------Data Orthogonalization done --------------------------------")
        print(data.f_matrix.head(), data.exposure().head())

        # Plot corr
        data.plot_corr(data.f_matrix, fig_name="check3_corr")
        # Plot dist
        data.plot_dist(data.f_matrix, fig_name="check3_dist")

        print("----------Data standarization start --------------------------------")
        data.standardize()
        print("----------Data standarization done --------------------------------")
        print(data.f_matrix.head(), data.exposure().head())

        # PCA
        print("----------PCA on f_matrix start --------------------------------")
        data.pca()
        print("----------PCA on f_matrix end --------------------------------")
        print(data.f_matrix.head(), data.exposure().head())

        # Outliers
        print("----------Outlier detection on f_matrix start --------------------------------")
        dates, outliers_flag = data.detect_outlier_moons()
        print(
            f"----------Outlier detection on f_matrix end, {dates.count()} outliers detected. --------------------------------"
        )
        if outliers_flag:
            f_matrix_o.drop(f_matrix_o.index[f_matrix_o["date"].isin(dates)], inplace=True)

            data.f_matrix = f_matrix_o
            data.b_matrix.drop(b_matrix.index[b_matrix["date"].isin(dates)], inplace=True)
        else:
            break

    print("----------standarize start--------------------------------")
    # Standardize
    data.standardize()
    print("----------standarize end--------------------------------")
    print(data.f_matrix.head(), data.exposure().head())

    # Plot corr
    data.plot_corr(data.f_matrix, fig_name="check4_corr")
    # Plot dist
    data.plot_dist(data.f_matrix, fig_name="check4_dist")

    print("----------Data quantization start --------------------------------")
    data.quantizer()
    print("----------Data quantization done --------------------------------")
    print(data.f_matrix.head(), data.exposure().head())

    # Plot corr
    data.plot_corr(data.f_matrix, fig_name="check5_corr")
    # Plot dist
    data.plot_dist(data.f_matrix, fig_name="check5_dist", ndist=100, gbell=False)


if __name__ == "__main__":
    main(gd=False)
