"""This module contains functions for data processing and analysis.

Public Functions:
- get_data(): Downloads data using a shell script and returns 0.
- main(gd=True): Main function for data processing and analysis.

Usage:
    if __name__ == "__main__":
        main(gd=False)
"""

import logging
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
    """Download Data From IPFS."""
    os.system("./data/get_data.sh")
    return 0


def main(gd=True):
    """Example of Feature Engineering Pipeline."""
    if gd:
        get_data()

    logging.info(
        "----------Targets quantization start --------------------------------"
    )
    targets = pd.read_parquet("./data/target.parquet")
    targets = tg_process(targets)
    logging.info(targets.head())
    logging.info(targets.describe())
    logging.info("----------Targets quantization done--------------------------------")

    f_matrix = pd.read_parquet("./data/f_matrix.parquet")
    b_matrix = pd.read_parquet("./data/b_matrix.parquet")

    data = Data(f_matrix=f_matrix, b_matrix=b_matrix)
    data.plot_dist(targets, fig_name="check_tg_dist", ndist=100, gbell=False)

    logging.info("----------Data reading done --------------------------------")
    logging.info(data.f_matrix.head(), data.exposure().head())
    logging.info(
        "----------Data orthogonalization start --------------------------------"
    )
    data.orthogonalize()
    logging.info(
        "----------Data orthogonalization done --------------------------------"
    )

    # Plot corr
    data.plot_corr(data.f_matrix, fig_name="check1_corr")
    # Plot dist
    data.plot_dist(data.f_matrix, fig_name="check1_dist")

    f_matrix_o = data.f_matrix.copy()
    outliers_flag = True
    while outliers_flag:
        logging.info(data.f_matrix.head(), data.exposure().head())
        logging.info(
            "----------Data Gaussianize start --------------------------------"
        )
        data.gaussianize()
        logging.info("----------Data Gaussianize done --------------------------------")

        # Plot corr
        data.plot_corr(data.f_matrix, fig_name="check2_corr")
        # Plot dist
        data.plot_dist(data.f_matrix, fig_name="check2_dist")

        logging.info(data.f_matrix.head(), data.exposure().head())
        logging.info(
            "----------Data Orthogonalization start --------------------------------"
        )
        data.orthogonalize()
        logging.info(
            "----------Data Orthogonalization done --------------------------------"
        )
        logging.info(data.f_matrix.head(), data.exposure().head())

        # Plot corr
        data.plot_corr(data.f_matrix, fig_name="check3_corr")
        # Plot dist
        data.plot_dist(data.f_matrix, fig_name="check3_dist")

        logging.info(
            "----------Data standarization start --------------------------------"
        )
        data.standardize()
        logging.info(
            "----------Data standarization done --------------------------------"
        )
        logging.info(data.f_matrix.head(), data.exposure().head())

        # PCA
        logging.info("----------PCA on f_matrix start --------------------------------")
        data.pca()
        logging.info("----------PCA on f_matrix end --------------------------------")
        logging.info(data.f_matrix.head(), data.exposure().head())

        # Outliers
        logging.info(
            "----------Outlier detection on f_matrix start --------------------------------"
        )
        dates, outliers_flag = data.detect_outlier_moons()
        logging.info(
            f"----------Outlier detection on f_matrix end, {dates.count()} outliers detected. --------------------------------"
        )
        if outliers_flag:
            f_matrix_o.drop(
                f_matrix_o.index[f_matrix_o["date"].isin(dates)], inplace=True
            )

            data.f_matrix = f_matrix_o
            data.b_matrix.drop(
                b_matrix.index[b_matrix["date"].isin(dates)], inplace=True
            )
        else:
            break

    logging.info("----------standarize start--------------------------------")
    data.standardize()
    logging.info("----------standarize end--------------------------------")
    logging.info(data.f_matrix.head(), data.exposure().head())

    # Plot corr
    data.plot_corr(data.f_matrix, fig_name="check4_corr")
    # Plot dist
    data.plot_dist(data.f_matrix, fig_name="check4_dist")

    logging.info("----------Data quantization start --------------------------------")
    data.quantizer()
    logging.info("----------Data quantization done --------------------------------")
    logging.info(data.f_matrix.head(), data.exposure().head())

    # Plot corr
    data.plot_corr(data.f_matrix, fig_name="check5_corr")
    # Plot dist
    data.plot_dist(data.f_matrix, fig_name="check5_dist", ndist=100, gbell=False)


if __name__ == "__main__":
    main(gd=False)
