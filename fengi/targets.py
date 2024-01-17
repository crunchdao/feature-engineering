"""Processing specific to labels."""
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from .utils_feat.quantization import hard_quantize, quantize


def tg_process(tg, rank=False, bits=7):
    """Process time-grouped data for each target variable in the input DataFrame.

    The function calculates statistical moments (mean, std, skewness, kurtosis) for each target variable
    within each time group. It then normalizes the moments, selects a reference time group, and quantizes
    the target variables based on the quantization bins derived from the reference group. If rank is True,
    it performs quantization based on rank percentiles; otherwise, it uses regular quantization.

    Note: The function assumes the input DataFrame has a "date" column for time grouping.
    """
    targets = tg.drop("date", axis=1).columns
    epochs = tg["date"].unique()
    tg_out = pd.DataFrame()
    tg_out["date"] = tg["date"]
    for target in targets:
        logging.info("-----------------")
        logging.info(target)
        mn_tg_list = []
        std_tg_list = []
        kurt_tg_list = []
        skew_tg_list = []
        for epoch in epochs:
            tg_date = tg[target][tg["date"] == epoch]
            std = tg_date.std()
            skew = scipy.stats.skew(tg_date)
            kurt = scipy.stats.kurtosis(tg_date)

            mn_tg_list.append(tg_date.mean())
            std_tg_list.append(std)
            skew_tg_list.append(skew)
            kurt_tg_list.append(kurt)

        mn_tg = np.array(mn_tg_list)
        std_tg = np.array(std_tg_list)
        skew_tg = np.array(skew_tg_list)
        kurt_tg = np.array(kurt_tg_list)

        moments = np.array([mn_tg, std_tg, skew_tg, kurt_tg])
        mean_moments = np.mean(moments, axis=1)
        for i in range(moments.shape[1]):
            moments[:, i] -= mean_moments
            distance_moments = np.linalg.norm(moments, axis=0)
        train_sample = tg[tg["date"] == epochs[distance_moments.argmin()]][target]

        logging.info(f"Bringing to Std=1 dividing by {np.std(train_sample)}")
        train_sample /= np.std(train_sample)
        quant_train = quantize(train_sample, bits=bits)
        logging.info(f"The unique bits values are: {np.unique(quant_train)}")

        bins = []
        for i in np.unique(quant_train):
            bins.append(np.count_nonzero(quant_train == i) / len(quant_train))
        for j in range(int(np.ceil(len(bins) / 2 - 1))):
            bins[j] = (bins[j] + bins[-(j + 1)]) / 2
            bins[-(j + 1)] = bins[j]

        for i in range(len(bins) - 1):
            bins[i + 1] += bins[i]
        bins[-1] = 1

        if rank:
            quant = (
                tg[["date", target]]
                .groupby("date", group_keys=False)
                .transform(
                    lambda x: hard_quantize(x.rank(pct=True, method="first"), bins)
                )
            )
        else:
            quant = (
                tg[["date", target]]
                .groupby("date", group_keys=False)
                .transform(lambda x: hard_quantize(x, bins))
            )

        tg_out = pd.concat([tg_out, quant], axis=1)

    return tg_out
