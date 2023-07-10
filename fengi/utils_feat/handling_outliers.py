import numpy as np
import pandas as pd
from numpy import linalg as LA
from scipy.stats import kurtosis


def detect_outliers_zscore(df, threshold=3.0):
    """
    df: Datframe
    threshold: multiple of std deviation

    Returns: Array with boolean values, True -> outlier
    """
    """
    average moon statistic:
    
    """
    df = df.drop(columns=["date"])
    cols = df.columns
    outliers_idx = np.array([False] * len(df))
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        outliers_idx = np.logical_or(outliers_idx, np.abs(df[col] - mean) >= threshold * std)
    return outliers_idx


def detect_outliers_zscore_kurtosis(df, threshold=3.0):
    """
    df: Datframe
    threshold: kurtosis threshold value for identifying outliers

    Returns: Array with boolean values, True -> outlier
    """
    df = df.drop(columns=["date"])
    cols = df.columns
    outliers_idx = np.array([False] * len(df))
    for col in cols:
        kurt = kurtosis(df[col])
        outliers_idx = np.logical_or(outliers_idx, kurt > threshold)
    return outliers_idx


def detect_outliers_zscore_multivariate(df, threshold=3):
    """
    df: Datframe
    threshold: multiple of std deviation

    Returns: Array with boolean values, True -> outlier
    """
    # Calculate the mean and standard deviation of the data
    df = df.drop(columns=["date"])
    mean = np.mean(df.values, axis=0)
    std = np.std(df.values, axis=0)
    z_scores_list = []
    # Loop through each row of the data
    for i in range(df.shape[0]):
        # Calculate the Z-score between the mean and the current row
        row = df.iloc[i, :].values
        z_score = (row - mean) / std
        z_scores_list.append(z_score)

    z_scores = np.array(z_scores_list)
    outliers = np.where(LA.norm(z_scores, axis=1) > threshold)

    # Return the outlier indices
    return outliers[0]


def detect_outliers_quantile(df, multiplier=1.5):
    """
    df: Datframe
    threshold: multiple of std deviation

    Returns: Array with boolean values, True -> outlier
    """
    cols = df.columns
    outliers_idx = np.array([False] * len(df))
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        outliers_idx = np.logical_or(outliers_idx, (df[col] < lower_bound) | (df[col] > upper_bound))
    return outliers_idx
