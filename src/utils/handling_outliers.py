import pandas as pd
import numpy as np

def detect_outliers_zscore(df, threshold=3.0):
    """
    df: Datframe
    threshold: multiple of std deviation
    
    Returns: Array with boolean values, True -> outlier
    """
    cols = df.columns
    outliers_idx = np.array([False] * len(df))
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        outliers_idx = np.logical_or(outliers_idx, np.abs(df[col] - mean) >= threshold * std)
    return outliers_idx


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
