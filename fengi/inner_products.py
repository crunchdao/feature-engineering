"""Feature Engineering Module for Machine Learning Vectors.

This module provides functions for computing various distances and 
transformations between vectors generated from Machine Learning models.
"""

import numpy as np
import pandas as pd


def B_to_BM_epoch(
    b_epoch: pd.DataFrame, risk_factors: list[str], symbol_column: str
) -> pd.DataFrame:
    """Compute Projection Matrix and Concatenate it to the factor matrix for a specific epoch."""
    b_epoch_risks = b_epoch[risk_factors]

    b_epoch_array = b_epoch_risks.to_numpy()
    b_pinv = np.linalg.pinv(b_epoch_array)

    M = b_epoch_array @ b_pinv  # Compute Projection Matrix

    M_df = pd.DataFrame(M, columns=b_epoch[symbol_column], index=b_epoch.index)
    BM_epoch = pd.concat([b_epoch_risks, M_df], axis=1)
    return BM_epoch


def concatenate_risk_dimensions_and_projection_matrix(
    dataframe: pd.DataFrame,
    betas_matrix: pd.DataFrame,
    time_column: str,
    symbol_column: str,
    risk_factors: list[str],
) -> pd.DataFrame:
    """Compute Projection Matrix and concatenate it to features and factors."""
    BM = betas_matrix.groupby(time_column, group_keys=False).apply(
        B_to_BM_epoch, risk_factors, symbol_column
    )
    return pd.concat([dataframe, BM], axis=1)
