"""Tests for hello function."""
import random

import numpy as np
import pandas as pd
import pytest

from fengi.inner_products import concatenate_risk_dimensions_and_projection_matrix

date_value = "2023-01-01"
symbols = [
    "XYQKH",
    "WBIKE",
    "XLUMG",
    "ZNMJE",
    "KFEWL",
    "RHMVI",
    "GYPUY",
    "XONZR",
    "IGHJM",
    "OWGSR",
    "FRJJW",
    "ROKEW",
    "SBTYK",
    "FFCNO",
    "NJRYF",
    "LGSOT",
    "SZEHI",
    "WNWPA",
    "ENOAO",
    "JALCN",
    "LRBPD",
    "KTILO",
    "GSFWX",
    "DGBJJ",
    "PZMPH",
    "AITHF",
    "WASTF",
    "SVRGN",
    "ILRRJ",
    "KSEZS",
    "ZQEBZ",
    "GYETH",
    "DVEWX",
    "FLYFR",
    "XHCHI",
    "GVERB",
    "DRUEL",
    "EBAHW",
    "KDQTN",
    "YSCZD",
    "UUBLB",
    "KMRPN",
    "OPUVM",
    "VNSHB",
    "KNQTH",
    "SJCGZ",
    "LOUPP",
    "DYMLW",
    "ITATO",
    "INDDD",
]

n_factors = 4
B = np.random.rand(len(symbols), n_factors)

betas_matrix = pd.DataFrame(B)
betas_matrix.columns = betas_matrix.columns.astype(str)

risk_factors = betas_matrix.columns.to_list()
betas_matrix["symbol"] = symbols
betas_matrix["date"] = date_value

subset_size = 40
subset = random.sample(symbols, subset_size)

data = {
    "date": [date_value] * len(symbols),
    "symbol": symbols,
    "value": np.random.rand(len(symbols)),
}

dataframe = pd.DataFrame(data)


def test_orthogonality_after_concatenation():
    df_B_M = concatenate_risk_dimensions_and_projection_matrix(
        dataframe, betas_matrix, "date", "symbol", risk_factors
    )
    value_post = df_B_M["value"]
    B_post = df_B_M[risk_factors]

    M_columns = [c for c in df_B_M.columns if "Projector" in c]
    M = df_B_M[M_columns].to_numpy()

    assert np.allclose(M, M.T)
    assert np.allclose(M @ M, M)

    value_post -= np.dot(M, value_post)
    assert np.allclose(B_post.T @ value_post, 0)
