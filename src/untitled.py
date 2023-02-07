import pandas as pd
import numpy as np
import scipy.stats as stats
from class_ import  Data

f_matrix = pd.read_parquet("../data/f_matrix.parquet")
b_matrix = pd.read_parquet("../data/f_matrix.parquet")

d = Data(f_matrix, b_matrix)
df_moments = d.cross_sectional_moments()
