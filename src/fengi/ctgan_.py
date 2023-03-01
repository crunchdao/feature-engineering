"""https://github.com/sdv-dev/CTGAN"""
"""https://sdv.dev/SDV/index.html"""
"""https://docs.sdv.dev/sdmetrics/getting-started/quickstart"""
import ctgan
from ctgan import CTGAN
from ctgan import load_demo
import pdb
import pandas as pd

# real_data = load_demo()

# # Names of the columns that are discrete
# discrete_columns = [
#     'workclass',
#     'education',
#     'marital-status',
#     'occupation',
#     'relationship',
#     'race',
#     'sex',
#     'native-country',
#     'income']

#real_data = pd.read_parquet("./data/f_matrix.parquet").drop(columns=["date"])
real_data = pd.read_parquet("./data/f_matrix.parquet")
real_data["moon"] = real_data.date.astype('category').cat.codes
real_data = real_data.drop(columns=['date'])
#discrete_columns = ["date"]

ctgan = CTGAN(epochs=100)
#ctgan.fit(real_data, discrete_columns)
ctgan.fit(real_data)

# Create synthetic data
synthetic_data = ctgan.sample(1000)
# check synthetic_data.moon.unique() --> increase epochs and see 
pdb.set_trace()
