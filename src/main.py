import pandas as pd
import numpy as np

import os

from class_ import Data

def get_data():
    """
    """    
    os.system('./data/get_data.sh')
    return 0


def main(get_data=False):
    """
    get_data: Boolean 
    
    """
    if get_data == True:
        get_data()
    
    f_matrix = pd.read_parquet("../data/f_matrix.parquet")
    b_matrix = pd.read_parquet("../data/b_matrix.parquet")


    data = Data(f_matrix = f_matrix, b_matrix = b_matrix)
    # exp0 = data.exposure()
    data.plot_corr(f_matrix, 'f_matrix0')
    data.plot_dist(f_matrix, 'f_hist0')

    f_orth = data.orthogonalize()
    # exp1 = data.exposure()
    # print(exp1)
    data.plot_dist(f_orth, 'f_hist1')
    data.plot_corr(f_orth, 'f_matrix1')
    f_gauss = data.gaussianize()
    data.plot_dist(f_orth, 'f_hist1')

if __name__ == '__main__':
    main()