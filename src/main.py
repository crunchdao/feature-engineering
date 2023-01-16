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
    
    f_matrix = pd.read_parquet("/Users/utkarshpratiush/Cr_D/Feature engg/feature_engineering/data/f_matrix.parquet")
    b_matrix = pd.read_parquet("/Users/utkarshpratiush/Cr_D/Feature engg/feature_engineering/data/f_matrix.parquet")
    


    data = Data(f_matrix = f_matrix, b_matrix = b_matrix)
    #print(f_matrix)

    print(data.exposure())

    

if __name__ == '__main__':
    main()