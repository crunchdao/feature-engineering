import pandas as pd
import numpy as np

class Data:

    

    def __init__(self, f_matrix, b_matrix ):
        """
        f_matrix:   features.parquet
        b_matrix:   factor_matrix.parquet
        """
        self.f_matrix = f_matrix
        self.b_matrix = b_matrix


    def exposure(self):
        """
        
        """
        def loc_exposure():
            b_mat_temp = self.b_matrix[self.b_matrix.columns[1:]]
            f_mat_temp = self.f_matrix
            features = f_mat_temp.columns[1:]
            
            fact_exp_matrix = []
            for feature in features:
                M = np.array(f_mat_temp[feature])
                factor_exposure = np.dot(b_mat_temp.to_numpy().T, M) # vector
                #print(factor_exposure)
                #fact_exp_matrix = np.concatenate((fact_exp_matrix, factor_exposure), axis=0) # fact_exp_list.append(factor_exposure)
                fact_exp_matrix.append(factor_exposure)
            # print(fact_exp_matrix)
            # fac_exp_mat = [np.concatenate((i), axis=1) for i in fact_exp_list]
            return fact_exp_matrix

        self.f_matrix.groupby('date', group_keys=False).apply(loc_exposure()) # f_exp_matrix = 
        return f_exp_matrix        

    def plot(data):
        """
        
        """


        
    def orthogonalize(self):
        """
        
        """
        def loc_orthogonalize():
            b_mat_temp = self.b_matrix
            f_mat_temp = self.f_matrix
            features = f_mat_temp.columns[1:]
            for feature in features:
                m = np.array(f_mat_temp[feature])
                m_parallel = np.dot(b_mat_temp.to_numpy(), np.dot(np.linalg.pinv(b_mat_temp), m))
                m -= m_parallel
                f_mat_temp[feature] = m
            return f_mat_temp

        self.f_matrix = self.f_matrix.groupby('date', group_keys=False).apply(loc_orthogonalize())
        pass




