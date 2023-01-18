import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import utils.gauss
from tqdm import tqdm
from sklearn.decomposition import PCA
from utils import gauss
from utils.quantization import quantize

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
        Returns: the f_exposure (measure of orthogonailty of f_matrix(master) to b_matrix(factors))
        """
        def loc_exposure(f_mat_temp):
            features = f_mat_temp.columns[1:]
            b_mat_temp = self.b_matrix.loc[self.b_matrix['date'].isin(f_mat_temp['date']), self.b_matrix.columns[1:]]
            fact_exp_lis = []
            for feature in features:
                M = np.array(f_mat_temp[feature])
                factor_exposure = np.dot(b_mat_temp.to_numpy().T, M) # vector
                fact_exp_lis.append(factor_exposure)
            fact_exp_matrix = np.array(fact_exp_lis)
            return fact_exp_matrix

        f_exp_matrix = self.f_matrix.groupby('date', group_keys=False).apply(lambda x: loc_exposure(x))
        return f_exp_matrix        

    def plot_corr(self, data, fig_name):
        """
        data: Matrix
        
        """
        Path('./paper/figures/').mkdir(parents=True, exist_ok=True)

        spear_corr = scipy.stats.spearmanr(data.drop(['date'], axis=1))[0]
        plt.figure(figsize=(15, 10))
        sns.heatmap(spear_corr)
        plt.savefig(f'./paper/figures/{fig_name}_spearman.png')

        plt.figure(figsize=(15, 10))
        sns.heatmap(data.drop(['date'], axis=1).corr())
        plt.savefig(f'./paper/figures/{fig_name}_pearson.png')
        return 0
    
    def plot_dist(self, data, fig_name, ndist=600, gbell=True):

        Path('./paper/figures/').mkdir(parents=True, exist_ok=True)

        for col in data.columns[1:]:
            mu = data[col].mean()
            sigma = data[col].std()
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            
            plt.figure(figsize=(15, 10))
            data[col].hist(bins=ndist, density=True)
            if gbell:
                plt.plot(x, scipy.stats.norm.pdf(x, mu, sigma))
            plt.title(f'{col}')
            plt.grid('on')
            plt.savefig(f'./paper/figures/{fig_name}_{col}.png')
        return 0

    def standardize(self):
        """

        return: updates the self.f_matrix to standardised version
        
        """
        def preprocess(local_f_matrix):
            """
            get sigma --> median of the std dev list
            
            """
            epochs = local_f_matrix[local_f_matrix.columns[0]].unique()
            for col in tqdm(local_f_matrix.columns[1:]):
                sigma_list = []
                for epoch in epochs:
                    M = local_f_matrix.loc[local_f_matrix['date'] == epoch, col]
                    sigma_list.append(M.std())
                sigma = np.median(sigma_list)
                for epoch in epochs:
                    local_f_matrix.loc[local_f_matrix['date'] == epoch, col] /= sigma
            
            return local_f_matrix

        self.f_matrix = preprocess(self.f_matrix)
        return 0

        
    def orthogonalize(self):
        """
        
        """
        def loc_orthogonalize(f_mat_temp):
            features = f_mat_temp.columns[1:]
            b_mat_temp = self.b_matrix.loc[self.b_matrix['date'].isin(f_mat_temp['date']), self.b_matrix.columns[1:]]

            for feature in features:
                m = np.array(f_mat_temp[feature])
                m_parallel = np.dot(b_mat_temp.to_numpy(), np.dot(np.linalg.pinv(b_mat_temp), m))
                m -= m_parallel
                f_mat_temp[feature] = m
            return f_mat_temp

        self.f_matrix = self.f_matrix.groupby('date', group_keys=False).apply(lambda x: loc_orthogonalize(x))
        return 0

    def gaussianize(self):

        f_local = self.f_matrix
        for col in tqdm(f_local.columns[1:]):
            gauss_kernel = gauss.Gaussianize(tol=1e-10, max_iter=1000)

            mn_tg = []
            std_tg = []
            kurt_tg = []
            skew_tg = []

            epochs = f_local[f_local.columns[0]].unique()
            for epoch in tqdm(epochs):
                f_local_epoch = f_local[col][f_local['date'] ==  epoch]
                std = f_local_epoch.std()
                skew = scipy.stats.skew(f_local_epoch)
                kurt = scipy.stats.kurtosis(f_local_epoch)
                
                mn_tg.append(f_local_epoch.mean())
                std_tg.append(std)
                skew_tg.append(skew)
                kurt_tg.append(kurt)
            mn_tg = np.array(mn_tg)
            std_tg = np.array(std_tg)
            skew_tg = np.array(skew_tg)
            kurt_tg = np.array(kurt_tg)

            moments = np.array([mn_tg, std_tg, skew_tg, kurt_tg])
            median_moments = np.median(moments, axis=1)
            for i in range(moments.shape[1]):
                moments[:, i] -= median_moments
                distance_moments = np.linalg.norm(moments, axis=0)
            train_sample = f_local[f_local['date'] == epochs[distance_moments.argmin()]][col]

            # Gaussianize
            gauss_kernel.fit(train_sample)

            for epoch in epochs:
                f_local_epoch = f_local[col][f_local['date'] ==  epoch]
                y = gauss_kernel.transform(f_local_epoch)
                self.f_matrix.loc[f_local['date'] ==  epoch, col] = np.squeeze(y)

        return 0


    def pca(self):
        """
        returns : updates f_matrix(nd) --> f_matrix(nd)
        """
        pca = PCA(n_components=len(self.f_matrix.columns[1:]))
        pca.fit(self.f_matrix[self.f_matrix.columns[1:]])

        f_pca = pd.DataFrame()
        f_pca['date'] = self.f_matrix['date']
        f_pca[self.f_matrix.columns[1:]] = np.nan
                    
        epochs = self.f_matrix[self.f_matrix.columns[0]].unique()
        for epoch in tqdm(epochs):
            daily = self.f_matrix[self.f_matrix['date'] ==  epoch][self.f_matrix.columns[1:]]
            daily_pca = pca.transform(daily)
            f_pca.loc[f_pca['date'] ==  epoch, self.f_matrix.columns[1:]] = daily_pca

        self.f_matrix = f_pca

        return 0


    def quantizer(self):
        """
        Returns: updated f_matrix after quantization

        """
        self.f_matrix = self.f_matrix.groupby('date', group_keys=False).apply(lambda x: quantize(x))
        return 0

