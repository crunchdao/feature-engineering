import pandas as pd
import scipy
import numpy as np

from utils.quantization import quantize
import matplotlib.pyplot as plt
from utils.quantization import hard_quantize

def tg_process(tg):
    
    targets = tg.drop('date', axis=1).columns
    epochs = tg['date'].unique()
    tg_out = pd.DataFrame()
    tg_out['date'] = tg['date']
    for target in targets:
        mn_tg = []
        std_tg = []
        kurt_tg = []
        skew_tg = []
        for epoch in epochs:
            tg_date = tg[target][tg['date'] ==  epoch]
            std = tg_date.std()
            skew = scipy.stats.skew(tg_date)
            kurt = scipy.stats.kurtosis(tg_date)
            
            mn_tg.append(tg_date.mean())
            std_tg.append(std)
            skew_tg.append(skew)
            kurt_tg.append(kurt)

        mn_tg = np.array(mn_tg)
        std_tg = np.array(std_tg)
        skew_tg = np.array(skew_tg)
        kurt_tg = np.array(kurt_tg)
        
        moments = np.array([mn_tg, std_tg, skew_tg, kurt_tg])
        mean_moments = np.mean(moments, axis=1)
        for i in range(moments.shape[1]):
            moments[:, i] -= mean_moments
            distance_moments = np.linalg.norm(moments, axis=0)
        train_sample = tg[tg['date'] == epochs[distance_moments.argmin()]][target]
        
        train_sample /= np.std(train_sample)
        quant_train = quantize(train_sample)

        bins = []
        for i in np.unique(quant_train):
            bins.append(np.count_nonzero(quant_train == i)/len(quant_train))
        for j in range(int(np.ceil(len(bins)/2 - 1))):
            bins[j] = (bins[j] + bins[-(j+1)])/2
            bins[-(j+1)] = bins[j]
        
        print(bins)
        for i in range(len(bins) - 1):
            bins[i+1] += bins[i]
        bins[-1] = 1 
        print(bins)

        quant = tg[['date', target]].groupby('date', group_keys=False).transform(lambda x: hard_quantize(x, bins))
        tg_out = pd.concat([tg_out, quant], axis=1)

    return tg_out