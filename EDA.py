# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 23:52:53 2019

@author: Tushar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.style.use('bmh')
plt.rcParams['figure.figsize'] = (10, 10)
title_config = {'fontsize': 20, 'y': 1.05}

train = pd.read_csv("./data/train.csv")


variables = train.iloc[:,2:].astype('float64')
target = train['target'].values

plt.imshow(variables[target==1].corr())
plt.colorbar()
plt.title('Correlation Matrix - Positive Class');
plt.savefig('./Plots/Correlation/pairwisepositiveclass.png')

# label density plots
plt.imshow(variables[target==0].corr())
plt.colorbar()
plt.title('Correlation Matrix - Negative Class');
plt.savefig('./Plots/Correlation/pairwisenegativeclass.png')

# label density plots
plt.imshow(variables[target==0].corr())
plt.colorbar()
plt.title('Correlation Matrix - Negative Class');
plt.savefig('./Plots/Correlation/pairwisenegativeclass.png')

#positive label density plots
plt.style.use('bmh')
plt.rcParams['figure.figsize'] = (10, 10)
title_config = {'fontsize': 20, 'y': 1.05}
plt.ioff()


for i in tqdm(variables):
    fig = plt.figure()
    pd.DataFrame(variables[i].iloc[target==1]).plot.kde(legend = False)
    plt.savefig('./Plots/DensityPlotsPositive/%s.png' %(i))
    plt.close(fig)
    
# mutual independence check
from sklearn.feature_selection import mutual_info_regression
mi = mutual_info_regression(variables, target, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)

from pandas.plotting import scatter_matrix
#positiive class
scatter_matrix(variables[target==1].iloc[:,0:5], alpha=0.2, figsize=(6, 6), diagonal='kde')

from pandas.plotting import scatter_matrix
#negative class
scatter_matrix(variables[target==0].iloc[:,0:5], alpha=0.2, figsize=(6, 6), diagonal='kde')

import seaborn as sns
import matplotlib.pyplot as plt
corr = variables[target==0].corr()
sns.heatmap(corr, vmax=1., cmap='inferno', square=True).set_title('Correlation Matrix - Negative Class')


    