# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 23:57:52 2019

@author: Tushar
"""

import pandas as pd
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

train = pd.read_csv("./data/train.csv")

variables1 = train.iloc[:,2:].astype('float64')
target = train['target'].values

variables = scaler.fit_transform(variables1)
variables = pd.DataFrame(variables)
variables.columns = variables1.columns.values.tolist()

variables.head()

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(variables, target, train_size = 
                                                    0.8, test_size = 0.2,
                                                   random_state=9)


print(X_train.shape, sum(y_train), sum(y_test))

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
import multiprocessing as mp
"""
def f(x):
    return x*x

if __name__ == '__main__':
    p = Pool(5)
    print(p.map(f, [1, 2, 3]))
"""
class KDEClassifier(BaseEstimator, ClassifierMixin):
    """Bayesian generative classification based on KDE
    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    """
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
     #1/(np.sqrt(len(i[j])))
    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = []
        for i in training_sets:
            self.models_.append([KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel).fit(np.asarray(i[j]).reshape(-1,1)) for j in i])
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets]
        return self
    
    def predict_proba_pool(self, X, mdlpool):
        return(mdlpool.score_samples(X))
            
    def predict_proba(self, X):
        logprobs = []
        pool = mp.Pool(mp.cpu_count())
        for mdl in self.models_:
            logprobs.append(np.array(pool.starmap(self.predict_proba_pool, [(X.iloc[:,(mdl.index(mdlclass)):(mdl.index(mdlclass)+1)], 
                                                                            mdlclass) for mdlclass in mdl])).T)
        pool.close()
        logprobs_ = np.array([np.sum(i, axis = 1) for i in logprobs]).T
        result = np.exp(logprobs_ + self.logpriors_)
        return result / result.sum(1, keepdims=True)
    
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


#from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from timeit import default_timer as timer
#digits = load_digits()

start = timer()
#bandwidths = 10 ** np.linspace(0, 2, 10)
#grid = GridSearchCV(KDEClassifier(),  {'bandwidth': [2.8]}, n_jobs = -1, cv=None)
#grid.fit(X_train, y_train)

grid = KDEClassifier.fit(KDEClassifier(0.4,'gaussian'),X_train,y_train)
gridTime = timer()-start
print(gridTime)

"""
import matplotlib.pyplot as plt
plt.semilogx(bandwidths, scores)
plt.xlabel('bandwidth')
plt.ylabel('accuracy')
plt.title('KDE Model Performance')
print(grid.best_params_)
print('accuracy =', grid.best_score_)
"""

from sklearn.metrics import roc_curve, auc
import  matplotlib.pyplot as plt

#logprobs_ = grid.predict_proba(X_test)

#Xt = np.asarray(X_test.iloc[:,0:10])
fpr, tpr, thr = roc_curve(y_test, logprobs_[:,1])
roc_auc = auc(fpr, tpr)
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - Non Parametric')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import accuracy_score
accuracy_score(y_test, np.argmax(logprobs_, 1), normalize=True)


