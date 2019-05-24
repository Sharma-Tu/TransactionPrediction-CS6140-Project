# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 23:55:25 2019

@author: Tushar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#plt.style.use('bmh')
#plt.rcParams['figure.figsize'] = (10, 10)
#title_config = {'fontsize': 20, 'y': 1.05}

train = pd.read_csv("./data/train.csv")

variables = train.iloc[:,2:].astype('float64')
target = train['target'].values

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(variables, target, train_size = 
                                                    0.8, test_size = 0.2,
                                                   random_state=5)


X_train.shape

from sklearn import linear_model

clf = linear_model.SGDClassifier(loss = 'log',
                                 max_iter = 1e5,
                                 learning_rate = 'adaptive',
                                 eta0 = 1e-5,
                                 tol = 1e-7,
                                 fit_intercept = True,
                                 shuffle=False,
                                 n_jobs=-2)

from sklearn.metrics import roc_curve, auc

y_score = clf.predict_proba(X_test)

print(y_score.T[1])

fpr = dict()
tpr = dict()
roc_auc = dict()

fpr, tpr, _ = roc_curve(y_test, y_score.T[1], pos_label=1)
roc_auc = auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

import pandas as pd
test = pd.read_csv('./data/test.csv')
X_test = test.iloc[:,1:].values.astype('float64')
submission = pd.read_csv('./data/sample_submission.csv')

submission['target'] = clf.predict_proba(X_test)[:,1]
submission.to_csv('submissionLG.csv', index=False)
submission.head()

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_train,y_train, cv=5)
scores

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=10)


classifier = linear_model.SGDClassifier(loss = 'log',
                                 max_iter = 1e5,
                                 learning_rate = 'adaptive',
                                 eta0 = 1e-5,
                                 tol = 1e-7,
                                 fit_intercept = True,
                                 shuffle=False,
                                 n_jobs=-2)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

variables = np.asarray(variables)
target = np.asarray(target)

i = 0
for train, test in cv.split(variables, target):
    probas_ = classifier.fit(variables[train], target[train]).predict_proba(variables[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(target[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.4f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Area Under ROC- Logistic Classifier')
plt.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
plt.show()

