import numpy as np
import pandas as pd
from collections import Counter

from sklearn.svm import SVC
from imblearn.metrics import geometric_mean_score

from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_iris

from abdg import ABDGImput


RS = 334
gscore = make_scorer(geometric_mean_score, average='binary')


def gmean(y_true, y_pred):
    return geometric_mean_score(y_true, y_pred, average='binary')


def rmse_func(y_true, y_pred,
              sample_weight=None,
              multioutput='uniform_average'):
    return np.fabs(mean_squared_error(y_true, y_pred,
                                      sample_weight, multioutput)) ** 0.5


iris = load_iris()
train = pd.DataFrame(iris.data, columns=['sepal length (cm)',
                                         'sepal width (cm)',
                                         'petal length (cm)',
                                         'petal width (cm)'])
target = pd.Series(iris.target, name='target')

idx = target.index.values
for n in range(5):
    np.random.seed(RS + n)
    idx = np.random.permutation(idx)

train.reindex(idx)
target.reindex(idx)

# imputation only on train
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=RS)

# induce missing values on 50% of values of a random column
N = target.shape[0] // 2
np.random.seed(RS)
idx = np.arange(X_train.shape[0])
cols = np.arange(X_train.shape[1])

# select random indices
na_index = np.random.choice(idx, size=N, replace=False)

# select a random column
na_col = np.random.choice(cols, size=1, replace=False)

X_miss = X_train.copy()
X_miss.iloc[na_index, na_col] = np.nan
y_miss = y_train.copy()

RS = RS + 100

rf = RandomForestClassifier()
params = {'class_weight': 'balanced',
          'criterion': 'entropy',
          'max_depth': 10,
          'max_features': 0.5,
          'min_samples_leaf': 1,
          'min_samples_split': 8,
          'min_weight_fraction_leaf': 0,
          'n_estimators': 20}
rf.set_params(**params)

abdg = ABDGImput(categorical_features='auto', n_iter=4, alpha=0.6, L=0.5,
                 sampling='normal', update_step=1, random_state=RS)

abdg.fit(X_miss, y_miss)
X_abdg, y_abdg = abdg.predict(X_miss, y_miss)

true = X_train.iloc[na_index, na_col]
predict = X_abdg.iloc[na_index, na_col]

print(rmse_func(np.log(true), np.log(predict)))
