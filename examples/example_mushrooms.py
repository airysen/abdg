# Dataset from https://archive.ics.uci.edu/ml/datasets/mushroom


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from imblearn.metrics import geometric_mean_score

from abdg import ABDGImput


RS = 334
gscore = make_scorer(geometric_mean_score, average='binary')


def gmean(y_true, y_pred):
    return geometric_mean_score(y_true, y_pred, average='binary')


mushrooms_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
attribute_list = ['target', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                  'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                  'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                  'stalk-surface-below-ring', 'stalk-color-above-ring',
                  'stalk-color-below-ring', 'veil-type', 'veil-color',
                  'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

mushroom = pd.read_csv(mushrooms_url, names=attribute_list)
mushroom = mushroom.replace('?', 'NAN')

# Missing Attribute Values: 2480 of them (denoted by "?"), all for
#    attribute #11 ('stalk-root').

train = mushroom.copy()
le_dict = {}
for col in train:
    if train[col].dtype == 'object':
        LE = LabelEncoder()
        train[col] = LE.fit_transform(train[col])
        le_dict[col] = LE

val = le_dict['stalk-root'].transform(['NAN'])[0]

train['stalk-root'] = train['stalk-root'].replace(val, np.nan)

X = train.drop('target', axis=1)
y = train['target']

imput = ABDGImput(categorical_features='all', n_iter=2, update_step=1000,
                  random_state=RS)

# 'bruises' instead of target, because of training on whole dataset
imput.fit(X.drop('bruises', axis=1), X['bruises'])
X_abdg, y_abdg = imput.predict(X.drop('bruises', axis=1), X['bruises'])
X_imp = X.copy()
X_imp['stalk-root'] = X_abdg['stalk-root']

rf = RandomForestClassifier()
parameters = {'n_estimators': [10 * k for k in range(2, 5)], 'criterion': ['entropy'],
              'max_features': np.arange(0.1, 0.5, 0.05).tolist(), 'max_depth': [10],
              'min_samples_split': [2, 4, 6, 8], 'min_samples_leaf': [1, 3, 5, 7],
              'min_weight_fraction_leaf': [0],
              'class_weight': ['balanced']}
# Drop NA
X_drop = X.dropna()
y_drop = y[X_drop.index]

# Fill with mode
X_mode = X.copy()
X_mode['stalk-root'] = X['stalk-root'].fillna(X['stalk-root'].mode()[0])

clf_drop = GridSearchCV(rf, parameters, scoring=gscore, n_jobs=-1, verbose=1, cv=5)
clf_drop.fit(X_drop, y_drop)

clf_mode = GridSearchCV(rf, parameters, scoring=gscore, n_jobs=-1, verbose=1, cv=5)
clf_mode.fit(X_mode, y)

clf_abdg = GridSearchCV(rf, parameters, scoring=gscore, n_jobs=-1, verbose=1, cv=5)
clf_abdg.fit(X_imp, y)

print('Dropped', clf_drop.best_score_)
print('Mode', clf_mode.best_score_)
print('ABDG', clf_abdg.best_score_)
