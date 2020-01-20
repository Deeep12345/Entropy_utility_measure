import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

adult = pd.read_csv('../adult/adult.csv', delimiter=', ')
data = adult

data['salary'] = data['salary'].map({ "<=50K": 0, ">50K": 1 })
data = pd.get_dummies(data, columns=[
    "workclass", "education", "marital-status", "occupation", "relationship",
    "race", "sex", "native-country",
])

data.to_csv('../adult/clean_adult.csv')
#X_train, X_test, y_train, y_test = train_test_split(
#    data, y, test_size=0.25, stratify=y)

#print(f'Sizes:\n\tX_train: {X_train.shape}\n\tX_test: {X_test.shape}\
#                \n\ty_train: {y_train.shape}\n\ty_test: {y_test.shape}')
#

# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

#
# param_grid = {
#     'solver':['lbfgs', 'sgd', 'adam'],
#     'activation':['relu'],
#     'alpha':[0.0001, 0.001, 0.01, 0.1],
#     'learning_rate':['constant', 'adaptive'],
#     'batch_size':[64, 128, 256],
#     'hidden_layer_sizes':[(200, 100, 50, 25,), (200, 200, 100,), (200,), (200, 50, 25, 10)]
# }
#
# nn = MLPClassifier()
#
# best_nn = GridSearchCV(nn, param_grid, n_jobs=-1, cv=5, verbose=2)
# best_nn.fit(X_train, y_train)
#
# print("best params: ", best_nn.best_params_)
# print("best acc: ", best_nn.best_score_)
