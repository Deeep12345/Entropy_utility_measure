import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from joblib import dump, load

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter("ignore")

def tune(anon_data, X_test_orig, y_test_orig):
    y = anon_data["class"]
    X = anon_data.drop("class", axis=1)
    X_train = X.drop(X_test_orig.index, axis=0)
    y_train = y.drop(y_test_orig.index, axis=0)

    parameters = {
        'n_estimators':list(range(10,210,10)),
        'max_depth':[2,3,4,5,6,7,8,None],
        "max_features":list(range(2,10,2))
    }

    rfc = RandomForestClassifier()
    gridsearch = GridSearchCV(rfc, parameters, n_jobs=-1)
    gridsearch.fit(X_train, y_train)

    predicted = gridsearch.predict(X_test_orig)
    acc = np.sum(predicted==y_test_orig)/len(predicted)

    return gridsearch.best_estimator_, acc


def train_test(orig_data, anon_data):
    y_orig = orig_data["class"]
    X_orig = orig_data.drop("class", axis=1)

    _, X_test_orig, _, y_test_orig = train_test_split(X_orig, y_orig,
        test_size=0.2, random_state=1)

    est, acc = tune(anon_data, X_test_orig, y_test_orig)
    return acc
