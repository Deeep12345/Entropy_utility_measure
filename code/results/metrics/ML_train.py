import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from joblib import dump, load

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.simplefilter("ignore")

def tune(anon_data, X_test_orig, y_test_orig, type):
    y = anon_data["class"]
    X = anon_data.drop("class", axis=1)
    X_train = X.drop(X_test_orig.index, axis=0)
    y_train = y.drop(y_test_orig.index, axis=0)


    if type == "logreg":
        model = LogisticRegression(random_state=1)
    elif type == "RF":
        parameters = {
            'n_estimators':list(range(100,500,25)),
            'max_depth':[2,3,4,5,6,7,8,None],
            "max_features":list(range(2,10,2))
        }
        model = GridSearchCV(model, parameters, n_jobs=-1)


    model.fit(X_train, y_train)
    predicted = model.predict(X_test_orig)
    acc = np.sum(predicted==y_test_orig)/len(predicted)

    if type == "logreg":
        test_proba = model.predict_proba(X_test_orig)
        roc_auc = roc_auc_score(list(y_test_orig), test_proba[:, 1])
    else:
        roc_auc= None

    return roc_auc, acc


def train_test(orig_data, anon_data, type):
    y_orig = orig_data["class"]
    X_orig = orig_data.drop("class", axis=1)

    _, X_test_orig, _, y_test_orig = train_test_split(X_orig, y_orig,
        test_size=0.2, random_state=1)

    roc, acc = tune(anon_data, X_test_orig, y_test_orig, type)
    return roc, acc
