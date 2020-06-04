import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from joblib import dump, load

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neural_network

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import warnings
warnings.simplefilter("ignore")

def logreg(X_train, y_train):
    model = LogisticRegression(random_state=1, max_iter=500, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def randomforest(X_train, y_train):
    no_attrs = len(X_train[0])
    max_feats = 10 if no_attrs >= 10 else no_attrs
    parameters = {
        'n_estimators':list(range(100,500,50)),
        'max_depth':[2,4,6,8,None],
        "max_features":list(range(1,max_feats,2))
    }
    rf = RandomForestClassifier()
    model = GridSearchCV(rf, parameters, verbose=0, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def k_neighbors(X_train, y_train):
    parameters = {
        'n_neighbors':[3, 5, 7, 9, 11],
        "weights":["uniform", "distance"]
    }
    knn = KNeighborsClassifier()
    model = GridSearchCV(knn, parameters, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


# def mlp(X_train, y_train):
#     parameters = {
#         'solver': ['lbfgs'],
#         'max_iter': [1500],
#         'alpha': [0.01],
#         'hidden_layer_sizes': [(50, 100, 25)],
#     }
#     model = GridSearchCV(neural_network.MLPClassifier(), parameters, n_jobs=-1)
#     model.fit(X_train, y_train)
#     return model


def get_accuracy(model, X_test, y_test):
    predicted = model.predict(X_test)
    acc = np.sum(predicted==y_test)/len(predicted)
    return acc


def get_auroc(model, X_test, y_test):
    if len(set(y_test)) == 2:
        test_proba = model.predict_proba(X_test)
        roc_auc = roc_auc_score(list(y_test), test_proba[:, 1])
    else:
    #multiclass
        test_proba = model.predict_proba(X_test)
        roc_auc = roc_auc_score(list(y_test), test_proba,
                                multi_class='ovr')
    return roc_auc


def prepare_data(orig_data, anon_data, use_pca=False):
    y_orig = orig_data["class"]
    X_orig = orig_data.drop("class", axis=1)

    _, X_test_orig, _, y_test_orig = train_test_split(X_orig, y_orig,
        test_size=0.2, random_state=1)

    y = anon_data["class"]
    X = anon_data.drop("class", axis=1)
    X_train = X.drop(X_test_orig.index, axis=0)
    y_train = y.drop(y_test_orig.index, axis=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test_orig = scaler.fit_transform(X_test_orig)

    if use_pca:
        x_sh = X_train.shape
        pca = PCA(.95)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        print(f"PCA {x_sh} -----> {X_train.shape}")
        X_test_orig = pca.transform(X_test_orig)

    return X_train, y_train, X_test_orig, y_test_orig




def train_test(orig_data, anon_data, type, use_pca=False):
    X_train, y_train, X_test_orig, y_test_orig = prepare_data(orig_data, anon_data, use_pca)


    if type == "lr":
        model = logreg(X_train, y_train)
    elif type == "rf":
        model = randomforest(X_train, y_train)
    elif type == "mlp":
        model = mlp(X_train, y_train)
    elif type == "knn":
        model = k_neighbors(X_train, y_train)
    acc = get_accuracy(model, X_test_orig, y_test_orig)
    roc_auc = get_auroc(model, X_test_orig, y_test_orig)

    return roc_auc, acc
