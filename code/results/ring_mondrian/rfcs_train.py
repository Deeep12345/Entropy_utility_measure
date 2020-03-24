import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from joblib import dump, load

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter("ignore")

def tune(k, X_test_orig, y_test_orig):
    k_data = pd.read_csv(f"../../anon_data/ring_mondrian/k{k}_minmaxed.csv").drop("Unnamed: 0", axis=1)
    y = k_data["class"]
    X = k_data.drop("class", axis=1)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=1)

    parameters = {
        'n_estimators':list(range(10,210,10)),
        'max_depth':[2,3,4,5,6,7],
        "max_features":list(range(2,10,2))
    }

    rfc = RandomForestClassifier()
    gridsearch = GridSearchCV(rfc, parameters, n_jobs=-1)
    gridsearch.fit(X_train, y_train)

    predicted = gridsearch.predict(X_test_orig)
    acc = np.sum(predicted==y_test_orig)/len(predicted)

    return gridsearch.best_estimator_, acc


###### Load original testing data ###########
print("Loading testing data...")
data = pd.read_csv(f"../../anon_data/ring_mondrian/k1_minmaxed.csv").drop("Unnamed: 0", axis=1)

y = data["class"]
X = data.drop("class", axis=1)

_, X_test_orig, _, y_test_orig = train_test_split(X, y,
    test_size=0.2, random_state=1)
############################################


###### Starting to train estimators ###########
print("Training RFCs...")

accs = {}

for k in list(range(1,51,2)) + list(range(100,3900,250)) + [7400]:

    rfc, acc = tune(k, X_test_orig, y_test_orig)
    dump(rfc, f'pickled_rfcs/{k}_rfc.joblib')
    accs[k] = acc
    print(f"best parameters: {rfc.get_params()}")
    print(f"best accuracy: {accs[k]}")

df = pd.DataFrame.from_dict(accs, orient="index")
df.to_csv("RFCs_accuracy.csv")
