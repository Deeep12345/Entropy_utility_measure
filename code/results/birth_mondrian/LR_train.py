import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from joblib import dump, load

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter("ignore")

def tune(k, X_test_orig, y_test_orig):
    k_data = pd.read_csv(f"../../anon_data/birth_mondrian/k{k}_minmaxed.csv").drop("Unnamed: 0", axis=1)
    y = k_data["class"]
    X = k_data.drop("class", axis=1)
    X_train = X.drop(X_test_orig.index, axis=0)
    y_train = y.drop(y_test_orig.index, axis=0)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    predicted = lr.predict(X_test_orig)
    acc = np.sum(predicted==y_test_orig)/len(predicted)

    return acc


###### Load original testing data ###########
print("Loading testing data...")
data = pd.read_csv(f"../../anon_data/birth_mondrian/k1_minmaxed.csv").drop("Unnamed: 0", axis=1)

y = data["class"]
X = data.drop("class", axis=1)

_, X_test_orig, _, y_test_orig = train_test_split(X, y,
    test_size=0.2, random_state=1)
############################################


###### Starting to train estimators ###########
print("Training LRs...")

accs = {}
results = {}
for k in range(1,101):
    results[k] = tune(k, X_test_orig, y_test_orig)
    print(f"####### K:{k} ##########")

print(results)
df = pd.DataFrame.from_dict(results, orient='index')
df.to_csv(f"lr_acc.csv")
