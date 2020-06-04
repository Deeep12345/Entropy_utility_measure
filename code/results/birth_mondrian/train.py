import pandas as pd
import numpy as np

from ML_train import train_test

import warnings
warnings.simplefilter("ignore")

###### Load original testing data ###########
print("Loading testing data...")
orig_data = pd.read_csv(f"../../anon_data/birth_mondrian/k1_minmaxed.csv").drop("Unnamed: 0", axis=1)
print(orig_data)
############################################


###### Starting to train estimators ###########

accs = {}
results = {}
for k in range(1,101):
    anon_data = pd.read_csv(f"../../anon_data/birth_mondrian/k{k}_minmaxed.csv").drop("Unnamed: 0", axis=1)
    r = {}
    print(f"####### K:{k} ##########")
    r["knn_pca_auroc"], r["knn_pca_acc"] = train_test(orig_data, anon_data, "knn", use_pca=True)
    print("knn, pca:   ", r["knn_pca_auroc"], r["knn_pca_acc"])

    r["rf_pca_auroc"], r["rf_pca_acc"] = train_test(orig_data, anon_data, "rf", use_pca=True)
    print("rf, pca:   ", r["rf_pca_auroc"], r["rf_pca_acc"])

    r["lr_auroc"], r["lr_acc"] = train_test(orig_data, anon_data, "lr")
    print("lr, no pca:   ", r["lr_auroc"], r["lr_acc"])
    results[k] = r
print(results)
df = pd.DataFrame.from_dict(results, orient='index')
df.to_csv(f"accuracies.csv")
