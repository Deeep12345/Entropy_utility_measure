import pandas as pd
import numpy as np

import sys
import yaml
from tqdm import tqdm

from metrics.ML_train import train_test

f = open(f"{sys.argv[1]}/config.yaml")
config = yaml.load(f, Loader=yaml.FullLoader)
print(f"##### Config file: {sys.argv[1]} ######")
print(yaml.dump(config))
sys.path.append(config["analysis_name"])


def load_csv(algo, no, original_oh=True):
    if original_oh:
        return pd.read_csv(f"../anon_data/{config['analysis_name']}/original_oh.csv")

    if algo == "datafly_shuffled":
        f = f"../anon_data/{config['analysis_name']}/datafly{no}_oh_shuffled.csv"
    else:
        f = f"../anon_data/{config['analysis_name']}/{algo}{no}_oh.csv"
    return pd.read_csv(f)

orig_data = load_csv(0,0)

results = {}
for no in range(1, config["no_instances"]+1):

    for algo in config["algos_used"]:
        print(no, algo)
        r = {}
        anon_data = load_csv(algo, no, original_oh=False)

        # r["mlp_auroc"], r["mlp_acc"] = train_test(orig_data, anon_data, "mlp")
        # print("mlp, no pca:   ", r["mlp_auroc"], r["mlp_acc"])
        # r["mlp_pca_auroc"], r["mlp_pca_acc"] = train_test(orig_data, anon_data, "mlp", use_pca=True)
        # print("mlp, pca:   ", r["mlp_pca_auroc"], r["mlp_pca_acc"])

        r["knn_auroc"], r["knn_acc"] = train_test(orig_data, anon_data, "knn")
        print("knn, no pca:   ", r["knn_auroc"], r["knn_acc"])
        r["knn_pca_auroc"], r["knn_pca_acc"] = train_test(orig_data, anon_data, "knn", use_pca=True)
        print("knn, pca:   ", r["knn_pca_auroc"], r["knn_pca_acc"])

        r["rf_auroc"], r["rf_acc"] = train_test(orig_data, anon_data, "rf")
        print("rf, no pca:   ", r["rf_auroc"], r["rf_acc"])
        r["rf_pca_auroc"], r["rf_pca_acc"] = train_test(orig_data, anon_data, "rf", use_pca=True)
        print("rf, pca:   ", r["rf_pca_auroc"], r["rf_pca_acc"])

        r["lr_auroc"], r["lr_acc"] = train_test(orig_data, anon_data, "lr")
        print("lr, no pca:   ", r["lr_auroc"], r["lr_acc"])
        r["lr_pca_auroc"], r["lr_pca_acc"] = train_test(orig_data, anon_data, "lr", use_pca=True)
        print("lr, pca:   ", r["lr_auroc"], r["lr_acc"])
        ######
        results[(algo, no)] = r

    if no % 20 == 0:
        df = pd.DataFrame.from_dict(results, orient='index')
        df.to_csv(f"{config['analysis_name']}/accuracy_results.csv",
                    index_label=["algo","no"])

print(results)
df = pd.DataFrame.from_dict(results, orient='index')
df.to_csv(f"{config['analysis_name']}/accuracy_results.csv",
            index_label=["algo","no"])
