import pandas as pd
import numpy as np

from diameter_metric import diam
from classification_metric import class_metric
from RFCs_train import train_test
from entropy import cond_entr

def load_csv(algo, no, orig=False):
    if orig:
        data = pd.read_csv(f'../../anon_data/birth_random_groups/original_oh.csv')
    else:
        data = pd.read_csv(f'../../anon_data/birth_random_groups/{algo}{no}_oh.csv')
    data.drop("Unnamed: 0", axis=1, inplace=True)
    return data


orig_data = load_csv("", 0, orig=True)
full_anon = orig_data.copy()
full_anon.replace(0, 1, inplace=True)

max_H = cond_entr(orig_data, full_anon)

results = {}
for no in range(1,2):
    for algo in ["datafly", "mondrian"]:
        r = {}
        anon_data = load_csv(algo, no)

        r["dm"] = diam(anon_data)
        r["cm"] = class_metric(anon_data)
        r["entropy"] = cond_entr(orig_data, anon_data)/max_H
        r["acc"] = train_test(orig_data, anon_data)

        results[(algo, no)] = r

print(results)
df = pd.DataFrame.from_dict(results, orient='index')
df.to_csv("metrics.csv")
