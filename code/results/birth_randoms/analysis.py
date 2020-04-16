import pandas as pd
import numpy as np

from diameter_metric import diam
from classification_metric import class_metric
from RFCs_train import train_test
from entropy import cond_entr
from hierarchy_metrics import precision

def load_csv(algo, no, original_oh=False, bounded=False):
    if original_oh:
        data = pd.read_csv(f'../../anon_data/birth_random_groups/original_oh.csv')
    elif bounded:
        cols = ["age","wife_ed","husb_ed","no_kids","wife_rel","wife_works",
            "husb_occupation","SOL_index","media_exp","class"]
        data = pd.read_csv(f"../../anon_data/birth_random_groups/{algo}{no}.csv",
            names=cols)
    else:
        data = pd.read_csv(f'../../anon_data/birth_random_groups/{algo}{no}_oh.csv')
    if not bounded:
        data.drop("Unnamed: 0", axis=1, inplace=True)
    return data


orig_data = load_csv("", 0, original_oh=True)
full_anon = orig_data.copy()
full_anon.replace(0, 1, inplace=True)

max_H = cond_entr(orig_data, full_anon)

results = {}
for no in range(1,201):
    for algo in ["mondrian", "datafly"]:
        r = {}
        anon_data = load_csv(algo, no)
        bounded_data = load_csv(algo, no, bounded=True)

        r["precision"] = precision(bounded_data, algo, no)
        r["dm"] = diam(anon_data)
        r["cm"] = class_metric(anon_data)
        r["entropy"] = cond_entr(orig_data, anon_data)/max_H
        r["acc"] = train_test(orig_data, anon_data)


        print(no, algo, r)
        results[(algo, no)] = r

print(results)
df = pd.DataFrame.from_dict(results, orient='index')
df.to_csv("metrics.csv")
