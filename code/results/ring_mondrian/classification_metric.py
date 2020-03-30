import numpy as np
import pandas as pd

import re
import time

def load_ring_csv(k):
    data = pd.read_csv(f'k{k}_minmaxed.csv')
    data.rename(columns={"Unnamed: 0":"index"}, inplace=True)
    data.set_index("index")
    return data


def get_class_metric(k):
    anon_data = load_ring_csv(k)

    grouped = anon_data.groupby(list(anon_data.columns[1:-1]))
    tot = 0

    for _, classes in grouped:
        cs = classes["class"].value_counts()

        max_class = cs.idxmax()
        pen_eq = cs.sum() - cs[max_class]

        tot+= pen_eq

    return tot/len(anon_data)


c_metric = pd.DataFrame()

print("k_val,cm")
for k in list(range(1,51)) + list(range(100, 7400,250)) + [7400]:
    cm = get_class_metric(k)
    print(f"{k},{cm}")
    c_metric = c_metric.append({'k_val':k, 'cm':cm}, ignore_index=True)

c_metric.to_csv("../../results/ring_mondrian/classif_metric.csv")
print("✓ CSV saved")
