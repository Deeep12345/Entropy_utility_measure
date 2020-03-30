import numpy as np
import pandas as pd

import re
import time

from sklearn.metrics import pairwise_distances

def load_ring_csv(k):
    data = pd.read_csv(f'../../anon_data/ring_mondrian/k{k}_minmaxed.csv')
    data.rename(columns={"Unnamed: 0":"index"}, inplace=True)
    data.set_index("index")
    return data


def diam_m(k):
    anon_data = load_ring_csv(k)
    cols = list(anon_data.columns[1:-1])
    rel_cols = anon_data[cols]
    rel_cols = rel_cols.drop_duplicates()

    dists = pairwise_distances(rel_cols, metric="hamming", n_jobs=-1)

    max_dist = 0

    for x in range(len(dists[0])):
        r_max = max(dists[x])
        if r_max > max_dist:
            max_dist = r_max
        if max_dist == 1:
            break

    return max_dist


diam_metric = pd.DataFrame()
print("k_val,diam_metric")

for k in list(range(1,51,2)) + list(range(100,3900,250)) + [7400]:
    dm = diam_m(k)
    print(f"{k},{dm}")
    diam_metric = diam_metric.append({'k_val':k, 'dm':dm}, ignore_index=True)

diam_metric.to_csv("diameter_metric.csv")
print("✓ CSV saved")
