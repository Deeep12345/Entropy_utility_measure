import numpy as np
import pandas as pd

import re
import time

def load_ring_csv(k):
    data = pd.read_csv(f'k{k}_minmaxed.csv')
    data.rename(columns={"Unnamed: 0":"index"}, inplace=True)
    data.set_index("index")
    return data


def distance(u,v):
    d = u == v
    counts = d.value_counts()
    return counts[False] if False in counts else 0


def diam_m(k):
    data = load_ring_csv(k)
    cols = list(data.columns[1:-1])
    data = data[cols]
    print(f"\tpre-drop size of data: {len(data)}")
    data = data.drop_duplicates()
    print(f"\tafter drop size of data: {len(data)}")

    diam = 0

    for x in range(len(data)):
        for y in range(x+1, len(data)):
            d = distance(data.iloc[x], data.iloc[y])

            if d == len(cols):
                return d

            if d > diam:
                diam = d
        if x % 10 == 0:
            print(f"\t({x}) Current best diam: {diam}")
    return diam


diam_metric = pd.DataFrame()
print("k_val,diam_metric")

for k in list(range(1,51)) + list(range(100, 7400,250)) + [7400]:
    print(f"Finding diameter for k: {k}...")
    dm = diam_m(k)
    print(f"\t{k},{dm}")
    diam_metric = diam_metric.append({'k_val':k, 'diam_metric':dm}, ignore_index=True)

diam_metric.to_csv("../../results/ring_mondrian/diameter_metric.csv")
print("✓ CSV saved")
