import numpy as np
import pandas as pd

import re
import time

def load_csv(k):
    data = pd.read_csv(f'../../anon_data/birth_datafly/k{k}_minmaxed.csv')
    data.rename(columns={"Unnamed: 0":"index"}, inplace=True)
    data.set_index("index")
    return data


def cell_H(data, col_name, min_val, max_val):
    probs = conditional_probs(data, col_name, min_val, max_val)
    H = probs.map(lambda p: p*np.log2(p))
    return H.sum()*-1

def conditional_probs(data, col_name, min_val, max_val):

    col = data[f"{col_name}_min"]
    col = col[min_val <= col]
    col = col[max_val >= col]

    counts = col.value_counts()
    cond_probs = counts/counts.sum()

    return cond_probs


def get_cond_entr(k):
    data = load_csv(1)
    anon_data = load_csv(k)

    cols = anon_data.columns[1:-1][0::2]
    cols = cols.map(lambda x: "_".join(x.split("_")[:-1]))

    tot_H = 0

    for c in cols:
        col_H = 0
        cache = {}

        for i in range(len(anon_data)):
            lo = anon_data[f"{c}_min"][i]
            hi = anon_data[f"{c}_max"][i]

            if (lo, hi) in cache:
                H = cache[(lo,hi)]
            else:
                H = cell_H(data, c, lo, hi)
                cache[(lo,hi)] = H

            col_H += H

        tot_H += col_H
    return tot_H


entr_metric = pd.DataFrame()

print("k_val,cond_entropy")
for k in [2]:
    entr = get_cond_entr(k)
    print(f"{k},{entr}")
    entr_metric = entr_metric.append({'k_val':k, 'cond_entropy':entr}, ignore_index=True)

entr_metric.to_csv("cond_entropy.csv")
print("✓ CSV saved")
