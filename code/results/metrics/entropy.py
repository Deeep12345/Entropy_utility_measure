import numpy as np
import pandas as pd

import re
import time

def cell_H(orig_data, choices, tup):
    probs = conditional_probs(orig_data, choices, tup)
    H = list(map(lambda p: p*np.log2(p) if p != 0 else 0, probs))
    return sum(H)*-1

def conditional_probs(orig_data, choices, tup):

    cols = orig_data[choices]
    n = {c:i for i,c in enumerate(cols.columns)}
    cols = cols.rename(n, axis=1)
    counts = []
    for i,val in enumerate(tup):
        if val:
            count = len(cols[cols[i] == 1])
            counts.append(count)

    s = sum(counts)
    cond_probs = [c/s for c in counts]

    return cond_probs


def cond_entr_metric(orig_data, anon_data, def_cols):
    cols = anon_data.columns[:-1]
    tot_H = 0

    for c in def_cols:
        col_H = 0

        choices = list(filter(lambda x: x[:len(c)] == c, cols))
        tups = anon_data.groupby(choices).size()

        for r in tups.index:
            tup = tuple(r)
            no_times = tups[r]
            H = cell_H(orig_data, choices, tup)
            col_H += H*no_times
        tot_H += col_H

    return tot_H
