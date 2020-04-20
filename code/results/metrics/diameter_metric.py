import numpy as np
import pandas as pd

import re
import time

from sklearn.metrics import pairwise_distances

def diam_metric(anon_data):
    cols = list(anon_data.columns[:-1])
    rel_cols = anon_data[cols]

    dists = pairwise_distances(rel_cols, metric="hamming", n_jobs=-1)

    max_dist = 0

    for x in range(len(dists[0])):
        r_max = max(dists[x])
        if r_max > max_dist:
            max_dist = r_max
        if max_dist == 1:
            break

    return max_dist
