import pandas as pd
import numpy as np

_SQRT2 = np.sqrt(2)

def get_distribution(df, qi):
    rel_cols = list(filter(lambda c: c[:len(qi)] == qi, df.columns))
    g = df.groupby(rel_cols).indices
    distr = np.zeros(len(rel_cols))
    for i in g:
        count = len(g[i])
        for index, val in enumerate(i):
            if val:
                distr[index] += count

    distr = distr/sum(distr)
    return distr


def hellinger(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2


def hellinger_metric(orig_data, anon_data, QIs):
    hs = []
    for c in QIs:
        p = get_distribution(orig_data, c)
        q = get_distribution(anon_data, c)
        hs.append(hellinger(p, q))
    return np.median(hs)
