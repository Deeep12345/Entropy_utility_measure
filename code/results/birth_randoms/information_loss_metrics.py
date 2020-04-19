import numpy as np
import pandas as pd

def discern_metric(anon_data):
    grouped = anon_data.groupby(list(anon_data.columns[:-1]))
    sizes = np.array(grouped.size())
    squares = sizes*sizes
    max_discern = len(anon_data) ** 2
    return sum(squares) / max_discern



def IL_metric(anon_data):
    eqs = anon_data[anon_data.columns[:-1]].drop_duplicates()
    orig_cols = set([c[:-1] for c in anon_data.columns[:-1]])
    sizes = anon_data.groupby(list(anon_data.columns[:-1]), as_index=False).size()

    il = 0
    for i, eq in eqs.iterrows():
        s = sizes[tuple(eq)]
        distance = 0

        for c in orig_cols:
            rel_cols = list(filter(lambda col: c in col, eqs.columns))
            dom = len(rel_cols) - 1
            rel_cols = list(filter(lambda col: eq[col] == 1, rel_cols))
            vals = [int(x[-1]) for x in rel_cols]
            width = (max(vals) - min(vals)) / dom
            distance += width

        il += distance * s

    il = il / (len(anon_data) * len(anon_data.columns[:-1]))
    return il
