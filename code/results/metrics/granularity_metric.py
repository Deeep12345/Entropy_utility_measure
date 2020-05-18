import numpy as np

def granularity_metric(anon_data, QIs):

    gran = 0
    max_gran = 0
    for qi in QIs:
        rel_cols = list(filter(lambda c: c[:len(qi)] == qi, anon_data.columns))
        col = anon_data[rel_cols]
        col_leaves = np.int64([sum(r)-1 for ind, r in col.iterrows()])
        col_gran = np.sum(col_leaves/len(rel_cols))

        max_gran += len(anon_data) * (len(rel_cols)-1)
        gran += col_gran

    rel_gran = gran / max_gran
    return rel_gran
