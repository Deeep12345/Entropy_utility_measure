import numpy as np
from operator import mul

def ambiguity_metric(anon_data, QIs):
    max_choices = []
    choices = []
    for qi in QIs:
        rel_cols = list(filter(lambda c: c[:len(qi)] == qi, anon_data.columns))
        no_choices = len(rel_cols)
        max_choices.append(no_choices)

        col = anon_data[rel_cols]
        col_choices = [sum(r) for ind, r in col.iterrows()]
        choices.append(col_choices)

    row_choices = (zip(*choices))
    tot = 0
    for i in row_choices:
        m = 1
        for x in i:
            m *= x
        tot += m

    dataset_amb = tot/len(anon_data)
    rel_amb = dataset_amb
    for i in max_choices:
        rel_amb /= i
    return rel_amb
