import numpy as np

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
        tot += np.prod(i)

    dataset_amb = tot/len(anon_data)
    max_amb = np.prod(max_choices)
    rel_amb = dataset_amb / max_amb
    return rel_amb
