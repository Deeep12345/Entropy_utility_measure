import numpy as np

def distance_squared_error(anon_data, orig_data, QIs):

    distances = []
    max_rec_dist = 0
    for qi in QIs:
        rel_cols = list(filter(lambda c: c[:len(qi)] == qi, anon_data.columns))
        max_rec_dist += (len(rel_cols)-1)**2

        anon_col = anon_data[rel_cols]
        orig_col = orig_data[rel_cols]

        values = np.array([list(r).index(1) for _, r in orig_col.iterrows()])
        means = [np.mean(np.where(r)) for _, r in anon_col.iterrows()]
        difs = (values - means)**2
        distances.append(difs)

    max_dist = len(anon_data) * np.sqrt(max_rec_dist)

    rec_distances = (zip(*distances))
    tot_dist = np.sum([np.sqrt(np.sum(rd)) for rd in rec_distances])
    return tot_dist/max_dist
