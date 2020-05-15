import pandas as pd
import numpy as np
import scipy.stats as st


def oh_to_labels(df, qi):
    rel_cols = list(filter(lambda c: c[:len(qi)] == qi, df.columns))
    col = df[rel_cols].apply(lambda x: ' '.join([str(i) for i in x]), axis=1)
    return col

def cramers_v(df, var1, var2):
    col1 = oh_to_labels(df, var1)
    col2 = oh_to_labels(df, var2)

    crosstab = np.array(pd.crosstab(col1,col2, rownames=None, colnames=None))
    stat = st.chi2_contingency(crosstab)[0]
    obs = np.sum(crosstab) # Number of observations
    mini = min(crosstab.shape)-1
    res = stat/(obs*mini)
    return res

def bivariate_corr_metric(orig_data, anon_data, QIs):
    corr_distances = []

    for i in QIs:
        for j in QIs:
            cram_anon = cramers_v(anon_data, i, j)
            if not np.isnan(cram_anon):
                cram_orig = cramers_v(orig_data, i, j)
                diff = abs(cram_orig-cram_anon)
                corr_distances.append(diff)

    return np.median(corr_distances)
