import numpy as np
import pandas as pd

import re


def parse_range(x):
    if isinstance(x, float):
        val = int(x)
        return val, val

    vals = re.split(':', x[1:-1])
    vals = [bound[:-2] for bound in vals]
    low = int(vals[0]) if x[0] == '[' else int(vals[0]) + 1
    high = int(vals[1]) if x[-1] == ']' else int(vals[1]) - 1
    return low, high


def min_max(no):
    print(f"splitting minmax for k={no}")
    data = pd.read_csv(f"k{no}.csv",
                       names=[f"A{i}" for i in range(20)] + ["class"],
                       index_col=False)

    print(f"Columns: {data.columns}")
    for col in data.columns[:-1]:
        mins, maxs = [], []
        for bounds in data[col]:
            lo,hi = parse_range(bounds)
            mins.append(lo)
            maxs.append(hi)

        data[f"{col}_min"] = mins
        data[f"{col}_max"] = maxs

        data = data.drop(col, axis=1)

    cols = list(data.columns)
    data = data[cols[1:] + [cols[0]]]
    print(f"Columns after {data.columns}")
    data.to_csv(f"k{no}_minmaxed.csv")
    print("############################")


for i in range(100,7400,250):
    min_max(i)

