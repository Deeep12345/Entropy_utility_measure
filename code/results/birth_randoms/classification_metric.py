import numpy as np
import pandas as pd

import re
import time

def class_metric(anon_data):
    grouped = anon_data.groupby(list(anon_data.columns[:-1]))
    tot = 0

    for _, classes in grouped:
        cs = classes["class"].value_counts()

        max_class = cs.idxmax()
        pen_eq = cs.sum() - cs[max_class]

        tot+= pen_eq

    return tot/len(anon_data)
