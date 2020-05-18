import numpy as np
import pandas as pd

def get_k(conf):
    print(conf.name())
    return

def avg_eq_size_metric(anon_data, conf):
    grouped = anon_data.groupby(list(anon_data.columns[:-1]))
    sizes = np.array(grouped.size())
    k = int(conf.attrib["k"])
    avg_size = (len(anon_data) / len(sizes)) / k
    max_size = len(anon_data)/k
    return avg_size/max_size
