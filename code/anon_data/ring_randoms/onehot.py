import numpy as np
import pandas as pd

from tqdm import tqdm

import re
import xml.etree.ElementTree as et

def parse_range(x):
    if ":" not in x:
        x = x.split(".")
        val = int(x[0][1:])
        return val, val

    vals = re.split(':', x[1:-1])
    vals[0] = vals[0].split(".")[0]
    vals[1] = vals[1].split(".")[0]
    low = int(vals[0]) if x[0] == '[' else int(vals[0]) + 1
    high = int(vals[1]) if x[-1] == ']' else int(vals[1]) - 1
    return low, high

def get_mapping(no, algo, root):
    mapping = {}
    for child in root[3]:
        attr = child.attrib["name"]
        m = {}
        for v in child[1]:
            m[v.attrib["int"]] = v.attrib["cat"]
        mapping[attr] = m

    return mapping



def onehot(no, algo, cols, shuffled=True):
    #print(f"OneHotting for no:{no}, algo:{algo}, shuffled:{shuffled}")
    shuff = "_shuffled" if algo == "datafly" and shuffled else ""
    data = pd.read_csv(f"{algo}{no}{shuff}.csv",names=cols,index_col=False)
    xml = et.parse(f"../../toolbox_linux64/configs/ring_randoms/{algo}{no}{shuff}.xml")

    root = xml.getroot()
    if shuffled:
        mapping = get_mapping(no, algo, root)
    else:
        mapping = {c:{str(i):str(i) for i in range(20)} for c in cols}
    new_df = pd.DataFrame()

    for c in data.columns[:-1]:
        no_choices = 20
        col = data[c]
        oh_l = []
        for x in col:
            mi, ma = parse_range(x)
            oh = [0] * no_choices
            for i in range(mi,ma+1):
                val = int(mapping[c][str(i)])
                oh[val] = 1
            oh_l.append(oh)

        cols = [c+str(i) for i in range(0, no_choices)]

        oh_df = pd.DataFrame.from_records(oh_l, columns=cols)
        new_df[cols] = oh_df

    new_df["class"] = data["class"]
    new_df.to_csv(f"{algo}{no}_oh{shuff}.csv", index=False)



for i in tqdm(range(1,201)):
    cols = [f"{i}th" for i in range(20)]
    cols.append("class")
    onehot(i, "datafly", cols, shuffled=False)
    onehot(i, "datafly", cols, shuffled=True)
    onehot(i, "mondrian", cols, shuffled=True)
