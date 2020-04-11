import numpy as np
import pandas as pd

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

def get_mapping(no, algo, root, bin_attrs):
    mapping = {}
    if algo == "datafly":
        for child in root[5]:
            attr = child.attrib["name"]
            m = {}
            for v in child:
                m[v.attrib["used"]] = v.attrib["mapbackto"]
            mapping[attr] = m
    else:
        for child in root[3]:
            attr = child.attrib["name"]
            m = {}
            if attr not in bin_attrs:
                    for v in child[1]:
                        m[v.attrib["int"]] = v.attrib["cat"]
                    mapping[attr] = m

    return mapping



def onehot(no, algo, cols):
    print(f"OneHotting for no:{no}, algo={algo}")
    data = pd.read_csv(f"{algo}{no}.csv",
                       names=cols,
                       index_col=False)

    bin_attrs = ["wife_rel", "wife_works", "media_exp"]
    #print(f"Columns: {data.columns}")

    xml = et.parse(f"../../toolbox_linux64/configs/random_configs/{algo}{no}.xml")
    root = xml.getroot()
    mapping = get_mapping(no, algo, root, bin_attrs)
    new_df = pd.DataFrame()

    for c in data.columns[:-1]:
        no_choices = 2 if c in bin_attrs else 4
        col = data[c]
        oh_l = []
        for x in col:
            mi, ma = parse_range(x)
            oh = [0] * no_choices
            for i in range(mi,ma+1):
                val = int(mapping[c][str(i)]) if no_choices == 4 else i+1
                oh[val-1] = 1
            oh_l.append(oh)
        if no_choices == 2:
            cols = [c+str(i) for i in range(0, no_choices)]
        else:
            cols = [c+str(i) for i in range(1, no_choices+1)]

        oh_df = pd.DataFrame.from_records(oh_l, columns=cols)
        new_df[cols] = oh_df

    new_df["class"] = data["class"]


    #print(f"Columns after {new_df.columns}")
    new_df.to_csv(f"{algo}{no}_oh.csv")



for i in range(1,201):
    cols = ["age","wife_ed","husb_ed","no_kids","wife_rel","wife_works",
    "husb_occupation","SOL_index","media_exp",	"class"]
    onehot(i, "datafly", cols)
    onehot(i, "mondrian", cols)
    print(f"{i}'s are anonymised!")
    print("############################")
