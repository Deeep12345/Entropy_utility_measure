import pandas as pd
import numpy as np
import xml.etree.ElementTree as et
import xmltodict as xdict
import re

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

def get_datafly_trees(root):
    trees = {}
    for child in root[3]:
        name = child.attrib["name"]
        v = child[0]
        v = et.tostring(v)
        t = xdict.parse(v)
        t = dict(t["vgh"])
        trees[name] = t
    return trees

def get_datafly_depths(trees):
    depths = {}
    for name, t in trees.items():
        d = {}
        if "node" in t:
            d[t["@value"]] = 2
            for node in t["node"]:
                d[node["@value"]] = 1
        else:
            d[t["@value"]] = 1
        depths[name] = d
    return depths

def get_mondrian_depths(bounded_data):
    depths = {}
    for col in bounded_data.columns[:-1]:
        vals = set(bounded_data[col])
        ds = {}
        for v in vals:
            lo, hi = parse_range(v)
            ds[v] = hi - lo
        depths[col] = ds
    return depths


def precision_metric(bounded_data, algo, no, root):
    if algo == "datafly":
        trees = get_datafly_trees(root)
        depths = get_datafly_depths(trees)
        max_depths = {attr:max(depths[attr].values()) for attr in depths}
    elif algo == "mondrian":
        depths =get_mondrian_depths(bounded_data)
        max_depths = {attr:(1 if attr in ["wife_works", "wife_rel", "media_exp"]
                            else 3) for attr in depths}

    prec = 0

    for c in bounded_data.columns[:-1]:
        counts = bounded_data[c].value_counts()
        dist = 0
        for count in counts.index:
            if '.' not in count:
                dist += counts[count] * depths[c][count] / max_depths[c]
        prec += dist

    prec /= (len(bounded_data) * len(bounded_data.columns[:-1]))
    return 1 - prec
