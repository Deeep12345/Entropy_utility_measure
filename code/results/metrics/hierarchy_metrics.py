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

def get_datafly_depths(root):
    trees = {}
    for child in root[3]:
        name = child.attrib["name"]
        v = child[0]
        v = et.tostring(v)
        t = xdict.parse(v)
        t = dict(t["vgh"])
        t = dat_tree_recur(t, 1)
        max_d = max(t.values())
        for v in t:
            t[v] = abs(t[v] - max_d -1)
        trees[name] = t

    return trees

def dat_tree_recur(tree, depth):
    branches = {tree['@value']: depth}
    if 'node' not in tree:
        return branches
    for branch in tree['node']:
        res = dat_tree_recur(branch, depth+1)
        branches.update(res)
    return branches


def get_mapping(no, algo, root):
    mapping = {}
    for child in root[3]:
        attr = child.attrib["name"]
        m = {}
        for v in child[1]:
            m[v.attrib["int"]] = v.attrib["cat"]
        mapping[attr] = m

    return mapping


def one_hot(trees, mappings):
    print(trees["16th"])
    print(mappings["16th"])
    oh_trees = {}
    for attr in trees:
        t = {}
        m = mappings[attr]
        for val in trees[attr]:
            # lo, hi = parse_range(val)
            # oh = [1 if i >= lo and i <= hi else 0
            #         for i in range(len(m))]
            # mapped_oh = [0] * len(m)
            # for used in m:
            #     mapped_oh[int(m[used])] = oh[int(used)]
            # t[tuple(mapped_oh)] = trees[attr][val]


            mi, ma = parse_range(val)
            oh = [0] * len(m)
            for i in range(mi,ma+1):
                v = int(m[str(i)])
                oh[v] = 1
            t[tuple(oh)] = trees[attr][val]

        len_tup = len(list(t.keys())[0])
        for i in range(len_tup):
            oh_0 = [0] * len_tup
            oh_0[i] = 1
            t[tuple(oh_0)] = 0
        oh_trees[attr] = t

    return oh_trees



def get_mondrian_depths(anon_data, QIs):
    depths = {}
    for col in QIs:
        rel_cols = list(filter(lambda c: c[:len(col)] == col, anon_data.columns))
        rel_cols = list(anon_data[rel_cols].itertuples(index=False, name=None))
        vals = set(rel_cols)
        ds = {}
        for v in vals:
            hots = list(filter(lambda x: x == 1, v))
            ds[v] = len(hots) - 1
        depths[col] = ds
    return depths


def precision_metric(anon_data, algo, no, root, QIs):
    if "datafly" in algo:
        bound_depths = get_datafly_depths(root)
        if algo == "datafly":
            mappings = {c:{str(i):str(i) for i in range(20)} for c in QIs}
        else:
            mappings = get_mapping(no, algo, root)
        depths = one_hot(bound_depths, mappings)
        max_depths = {attr:max(depths[attr].values()) for attr in depths}
    elif algo == "mondrian":
        depths =get_mondrian_depths(anon_data, QIs)
        max_depths = {attr:len(list(filter(lambda c: attr in c, anon_data.columns)))-1 for attr in QIs}

    prec = 0
    for c in QIs:
        rel_cols = list(filter(lambda col: c == col[:len(c)], anon_data.columns))
        counts = anon_data.groupby(rel_cols).size()
        for count in counts.index:
            prec += counts[count] * depths[c][count] / max_depths[c]

    prec /= (len(anon_data) * len(QIs))
    return 1 - prec
