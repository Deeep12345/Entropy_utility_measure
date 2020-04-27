import pandas as pd
import numpy as np

import more_itertools as mit
import itertools
import random

from graphviz import Digraph
import lxml
from bs4 import BeautifulSoup as bs

def flatten(ls):
    flattened = []
    for l in ls:
        flattened.extend(l)
    return flattened

def categorize(data, no_bins):
    for c in data.columns[:-1]:
        print(c)
        col = data[c]
        mi, ma = col.min(), col.max()
        cuts = np.linspace(mi, ma+1, num=no_bins+1)
        col_new = pd.cut(col, cuts,
                        labels=[i for i in range(0,20)],
                        right=False)
        data[c] = col_new
    return data


def cut(l, i):
    return [l[:i], l[i:]]


def random_partition(l):
    bins = len(l)
    p = [[] for i in range(5)]
    for i in l:
        p[np.random.randint(0,len(p))].append(i)
    p = list(filter(lambda x: x, p))
    return p


def ordered_partition(l):
    len_splits = []
    p = random_partition(l)
    s_sizes = [len(part) for part in p]
    o_p = [ l[x-y:x] for x, y in zip(itertools.accumulate(s_sizes), s_sizes)]
    return o_p


def make_tree(l):
    tree = {}
    l = ordered_partition(l)
    lens = [len(item) for item in l]
    cur_val = (min(l[0]), max(l[len(l)-1]))
    tree["value"] = cur_val
    if min(lens) > 1:
        cs = []
        for subset in l:
            cs.append(make_tree(subset))
            tree["children"] = cs
    else:
        minmaxs = [{"value":(min(item), max(item))} for item in l]
        tree["children"] = minmaxs
    return tree


def min_depth(t):
    mi, ma = t["value"]
    if mi == ma or "children" not in t:
        return 1
    else:
        return min([1 + min_depth(ch) for ch in t["children"]])


def flatten_recur(t, d, max_d):
    if d == max_d:
        if "children" in t:
            del(t["children"])
    else:
        for child in t["children"]:
            flatten_recur(child, d+1, max_d)


def flatten_tree(t):
    min_d = min_depth(t)
    flatten_recur(t, 1, min_d)


def make_vgh(no_bins):
    l = list(range(no_bins))
    t = make_tree(l)
    flatten_tree(t)
    return t


def tree_to_xml(t, head=False):
    mi, ma = t["value"]
    res = ""
    title = "vgh" if head else "node"
    if 'children' not in t:
        res += f"<{title} value='[{mi}:{ma}]'/>"
    else:
        res += f"<{title} value='[{mi}:{ma}]'>"
        if len(t["children"]) == 1:
            mi, ma =t["children"][0]["value"]
            res += f"""
            <{title} value='[{mi}:{mi +(ma-mi)//2}]'/>
            <{title} value='[{mi + 1 +(ma-mi)//2}:{ma}]'/>"""
        else:
            for c in t["children"]:
                res += tree_to_xml(c)
        res += (f"</{title}>")
    return res


def make_mondrian_config(name, k, no_cols, no_bins):
    mappings = {}
    full_xml = f"""
    <config method='Mondrian' k='{k}'>
    <input filename='../../datasets/ring/ring_cat.csv' separator=','/>
    <output filename='../anon_data/ring_randoms/mondrian{name}.csv' format ='genVals'/>
    <id></id>
    <qid>"""

    for i in range(0,no_cols):
        full_xml += f"""
        <att index='{i}' name='{i}th'>
        <vgh value='[0:{no_bins-1}]'/>"""

        ord = list(range(0,no_bins))
        random.shuffle(ord)
        full_xml += "<map>"
        for j, m in enumerate(ord):
            full_xml += f"<entry cat='{j}' int='{m}'/>"
        full_xml += "</map></att>"

    full_xml += f"""
    </qid>
    <sens><att index='{no_cols}' name='class'/></sens>
    </config>"""
    xml_tree = bs(full_xml, 'xml')
    f = open(f"ring_randoms/mondrian{name}.xml", "w+")
    f.write(xml_tree.prettify())
    f.close()

    return xml_tree


def make_datafly_config(name, k, no_cols, no_bins, shuffled=False):
    is_shuffled = "_shuffled" if shuffled else ""
    full_xml = f"""
    <config method='Datafly' k='{k}'>
    <input filename='../../datasets/ring/ring_cat.csv' separator=','/>
    <output filename='../anon_data/ring_randoms/datafly{name}{is_shuffled}.csv' format ='genVals'/>
    <id></id>
    <qid>"""

    for i in range(no_cols):
        full_xml += (f"<att index='{i}' name='{i}th'>")
        #VGH
        tree = make_vgh(no_bins)
        tree = tree_to_xml(tree, head=True)
        full_xml += tree

        #Mapping
        if shuffled:
            ord = list(range(no_bins))
            random.shuffle(ord)
            full_xml += "<map>"
            for j, m in enumerate(ord):
                full_xml += f"<entry cat='{j}' int='{m}'/>"
            full_xml += "</map>"
        full_xml += "</att>"

    full_xml += f"""
    </qid>
    <sens>
    <att index='20' name='class'/>
    </sens>
    </config>"""

    xml_tree = bs(full_xml, 'xml')
    f = open(f"ring_randoms/datafly{name}{is_shuffled}.xml", "w+")
    f.write(xml_tree.prettify())
    f.close()

    return full_xml


no_bins = 20
no_cols = 20
ks = np.random.poisson(1, 200)
ks = ks.astype(int) + 2
print(pd.Series(ks).value_counts(), "\n")

for i, k in enumerate(ks):
    make_datafly_config(i+1, k, no_cols, no_bins, shuffled=False)
    make_datafly_config(i+1, k, no_cols, no_bins, shuffled=True)
