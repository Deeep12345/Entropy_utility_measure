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


def make_mondrian_config(name, k, vals):
    mappings = {}
    full_xml = f"""
    <config method='Mondrian' k='{k}'>
    <input filename='../../datasets/adult/adult_cat_mapped.csv' separator=','/>
    <output filename='../anon_data/adult_randoms/mondrian{name}.csv' format ='genVals'/>
    <id/>
    <qid>"""

    for i, attr_vals in enumerate(vals):
        full_xml += f"<att index='{i}' name='{i}th'>"
        ord = list(range(len(attr_vals)))
        random.shuffle(ord)
        full_xml += "<map>"
        for j, v in enumerate(attr_vals):
            full_xml += f"<entry cat='{v}' int='{ord[j]}'/>"
        full_xml += f"""
        </map>
        <vgh value='[0:{len(attr_vals)-1}]'/>
        </att>"""

    full_xml += f"""
    </qid>
    <sens><att index='12' name='class'/></sens>
    </config>"""
    xml_tree = bs(full_xml, 'xml')
    f = open(f"adult_randoms/mondrian{name}.xml", "w+")
    f.write(xml_tree.prettify())
    f.close()

    return xml_tree


def make_datafly_config(name, k, vals, cols, non_ordered_cols, shuffled=False):
    is_shuffled = "_shuffled" if shuffled else ""
    full_xml = f"""
    <config method='Datafly' k='{k}'>
    <input filename='../../datasets/adult/adult_cat_mapped.csv' separator=','/>
    <output filename='../anon_data/adult_randoms/datafly{name}{is_shuffled}.csv' format ='genVals'/>
    <id/>
    <qid>"""

    for i, (at, attr_vals) in enumerate(zip(cols, vals)):
        full_xml += (f"<att index='{i}' name='{i}th'>")
        #VGH
        tree = make_vgh(len(attr_vals))
        tree = tree_to_xml(tree, head=True)

        #Mapping
        ord = list(range(len(attr_vals)))
        if shuffled or at in non_ordered_cols:
            random.shuffle(ord)

        full_xml += "<map>"
        for j, v in enumerate(attr_vals):
            full_xml += f"<entry cat='{v}' int='{ord[j]}'/>"
        full_xml += "</map>"

        full_xml += tree

        full_xml += "</att>"

    full_xml += f"""
    </qid>
    <sens>
    <att index='12' name='class'/>
    </sens>
    </config>"""

    xml_tree = bs(full_xml, 'xml')
    f = open(f"adult_randoms/datafly{name}{is_shuffled}.xml", "w+")
    f.write(xml_tree.prettify())
    f.close()

    return full_xml


ks = np.random.poisson(1, 200)
ks = ks.astype(int) + 2
print(pd.Series(ks).value_counts(), "\n")


cols = ["age","workclass","education-num","marital-status","occupation",
    "relationship","race","sex","capital-gain","capital-loss","hours-per-week",
    "native-country"]

non_ordered_cols = ["workclass", "marital-status", "occupation", "relationship",
"race", "native-country"]
df = pd.read_csv("../../../datasets/adult/adult_cat_mapped.csv")
vals = []
for c in df.columns[:-1]:
    s = set(df[c])
    vals.append(s)

print(vals)
for i, k in enumerate(ks):
    #make_mondrian_config(i+1, k, vals)
    #make_datafly_config(i+1, k, vals, cols, non_ordered_cols, shuffled=True)
    make_datafly_config(i+1, k, vals, cols, non_ordered_cols, shuffled=False)
