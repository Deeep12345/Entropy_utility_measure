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


#
# values = {}
# for col in data.columns[:-1]:
#     values[col] = set(data[col])
#     print(f"{col}..... {values[col]}")
#
# ps = list(mit.set_partitions([1,2,3,4], 2))
# ps.extend(list(mit.set_partitions([1,2,3,4], 3)))
# ps.extend(list(mit.set_partitions([1,2,3,4], 4)))
#
# choices = {c:len(set(values[c])) for c in data.columns[:-1]}
#
# def create_4tree():
#     x = list(range(1,5))
#     res = "<vgh value='[1:4]'>\n"
#
#     part = random.choice(ps)
#     fp = flatten(part)
#
#     mapping = {}
#     for j, m in enumerate(fp):
#         mapping[m] = j+1
#     mapped_p = [list(map(lambda x: mapping[x], l)) for l in part]
#
#     for p in mapped_p:
#         res += f"\t<node value='[{min(p)}:{max(p)}]'/>\n"
#
#     res += "</vgh>"
#     return res, mapping
#
#
# def make_datafly_config(name):
#     mappings = {}
#     full_xml = """
#     <config method='Datafly' k='2'>
#     <input filename='../../datasets/birth_control/cmc_ups_4cat.csv' separator=','/>
#      <!-- If left blank, separator will be set as comma by default.-->"""
#     full_xml += f"<output filename='../anon_data/birth_random_groups/datafly{name}.csv' format ='genVals'/>"
#     full_xml+= """<!-- Format options = {genVals, genValsDist, anatomy}. If left blank,
#     output format will be set as genVals by default.-->
#     <id> <!-- List of identifier attributes, if any, these will be excluded from the output -->
#     </id>
#     <qid>"""
#
#     for i, attr in enumerate(data.columns[:-1]):
#         full_xml = full_xml + (f"<att index='{i}' name='{attr}'>")
#
#         if choices[attr] == 2:
#             full_xml = full_xml + ("<vgh value='[0:1]'/>")
#         else:
#             tree, mapping = create_4tree()
#             #print(attr, selected_t[0])
#             #print(trees_4[selected_t[0]]["xml_tree"])
#             mappings[attr] = mapping
#             full_xml = full_xml + tree
#
#         full_xml = full_xml + (f"</att>")
#
#     full_xml = full_xml + ("</qid>")
#
#     full_xml = full_xml + ("""
#     <sens>
#     <att index='9' name='class'/>
#     </sens>""")
#
#     full_xml = full_xml + "<mapping>"
#     for attr, mapp in mappings.items():
#         full_xml = full_xml + f"<att name='{attr}'>"
#         for orig, m in mapp.items():
#             full_xml = full_xml + f"<node used='{m}' mapbackto='{orig}'/>"
#         full_xml = full_xml + f"</att>"
#     full_xml = full_xml + "</mapping>"
#     full_xml = full_xml + "</config>"
#     xml_tree = bs(full_xml, 'xml')
#     f = open(f"datafly{name}.xml", "w+")
#     f.write(xml_tree.prettify())
#     f.close()
#
#     return full_xml
#
#
def make_mondrian_config(name, k, no_cols, no_bins):
    mappings = {}
    full_xml = f"""
    <config method='Mondrian' k='{k}'>
    <input filename='../../datasets/ring/ring_cat.csv' separator=','/>"""
    full_xml += f"<output filename='../anon_data/ring_randoms/mondrian{name}.csv' format ='genVals'/>"
    full_xml += """<id></id><qid>"""

    for i in range(0,no_cols):
        full_xml += f"<att index='{i}' name='{i}th'>"
        full_xml += f"<vgh value='[0:{no_bins-1}]'/>"

        ord = list(range(0,no_bins))
        random.shuffle(ord)
        full_xml += "<map>"
        for j, m in enumerate(ord):
            full_xml += f"<entry cat='{j+1}' int='{m}'/>"
        full_xml += "</map>"
        full_xml += f"</att>"

    full_xml += "</qid>"

    full_xml += f"""
    <sens>
    <att index='{no_cols}' name='class'/>
    </sens>"""

    full_xml = full_xml + "</config>"
    xml_tree = bs(full_xml, 'xml')
    f = open(f"ring_randoms/mondrian{name}.xml", "w+")
    f.write(xml_tree.prettify())
    f.close()

    return xml_tree

ks = np.random.poisson(1, 200)
ks = ks.astype(int) + 2
print(pd.Series(ks).value_counts())
for i, k in enumerate(ks):
    no_bins = 20
    no_cols = 20
    conf = make_mondrian_config(i+1, k, no_cols, no_bins).prettify()
