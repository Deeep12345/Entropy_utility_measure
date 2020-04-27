import pandas as pd
import numpy as np
import xml.etree.ElementTree as et
import xmltodict as xdict

import sys
import yaml
from tqdm import tqdm

f = open(f"{sys.argv[1]}/config.yaml")
config = yaml.load(f, Loader=yaml.FullLoader)
print(f"##### Config file: {sys.argv[1]} ######")
print(yaml.dump(config))
sys.path.append(config["analysis_name"])

from metrics.diameter_metric import diam_metric
from metrics.classification_metric import class_metric
from metrics.entropy import cond_entr_metric
from metrics.hierarchy_metrics import precision_metric
from metrics.information_loss_metrics import discern_metric, IL_metric
from metrics.RFCs_train import train_test

def load_csv(algo, no, original_oh=True):
    if original_oh:
        return pd.read_csv(f"../anon_data/{config['analysis_name']}/original_oh.csv")

    f = f"../anon_data/{config['analysis_name']}/{algo}{no}_oh"
    shuff = "_shuffled.csv" if algo == "datafly_shuffled" else ".csv"
    fp = f + shuff
    return pd.read_csv(fp)


def load_config(algo, no):
    f = f"../toolbox_linux64/configs/{config['analysis_name']}/{algo}{no}"
    shuff = "_shuffled.xml" if algo == "datafly_shuffled" else ".xml"
    fp = f + shuff
    print(algo, fp)
    xml = et.parse(fp)
    root = xml.getroot()
    return root

QIs = config["cols"][:-1]

orig_data = load_csv(0,0)
full_anon = orig_data.copy()
full_anon.replace(0, 1, inplace=True)

max_H = cond_entr_metric(orig_data, full_anon, QIs)

results = {}
for no in tqdm(range(1, config["no_instances"]+1)):

    for algo in config["algos_used"]:
        r = {}
        anon_data = load_csv(algo, no, original_oh=False)
        conf = load_config(algo, no)

        r["precision"] = precision_metric(anon_data, algo, no, conf, QIs)
        r["dm"] = diam_metric(anon_data)
        r["cm"] = class_metric(anon_data)
        r["entropy"] = cond_entr_metric(orig_data, anon_data, QIs)/max_H
        r["discern"] = discern_metric(anon_data)
        r["ilm"] = IL_metric(anon_data, QIs)
        print("did metrics successfuly")
        r["acc"] = train_test(orig_data, anon_data)

        print(no, algo, r)
        results[(algo, no)] = r

print(results)
df = pd.DataFrame.from_dict(results, orient='index')
df.to_csv(f"{config['analysis_name']}/metrics_prectest.csv",
            index_label=["algo","no"])
