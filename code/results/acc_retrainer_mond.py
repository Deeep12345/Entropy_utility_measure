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

    if algo == "datafly_shuffled":
        f = f"../anon_data/{config['analysis_name']}/datafly{no}_oh_shuffled.csv"
    else:
        f = f"../anon_data/{config['analysis_name']}/{algo}{no}_oh.csv"
    return pd.read_csv(f)


def load_config(algo, no):
    if algo == "datafly_shuffled":
        f = f"../toolbox_linux64/configs/{config['analysis_name']}/datafly{no}_shuffled.xml"
    else:
        f = f"../toolbox_linux64/configs/{config['analysis_name']}/{algo}{no}.xml"
    xml = et.parse(f)
    root = xml.getroot()
    return root

orig_data = load_csv(0,0)
results = {}

for no in range(2, config["no_instances"]+1):

    for algo in ["mondrian"]:
        r = {}
        anon_data = load_csv(algo, no, original_oh=False)
        conf = load_config(algo, no)
        r["acc"] = train_test(orig_data, anon_data)

        print(no, algo, r)
        results[(algo, no)] = r

    if no % 20 == 0:
        df = pd.DataFrame.from_dict(results, orient='index')
        df.to_csv(f"{config['analysis_name']}/acc_mondrian.csv",
                    index_label=["algo","no"])

print(results)
df = pd.DataFrame.from_dict(results, orient='index')
df.to_csv(f"{config['analysis_name']}/acc_mondrian.csv",
            index_label=["algo","no"])
