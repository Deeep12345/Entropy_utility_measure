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
from metrics.ambiguity_metric import ambiguity_metric
from metrics.hellinger_metric import hellinger_metric
from metrics.bivariate_corr_metric import bivariate_corr_metric
from metrics.information_loss_metrics import discern_metric, IL_metric
from metrics.avg_eq_size_metric import avg_eq_size_metric
from metrics.granularity_metric import granularity_metric
from metrics.distance_squared_error import distance_squared_error
from metrics.ML_train import train_test


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
QIs = config["cols"][:-1]

full_anon = orig_data.copy()
full_anon.replace(0, 1, inplace=True)
max_H = cond_entr_metric(orig_data, full_anon, QIs)

results = {}
for no in range(1, config["no_instances"]+1):

    for algo in config["algos_used"]:
        print(no, algo)
        r = {}
        anon_data = load_csv(algo, no, original_oh=False)
        conf = load_config(algo, no)

        r["entropy"] = cond_entr_metric(orig_data, anon_data, QIs)/max_H
        r["cm"] = class_metric(anon_data)
        r["dm"] = diam_metric(anon_data)
        r["discern"] = discern_metric(anon_data)
        r["precision"] = precision_metric(anon_data, algo, no, conf, QIs)
        r["ilm"] = IL_metric(anon_data, QIs)
        r["hellinger"] = hellinger_metric(orig_data, anon_data, QIs)
        r["bivariate_corr"] = bivariate_corr_metric(orig_data, anon_data, QIs)
        r["avg_eq_size"] = avg_eq_size_metric(anon_data, conf)
        r["ambiguity"] = ambiguity_metric(anon_data, QIs)
        r["granularity"] = granularity_metric(anon_data, QIs)
        r["dse"] = distance_squared_error(anon_data, orig_data, QIs)
        r["auroc"], r["lr_acc"] = train_test(orig_data, anon_data, "logreg")
        results[(algo, no)] = r

    if no % 20 == 0:
        df = pd.DataFrame.from_dict(results, orient='index')
        df.to_csv(f"{config['analysis_name']}/metrics.csv",
                    index_label=["algo","no"])

print(results)
df = pd.DataFrame.from_dict(results, orient='index')
df.to_csv(f"{config['analysis_name']}/metrics.csv",
            index_label=["algo","no"])
