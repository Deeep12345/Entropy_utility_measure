import numpy as np
import pandas as pd

import yaml
import sys
import re


f = open(f"{sys.argv[1]}/config.yaml")
config = yaml.load(f, Loader=yaml.FullLoader)
sys.path.append(config["analysis_name"])

cols = config["cols"]
data = pd.read_csv(f"../../datasets/{config['dataset_path']}", names=cols)
oh = data[cols[:-1]]
oh = pd.get_dummies(oh, columns=cols[:-1])
oh["class"] = data["class"]

m = {}
for c in cols:
    if c == "class":
        m[c] = c
    else:
        rel_cols = list(filter(lambda x: c == x[:len(c)], oh.columns))
        for i in range(len(rel_cols)):
            m[c+f"_{i}"] = c+str(i)

oh = oh.rename(m, axis=1)
oh.to_csv(f"{config['analysis_name']}/original_oh.csv", index=False)
