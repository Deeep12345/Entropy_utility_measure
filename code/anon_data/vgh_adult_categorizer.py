import pandas as pd
import numpy as np
import sys

def categorize(file):
    data = pd.read_csv(file, index_col=0,
                       names=["index","age","workclass","fnlwgt",
                              "education","marital-status","occupation",
                              "race", "sex", "capital-gain", "capital-loss",
                              "hours-per-week", "native-country","salary"])

    data["workclass"] = data["workclass"].map({
        "[0:6]":"Suppressed",
        "[0:2]":"Gov. Employed",
        "[3:5]":"Self/Privately Employed",
        "[6:6]":"Employed-no pay"
    })

    data["education"] = data["education"].map({
        "[0:15]":"Suppressed",
        "[0:8]":"H.S. or less",
        "[0:4]":"Less than H.S.",
        "[5:8]":"Up to H.S.",
        "[9:15]":"Further Studies",
        "[9:13]":"Bachelors or less",
        "[14:15]":"Masters or more"
    })

    data["marital-status"] = data["marital-status"].map({
        "[0:6]":"Suppressed",
        "[0:2]":"Married",
        "[3:5]":"Once Married",
        "[6:6]":"Never Married"
    })

    data["occupation"] = data["occupation"].map({
        "[0:13]":"Suppressed",
        "[0:5]":"Manual job",
        "[6:10]":"Desk job",
        "[11:12]":"Military or Police",
        "[13:13]":"Other"
    })

    data["race"] = data["race"].map({
        "[0:4]":"Suppressed",
        "[0:0]":"Majority",
        "[1:4]":"Minority"
    })

    data["sex"] = data["sex"].map({
        "[0:1]":"Suppressed",
        "[1.0]":"Female",
        "[0.0]":"Male"
    })

    data["hours-per-week"] = data["hours-per-week"].map({
        "[0:100]":"Suppressed",
        "[0:20]":"Part time",
        "[21:40]":"Full time",
        "[41:60]":"Overtime",
        "[61:100]":"Workaholic"
    })

    data["native-country"] = data["native-country"].map({
        "[0:39]":"Suppressed",
        "[0:3]":"North American",
        "[4:16]":"South/Central American",
        "[17:26]":"Asian",
        "[27:38]":"European",
        "[39:39]":"Middle Eastern"
    })

    data["capital-gain"] = data["capital-gain"].map({
        "[0:100000)":"Suppressed",
        "[0:1)":"None",
        "[1:7298)":"Low",
        "[7298:100000)":"High"
    })

    data["capital-loss"] = data["capital-loss"].map({
        "[0:4357)":"Suppressed",
        "[0:1)":"None",
        "[1:1887)":"Low",
        "[1887:4357)":"High"
    })

    data.to_csv(f"{file[:-4]}_cat.csv")

    for i in data.columns:
        print(data[i].unique())

if __name__ == "__main__":
    categorize(sys.argv[1])
