import pickle
import numpy as np
import pandas as pd
import autosklearn
import matplotlib.pyplot as plt
import sys

def pairwise_comparisons(model, dataset, target):
    #until new model, remove new metrics
    X_test = pd.read_csv(f"{dataset}/metrics_testset.csv")
    X_test = X_test.drop(["lr_acc", "algo", "no", "auroc", "precision"], axis=1)

    y_test = pd.read_csv(f"{dataset}/accuracies_testset.csv")
    y_test = y_test[target]

    print(X_test.shape, y_test.shape)

    predicted = model.predict(X_test)

    correct_counter, test_counter = 0, 0

    for i, x in enumerate(y_test[:-1]):
        for j, y in enumerate(y_test[i:]):
            if j != 0:
                print(f"Comparing test dataset {i} to test dataset {j+i}")
                print(f"\tpredicted {target} for {i}: {predicted[i]}")
                print(f"\tpredicted {target} for {j+i}: {predicted[j+i]}")
                predicted_larger = predicted[i] >= predicted[j+i]
                print(f"\t{i} predicted more useful than {j+i}? {predicted_larger}")
                print(f"\tTrue {target} for {i}: {y_test[i]}")
                print(f"\tTrue {target} for {j+i}: {y_test[j+i]}")
                larger = y_test[i] >= y_test[j+i]
                print(f"\t{i} truly more useful than {j+i}? {larger}")
                if larger == predicted_larger:
                    print("metrics work!")
                    correct_counter+=1
                test_counter+=1

    print(f"The super metric correctly predicted better datasets {correct_counter/test_counter} percent of the time")
    return correct_counter/test_counter


# dataset = sys.argv[1]
# target = sys.argv[2]

results = {}
for algo in ["mondrian"]:#, "datafly", "datafly_shuffled"]:
    res = {}
    for dataset in ["birth", "ring", "adult", "heart"]:
        dataset += "_randoms"
        r = {}
        for target in ["lr_acc", "lr_auroc", "knn_pca_acc", "knn_pca_auroc",
                        "rf_pca_acc", "rf_pca_auroc"]:
            model = pickle.load(open(f"autosklearn_models/{dataset}_{target}_{algo}.pkl", "rb"))
            print(model.get_models_with_weights())

            t_res = pairwise_comparisons(model, dataset, target)
            r[target] = t_res

        res[dataset] = r
    results[algo] = res

print(results)
df = pd.DataFrame.from_dict(res, orient='index')
df.to_csv(f"correct_choices.csv",index_label=["dataset"])
