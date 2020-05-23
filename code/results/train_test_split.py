import pandas as pd
import sys

dataset = sys.argv[1]

metrics = pd.read_csv(f"{dataset}/metrics.csv")
accs = pd.read_csv(f"{dataset}/accuracy_results.csv")

test_metrics = metrics[metrics["no"] <= 20]
train_metrics = metrics[metrics["no"] > 20]

test_accs = accs[accs["no"] <= 20]
train_accs = accs[accs["no"] > 20]

test_metrics.to_csv(f"{dataset}/metrics_testset.csv", index=False)
train_metrics.to_csv(f"{dataset}/metrics_trainset.csv", index=False)

test_accs.to_csv(f"{dataset}/accuracies_testset.csv", index=False)
train_accs.to_csv(f"{dataset}/accuracies_trainset.csv", index=False)
