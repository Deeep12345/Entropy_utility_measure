import pickle
import numpy as np
import pandas as pd
import autosklearn
import matplotlib.pyplot as plt
import sys

fn = sys.argv[1]

model = pickle.load(open(f"automl_{fn}.pkl", "rb"))
print(model.get_models_with_weights())


#until new model, remove new metrics
testset = pd.read_csv(f"{fn}/testset.csv")
X_test = testset.drop(["lr_acc", "algo", "no", "auroc"], axis=1)
y_test = testset["lr_acc"]

print(X_test, y_test)
print(X_test.shape, y_test.shape)

predicted = model.predict(X_test)
print("##### Predictions ######\n", predicted)
abs_dist = np.abs(predicted - y_test)
print("##### abs(y_test - predicted) ######\n", abs_dist)

plt.hist(abs_dist, label="abs_dist")
plt.title("Absolute distance between accuracy predictions and true values")
plt.xlabel("|y_true - y_pred|")
plt.ylabel("count")
plt.show()
