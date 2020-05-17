import pandas as pd

import sys
import shutil

from sklearn.model_selection import train_test_split
import sklearn.metrics
import autosklearn.regression

import pickle
import yaml

try:
     shutil.rmtree("/vol/bitbucket/rd2016/tmp")
     shutil.rmtree("/vol/bitbucket/rd2016/out")
except:
    print("error deleting previous training dirs")

f = open(f"{sys.argv[1]}/config.yaml")
config = yaml.load(f, Loader=yaml.FullLoader)
print(f"##### Config file: {sys.argv[1]} ######")
print(yaml.dump(config))
sys.path.append(config["analysis_name"])


def auto_tune(file, config):
    print("Loading anonymized training data...")
    data = pd.read_csv(f"{file}/trainset.csv")

    print("Preparing train and test split...")
    y = data["lr_acc"]
    X = data.drop(["lr_acc", "algo", "no", "auroc"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    print(y)
    print(X)

    print(f"X shapes: {X_test.shape, X_train.shape}")
    print(f"y shapes: {y_test.shape, y_train.shape}")

    print(f"Starting training on {file}...\n")

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=24*60*60,
        per_run_time_limit=60*60,
        disable_evaluator_output=False,
        initial_configurations_via_metalearning=False,
        n_jobs=6,
        tmp_folder="/vol/bitbucket/rd2016/tmp",
        output_folder="/vol/bitbucket/rd2016/out"
    )
    automl.fit(X_train, y_train, X_test=X_test, y_test=y_test)
    print("############ Done ! ################\n")

    ensemble = automl.get_models_with_weights()
    print(automl.show_models())
    print(automl.sprint_statistics())
    predictions = automl.predict(X_test)
    fn = f"automl_{config['analysis_name']}.pkl"
    outfile = open(fn, 'wb+')
    pickle.dump(automl, outfile)



if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Wrong number of arguments: python3 auto_tune.py [directory]")
    else:

        print(f"Tuning {sys.argv[1]}")
        try:
            shutil.rmtree("/vol/bitbucket/rd2016/tmp")
            shutil.rmtree("/vol/bitbucket/rd2016/out")
            print("Emptied previous tmp and out dirs")
        except:
            print("No previous tmp and out dirs")

        f = open(f"{sys.argv[1]}/config.yaml")
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(f"##### Config file: {sys.argv[1]} ######")
        print(yaml.dump(config))
        sys.path.append(config["analysis_name"])

        auto_tune(sys.argv[1], config)



