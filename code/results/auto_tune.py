import pandas as pd

import sys
import shutil

from sklearn.model_selection import train_test_split
import sklearn.metrics
import autosklearn.regression

import pickle
import yaml

def auto_tune(dataset, target_var):
    print("Loading anonymized training data...")
    data = pd.read_csv(f"{dataset}/metrics_trainset.csv").drop('precision', axis=1)
    targets = pd.read_csv(f"{dataset}/accuracies_trainset.csv")

    print(f"Using {target_var}...")
    y = targets[target_var]
    X = data.drop(["lr_acc", "algo", "no", "auroc"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    print(y)
    print(X.columns)

    print(f"X shapes: {X_test.shape, X_train.shape}")
    print(f"y shapes: {y_test.shape, y_train.shape}")

    print(f"Starting training on {dataset}...\n")

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=24*60*60,
        per_run_time_limit=60*60,
        disable_evaluator_output=False,
        initial_configurations_via_metalearning=False,
        n_jobs=6,
        tmp_folder=f"/vol/bitbucket/rd2016/tmp_{dataset}_{target_var}",
        output_folder=f"/vol/bitbucket/rd2016/out_{dataset}_{target_var}"
    )
    automl.fit(X_train, y_train, X_test=X_test, y_test=y_test)
    print("############ Done ! ################\n")

    ensemble = automl.get_models_with_weights()
    print(automl.show_models())
    print(automl.sprint_statistics())
    predictions = automl.predict(X_test)
    fn = f"autosklearn_models/automl_{dataset}_{target_var}_noprec.pkl"
    outfile = open(fn, 'wb+')
    pickle.dump(automl, outfile)



if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Wrong number of arguments: python3 auto_tune.py [directory] [target_var]")
    else:
        dataset =sys.argv[1]
        target_var = sys.argv[2]
        print(f"Tuning {dataset} for {target_var}")

        # f = open(f"{sys.argv[1]}/config.yaml")
        # config = yaml.load(f, Loader=yaml.FullLoader)
        # print(f"##### Config file: {sys.argv[1]} ######")
        # print(yaml.dump(config))
        # sys.path.append(config["analysis_name"])

        try:
            shutil.rmtree(f"/vol/bitbucket/rd2016/tmp_{dataset}_{target_var}")
            shutil.rmtree(f"/vol/bitbucket/rd2016/out_{dataset}_{target_var}")
            print("Emptied previous tmp and out dirs")
        except:
            print("No previous tmp and out dirs")


        auto_tune(dataset, target_var)
