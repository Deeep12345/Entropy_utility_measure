import pandas as pd
import sys
from sklearn.model_selection import train_test_split
import sklearn.metrics
import autosklearn.regression
import pickle
import yaml

f = open(f"{sys.argv[1]}/config.yaml")
config = yaml.load(f, Loader=yaml.FullLoader)
print(f"##### Config file: {sys.argv[1]} ######")
print(yaml.dump(config))
sys.path.append(config["analysis_name"])


def auto_tune(file, y_name):
    print("Loading anonymized training data...")
    data = pd.read_csv(f"{file}/metrics.csv")

    print("Preparing train and test split...")
    y = data[y_name]
    X = data.drop([y_name, "algo", "no", "acc", "auroc"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    print(y)
    print(X)

    print(f"X shapes: {X_test.shape, X_train.shape}")
    print(f"y shapes: {y_test.shape, y_train.shape}")

    print(f"Starting training on {file}...\n")

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=1*60*60,
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
    if len(sys.argv) != 3:
        print("Wrong number of arguments: python3 auto_tune.py [directory] [ycol]")
    else:
        print(f"Tuning {sys.argv[1]}")
        auto_tune(sys.argv[1], sys.argv[2])
