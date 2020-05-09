import pandas as pd
import sys
from sklearn.model_selection import train_test_split
import autosklearn.regression


def auto_tune(file, y_name):
    print("Loading anonymized training data...")
    data = pd.read_csv(f"{file}/metrics.csv")

    print("Preparing train and test split...")
    y = data[y_name]
    X = data.drop([y_name, "algo", "no"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    print(y)
    print(X)

    print(f"X shapes: {X_test.shape, X_train.shape}")
    print(f"y shapes: {y_test.shape, y_train.shape}")

    print(f"Starting training on {file}...\n")
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=2*60*60,
        disable_evaluator_output=False,
        initial_configurations_via_metalearning=False,
        n_jobs=1
    )
    automl.fit(X_train, y_train)
    print("############ Done ! ################\n")

    ensemble = automl.get_models_with_weights()
    print(automl.show_models())
    print(automl.sprint_statistics())
    predictions = automl.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
    with open(f'automl_{file}.pkl', 'w+') as output:
        pickle.dump(automl,output)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Wrong number of arguments: python3 auto_tune.py [directory] [ycol]")
    else:
        print(f"Tuning {sys.argv[1]}")
        auto_tune(sys.argv[1], sys.argv[2])
