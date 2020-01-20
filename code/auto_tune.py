import pandas as pd

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification


def auto_tune():
    data = pd.read_csv("../adult/clean_adult.csv")
    y = data['salary']
    X = data.drop(columns=['salary'], axis=1)
    print(X)
    print(y)

    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=12*60*60,
        per_run_time_limit=60*60,
        tmp_folder='/tmp/autosklearn_parallel_1_example_tmp',
        output_folder='/tmp/autosklearn_parallel_1_example_out',
        disable_evaluator_output=False,
        n_jobs=4,
        seed=5,
        delete_output_folder_after_terminate=True,
        delete_tmp_folder_after_terminate=True,
    )
    automl.fit(X_train, y_train, dataset_name='adult_census')
    print("\n\n\n\n\n######################################")
    # Print the final ensemble constructed by auto-sklearn.
    print(automl.show_models())
    predictions = automl.predict(X_test)

    print("\n\n\n\n\n######################################")
    # Print the final ensemble constructed by auto-sklearn.
    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(automl.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))

    
    print("\n\n\n\n\n######################################")
    print(automl.get_params())

if __name__ == '__main__':
    auto_tune()
