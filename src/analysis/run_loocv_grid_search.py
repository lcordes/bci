from natsort import natsorted
import os
from pathlib import Path
import sys

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)

from dotenv import load_dotenv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ParameterGrid
from classification.train_test_model import create_config
from classification.classifiers import CLASSIFIERS
from data_acquisition.preprocessing import preprocess_recording
from feature_extraction.extractors import CSPExtractor
import numpy as np
from datetime import datetime
import json



load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
RESULTS_PATH = os.environ["RESULTS_PATH"]


def score_loocv(X, y, config):
    looc = LeaveOneOut()
    looc.get_n_splits(X)
    y_true = []
    y_pred = []

    for train_index, test_index in looc.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        extractor = CSPExtractor(config)
        X_train_transformed = extractor.fit_transform(X_train, y_train)
        X_test_transformed = extractor.transform(X_test)

        clf = CLASSIFIERS[config["model_type"]](config)
        clf.fit(X_train_transformed, y_train)
        y_true.append(int(y_test))
        y_pred.append(int(clf.predict(X_test_transformed)))

    return y_true, y_pred


def run_grid_search(users, configs, testing=False):
    print(f"Running grid search for {len(configs)} configs")
    if testing:
        results_file = f"{RESULTS_PATH}/loocv_grid_search/grid_search_run_test.json"
    else:
        results_file = f"{RESULTS_PATH}/loocv_grid_search/grid_search_run_{datetime.now().strftime('%d-%m-%Y_%H-%M')}.json"

    with open(results_file, "w") as f:
        json.dump([], f)

    failed = []
    for c, config in enumerate(configs):
        try:
            print(f"C{c+1}",config['description'])
            config_results = {"config": config, "users": []}
            config_results["start_time"] = datetime.now().strftime(
                "%d-%m-%Y_%H-%M-%S")

            for recording_name in users:
                X, y = preprocess_recording(recording_name, config)
                y_true, y_pred = score_loocv(X, y, config)
                user_results = {"user": recording_name, "y_true": y_true, "y_pred": y_pred}
                config_results["users"].append(user_results)
            config_results["end_time"] = datetime.now().strftime(
                "%d-%m-%Y_%H-%M-%S")

            user_avgs = [np.mean([true == pred for true, pred in zip(u["y_true"], u["y_pred"])]) for u in config_results["users"]]
            print(f"Acc: {np.round(np.mean(user_avgs),3)}, SD: {np.round(np.std(user_avgs),3)}\n")
            save_results(config_results, results_file)

        except Exception as e:
            print(
                f"{datetime.now().strftime('%H:%M')}: Config {c+1} ({config['description']}) got error:",
                e,
            )
            failed.append(c + 1)

    if failed:
        print("Configs failed:", failed)
    else:
        print("All configs ran successfully.")


def save_results(config_results, results_file):
    with open(results_file, "r") as f:
        data = json.load(f)

    data.append(config_results)
    with open(results_file, "w") as f:
        json.dump(data, f)


def get_grid_configs(hyper_grid, clf_specific):
    permutations = []

    # Estimate permutations separately per classifier, then join
    for clf in clf_specific.keys():

        clf_params = {**hyper_grid, **clf_specific[clf]}
        clf_perms = list(ParameterGrid(clf_params))
        for perm in clf_perms:
            perm["model_type"] = clf
        permutations.extend(clf_perms)

    config_args = []
    for perm in permutations:
        # separate permutation args into general and clf specific
        hyper_args, clf_specific_args = {}, {}
        for key, value in perm.items():
            if key in hyper_grid.keys() or key == "model_type":
                hyper_args[key] = value
            else:
                clf_specific_args[key] = value
        hyper_args["clf_specific"] = clf_specific_args

        hyper_args["description"] = ";".join(
            f"{param}={value}" for param, value in perm.items()
        )
        config_args.append(hyper_args)
    return [create_config(**config_arg) for config_arg in config_args]


if __name__ == "__main__":
    dir = Path(f"{DATA_PATH}/recordings/training_data_collection")
    users = natsorted([path.stem for path in dir.glob("*.hdf5")])

    hyper = {"bandpass": [(10, 12)]}
    lda_specific = {"LDA": {"shrinkage": ["auto"], "solver": ["eigen"]}}
    lda_configs = get_grid_configs(hyper, lda_specific)

    run_grid_search(users, lda_configs, testing=False)
