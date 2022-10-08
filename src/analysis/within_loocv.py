from natsort import natsorted
from sklearn.model_selection import ParameterGrid, LeaveOneOut
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
src_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(src_dir)
from pipeline.utilities import create_config, train_model, test_model
from pipeline.preprocessing import preprocess_recording, get_users

import os
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
RESULTS_PATH = os.environ["RESULTS_PATH"]


def loocv(X, y, config):
    looc = LeaveOneOut()
    looc.get_n_splits(X)
    scores = []

    for train_index, test_index in looc.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = train_model(X_train, y_train, config)
        acc = test_model(X_test, y_test, model)
        scores.append(acc)
    return np.mean(scores)


def run_configs(configs, save=False):
    print(f"Getting results for {len(configs)} configs")
   
    results = {"Config": [], "Mean": [], "SD": []}
    for config in configs:
        users = get_users(config)
        accs = []
        for user in users:
            X, y = preprocess_recording(user, config)
            acc = loocv(X, y, config)
            accs.append(acc)
            print(f"{user}: {acc:.3f}")
        mean_acc = np.mean(accs)
        mean_sd = np.std(accs)
        print(f"Acc: {mean_acc:.3f}, SD: {mean_sd:.3f}\n")
        results["Config"].append(config["description"])
        results["Mean"].append(mean_acc)
        results["SD"].append(mean_sd)
    
    if save:
        df = pd.DataFrame(results)
        df.to_csv(f"{RESULTS_PATH}/within_loocv/config_results_{datetime.now().strftime('%d-%m-%Y_%H-%M')}")

def get_grid_configs(general, clf_specific):
    permutations = []

    # Estimate permutations separately per classifier, then join
    for clf in clf_specific.keys():
        clf_params = {**general, **clf_specific[clf]}
        clf_perms = list(ParameterGrid(clf_params))
        for perm in clf_perms:
            perm["model_type"] = clf
        permutations.extend(clf_perms)

    config_params = []
    for perm in permutations:
        # separate permutation args into general and clf specific
        hyper_args, clf_specific_args = {}, {}
        for key, value in perm.items():
            if key in general.keys() or key == "model_type":
                hyper_args[key] = value
            else:
                clf_specific_args[key] = value
        hyper_args["clf_specific"] = clf_specific_args
        config_params.append(hyper_args)
    return [create_config(config_arg) for config_arg in config_params]


if __name__ == "__main__":
    manual_configs = [
        create_config({"data_set": "training"})
    ]
    print(manual_configs)

    general = {"imagery_window": [3,4]}
    clf_specific = {"LDA": {"shrinkage": ["auto", None], "solver": ["eigen"]}}
    grid_configs = get_grid_configs(general, clf_specific) 
    
    run_configs(manual_configs, save=True)
