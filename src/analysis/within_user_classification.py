
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
RESULTS_PATH = os.environ["RESULTS_PATH"]

import sys
from pathlib import Path
src_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(src_dir)

from pipeline.utilities import create_config
from pipeline.preprocessing import preprocess_recording, get_users
from pipeline.utilities import train_model, test_model
from sklearn.model_selection import train_test_split


def within_train_test(config, train_size=0.5, repetitions=100, online_sim=False):
    users = get_users(config)
    X_all, y_all = [], []
    for user in users:
        X, y = preprocess_recording(user, config)
        X_all.append(X)
        y_all.append(y)

    n_trials = X_all[0].shape[0]
    results = np.zeros((len(users), repetitions))

    for u, user in enumerate(users):
        for rep in range(repetitions):           
            if config["data_set"] == "evaluation" and online_sim:
                train_idx = list(range(60))
                test_idx = list(range(60, 90))
            else:
                train_idx, test_idx = train_test_split(
                    list(range(n_trials)),
                     train_size=train_size,
                      stratify=y_all[u]
                )
            X_train = X_all[u][train_idx, :, :]
            X_test = X_all[u][test_idx, :, :]
            y_train = y_all[u][train_idx]
            y_test = y_all[u][test_idx]

            model = train_model(X_train, y_train, config)
            acc = test_model(X_test, y_test, model)
            results[u, rep] = acc
        print(f"{user}: {np.mean(results[u, :]):.3f}")

    print(f"Average: {np.mean(np.mean(results, axis=1)):.3f}")

    return np.mean(results, axis=1)



if __name__ == "__main__":
    config = create_config({"data_set": "evaluation"})
    within_train_test(config, repetitions=1)