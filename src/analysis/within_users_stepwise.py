import sys
from pathlib import Path
import json

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)

from classification.classifiers import (
    LDAClassifier,
    SVMClassifier,
    RFClassifier,
    MLPClassifier,
)
from data_acquisition.preprocessing import preprocess_recording
from analysis.stepwise_selection import stepwise_selection
from classification.train_test_model import train_model, test_model, create_config
from sklearn.model_selection import train_test_split
import numpy as np
from natsort import natsorted
import os
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
RESULTS_PATH = os.environ["RESULTS_PATH"]
TRIALS_PER_CLASS = int(os.environ["TRIALS_PER_CLASS"]) * 3


def single_split_validation(users, classifier, config, train_size, save_model=False):
    data = {"users": [], "classifier": classifier.__name__, "train_size": train_size}
    for user in users:
        _, y = preprocess_recording(user, config)
        split1, split2 = train_test_split(
            list(range(TRIALS_PER_CLASS)), train_size=train_size, stratify=y
        )
        accs, selected = stepwise_selection(
            users, user, classifier, config, subset_idx=split1
        )
        best_set = selected[: (np.argmax(accs) + 1)]
        train_acc = accs[np.argmax(accs)]
        print(f"Final model for {user} selected: {best_set}")
        print(f"Train acc on split 1: {np.round(train_acc, 3)}")
        model = train_model(best_set, classifier, config, save=save_model)
        val_acc = test_model(user, model, subset_idx=split2)
        print("Validation acc on split 2:", np.round(val_acc, 3))

        within_model = train_model(
            user, classifier, config, save=save_model, subset_idx=split1
        )
        within_acc = test_model(user, within_model, subset_idx=split2)
        print("Within acc on split 2:", np.round(within_acc, 3), "\n")
        result = dict(
            user=user, train_acc=train_acc, val_acc=val_acc, within_acc=within_acc
        )
        data["users"].append(result)
    return data


if __name__ == "__main__":
    dir = Path(f"{DATA_PATH}/recordings/users")
    users = natsorted([path.stem for path in dir.glob("*.hdf5")])

    params = {}
    config = create_config(**params)
    classifier = LDAClassifier
    train_size = 0.34
    results = single_split_validation(users, classifier, config, train_size=train_size)
    with open(
        f"{RESULTS_PATH}/stepwise_selection/within_stepwise_{classifier.__name__}_{train_size}.json",
        "w",
    ) as f:
        json.dump(results, f)
