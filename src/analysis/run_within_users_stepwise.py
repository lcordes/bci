import sys
from pathlib import Path
import json
from random import sample

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)

from data_acquisition.preprocessing import preprocess_recording
from analysis.run_stepwise_selection import stepwise_selection
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


def single_split_validation(users, config, train_size):
    data = {"users": [], "classifier": config["model_type"], "train_size": train_size}
    for user in users:
        _, y = preprocess_recording(user, config)
        split1, split2 = train_test_split(
            list(range(TRIALS_PER_CLASS)), train_size=train_size, stratify=y
        )

        # Optimal set tested on split 1 (train)
        accs, selected = stepwise_selection(users, user, config, subset_idx=split1)
        train_acc = accs[np.argmax(accs)]
        best_set = selected[: (np.argmax(accs) + 1)]
        print(f"Final model for {user} selected: {best_set}")
        print(f"Train acc on split 1: {np.round(train_acc, 3)}")

        # Optimal set tested on split 2 (test)
        model = train_model(best_set, config)
        val_acc = test_model(user, model, subset_idx=split2)
        print("Validation acc on split 2:", np.round(val_acc, 3))

        # Also test on a set of randomly drawn users (same length) as baseline
        base_candidates = [u for u in users if not u == user]
        base_set = sample(base_candidates, len(best_set))

        # Baseline set tested on split 1 (train)
        model = train_model(base_set, config)
        base_train_acc = test_model(user, model, subset_idx=split1)
        print("Baseline train acc on split 1:", np.round(base_train_acc, 3))

        # Baseline set tested on split 1 (train)
        model = train_model(base_set, config)
        base_val_acc = test_model(user, model, subset_idx=split2)
        print("Baseline validation acc on split 2:", np.round(base_val_acc, 3))

        # Within, trained on split 1, tested on split 2
        within_model = train_model(user, config, subset_idx=split1)
        within_acc = test_model(user, within_model, subset_idx=split2)
        print("Within acc on split 2:", np.round(within_acc, 3), "\n")

        result = dict(
            user=user,
            train_acc=train_acc,
            base_train_acc=base_train_acc,
            val_acc=val_acc,
            base_val_acc=base_val_acc,
            within_acc=within_acc,
        )
        data["users"].append(result)
    return data


if __name__ == "__main__":
    dir = Path(f"{DATA_PATH}/recordings/users")
    users = natsorted([path.stem for path in dir.glob("*.hdf5")])

    params = {"model_type": "LDA"}
    config = create_config(**params)
    train_size = 0.67
    results = single_split_validation(users, config, train_size=train_size)
    with open(
        f"{RESULTS_PATH}/within_stepwise/within_stepwise_{params['model_type']}_{train_size}.json",
        "w",
    ) as f:
        json.dump(results, f)
