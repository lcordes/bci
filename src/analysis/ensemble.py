"""
Trains a classifier for each user in the training set. Predictions on new data are pooled through majority vote.
"""


from natsort import natsorted
import os
from pathlib import Path
import sys

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)

from dotenv import load_dotenv
from classification.train_test_model import create_config, train_model, test_model
from data_acquisition.preprocessing import preprocess_openbci
import numpy as np
from scipy.stats import mode


load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
RESULTS_PATH = os.environ["RESULTS_PATH"]

def loocv_ensemble(users, config):
    accs = []
    agreements = []
    for test_user in users:

        # Create subset exluding the current test user
        subset = [u for u in users if u != test_user]
        _, y = preprocess_openbci(test_user, config)

        # Train clfs for all other users
        clf_preds = np.zeros((len(y), len((subset))))
        for i, subset_user in enumerate(subset):
            model = train_model(subset_user, config)
            clf_preds[:, i] = test_model(test_user, model, score=False)

        # Get majority vote
        ensemble_preds, counts = mode(clf_preds, axis=1, keepdims=True)
        agree = np.mean(counts)
        print(f"User {test_user}")
        print("Average votes for winning class:", agree)
        scores = [true == pred for true, pred in zip(y, list(ensemble_preds))]
        acc = np.mean(scores)
        accs.append(acc)
        agreements.append(agree)
        print("Accuracy", np.round(acc, 3))
    
    return np.mean(accs), np.mean(agreements)



if __name__ == "__main__":
    dir = Path(f"{DATA_PATH}/recordings/training_data_collection")
    users = natsorted([path.stem for path in dir.glob("*.hdf5")])

    config = create_config(model_type="LDA", clf_specific={"solver": "eigen", "shrinkage": "auto"})
    total_acc, agreement = loocv_ensemble(users, config)
    print("Accuracy across users:", np.round(total_acc, 3))
    print("Average agreement on winning class:", np.round(agreement, 3))
