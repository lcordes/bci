import os
from pathlib import Path
import sys

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)

from natsort import natsorted
from dotenv import load_dotenv
import numpy as np
from classification.train_test_model import create_config
from classification.classifiers import CLASSIFIERS
from analysis.within_loocv_classification import loocv
from data_acquisition.preprocessing import preprocess_openbci
import numpy as np
from sklearn.metrics import accuracy_score
from random import sample
import mne
mne.set_log_level(verbose="ERROR")
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
RESULTS_PATH = os.environ["RESULTS_PATH"]


def run_config(config, users):
    accs = []
    for user in users:
        X, y = preprocess_openbci(user, config)
        y_true, y_pred = loocv(X, y, config)
        acc = accuracy_score(y_true, y_pred)
        accs.append(acc)

    return accs



if __name__ == "__main__":
    dir = Path(f"{DATA_PATH}/recordings/competition_IV_2a")
    users = natsorted([path.stem for path in dir.glob("*.gdf") if path.stem[-1] == "T"]) # Only loads test sessions (have label info)

    configs = [
        create_config(clf_specific={"solver": "eigen", "shrinkage": "auto"}, bandpass=(8, 30)),
            ]
    
    title = "Within classification loocv(benchmark data, 8-30)"
    for c in configs:
        print("Parameters", c)
        accs = run_config(c, users)
        print(f"Mean acc: {np.mean(accs):.3f} (SD = {np.std(accs):.3f})")
        with open(f"{RESULTS_PATH}/within/{title}.npy", 'wb') as f:
            np.save(f, np.array(accs))
