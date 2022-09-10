import os
from pathlib import Path
import sys

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)

from dotenv import load_dotenv
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ParameterGrid
from classification.train_test_model import create_config
from classification.classifiers import CLASSIFIERS
from data_acquisition.preprocessing import preprocess_recording
from feature_extraction.extractors import CSPExtractor
import numpy as np
from datetime import datetime
import json

from mne.io import read_raw_gdf

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
RESULTS_PATH = os.environ["RESULTS_PATH"]


def array_from_gdf(path):
    gdf = read_raw_gdf(path)
    gdf.drop_channels(["EOG-left","EOG-central","EOG-right"])
    
    raw_onsets = gdf._raw_extras[0]["events"][1]
    raw_labels = gdf._raw_extras[0]["events"][2]
    assert len((raw_onsets) == len(raw_labels))

    onsets, labels = [], []
    for i, label in enumerate(raw_labels):
        if 769 <= label <= 772:
            onsets.append(raw_onsets[i])
            labels.append(label-768)         
    
    print(np.unique(labels, return_counts=True))

if __name__ == "__main__":
    dir = Path(f"{DATA_PATH}/recordings/competition_IV_2a")
    users = [path for path in dir.glob("*.gdf") if path.stem[-1] == "T"] # Only load test sessions with label information
    #users = [users[0]]

    for user in users:
        print(user)
        array_from_gdf(user)
