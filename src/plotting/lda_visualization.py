import sys
from pathlib import Path

src_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(src_dir)

from pipeline.preprocessing import preprocess_recording, get_users
from pipeline.utilities import create_config
from pipeline.feature_extraction import CSPExtractor
from pipeline.classification import LDAClassifier
from pipeline.transfer_learning import get_align_mat, align
import numpy as np
import os
from dotenv import load_dotenv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

load_dotenv()
ONLINE_FILTER_LENGTH = float(os.environ["ONLINE_FILTER_LENGTH"])
RESULTS_PATH = os.environ["RESULTS_PATH"]

def lda_visualization(config, user, align_X=False, save=False, rm_outliers=True):    
    if user == "all":
        users = get_users(config)
        config["discard_railed"] = False
    else:
        users = [user]

    X_all, y_all = [], []
    for u in users:
        X, y = preprocess_recording(u, config)
        if align_X:
            align_mat = get_align_mat(X)
            X = align(X, align_mat)     
        X_all.append(X)
        y_all.extend(y)

    X_all = np.concatenate(X_all, axis=0)

    extractor = CSPExtractor(config)
    X_transformed = extractor.fit_transform(X_all, y_all)
    classifier = LDAClassifier(config)
    lda_comps = classifier.fit_transform(X_transformed, y_all)
    
    if rm_outliers:
        # Remove outliers more than three sds away from mean
        comp1, comp2 = lda_comps[:, 0], lda_comps[:, 1]
        comp1_included = abs(comp1 - np.mean(comp1)) < 4 * np.std(comp1)
        comp2_included = abs(comp2 - np.mean(comp2)) < 4 * np.std(comp2)
        included = np.logical_and(comp1_included, comp2_included)
        comp1 = comp1[included]
        comp2 = comp2[included]
        y_all = [val for i, val in enumerate(y_all) if included[i]]

    df = pd.DataFrame({"LDA Component 1": comp1,
    "LDA Component 2": comp2,
    "Class": y_all
    })
    df = df.sort_values("Class")
    df["Class"] = df["Class"].replace([1, 2, 3], ["Left hand", "Right hand", "Feet"])
    
    marker_size = 10 if user == "all" else 22
    sns.scatterplot(data=df, x="LDA Component 1", y="LDA Component 2", hue="Class", 
    palette="Set1", s=marker_size)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.legend(framealpha=0.8)
    if save:
        aligned = "_aligned" if align_X else ""
        title = f"lda_{config['data_set']}_{user}{aligned}"
        plt.savefig(f"{RESULTS_PATH}/data_exploration/{title}.png", dpi=400)
        plt.clf()
    else:
        plt.show()


if __name__ == "__main__":
    for data_set in ["training", "benchmark", "evaluation"]:
        for align_X in [False, True]:
            config = create_config({"data_set": data_set})
            lda_visualization(config, user="all", align_X=align_X, save=True)

    # config = create_config({"data_set": "evaluation"})
    # lda_visualization(config, user="E02", align_X=False, save=True)
    # lda_visualization(config, user="E06", align_X=False, save=True)
