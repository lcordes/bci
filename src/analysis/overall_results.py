import pandas as pd
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
from pipeline.preprocessing import get_users
from analysis.between_users_transfer import between_classification
from analysis.hyperparameter_estimation import within_loocv
from analysis.within_train_test import within_train_test


def compile_results(config, save=False):
    users = get_users(config)
    df = pd.DataFrame(columns=users)

    print("Working on between users transfer")
    transfer_data_set = "training" if config["data_set"] == "evaluation" else config["data_set"]
    df.loc["Baseline Transfer"], df.loc["EA Transfer"] = between_classification(config, transfer_data_set=transfer_data_set)
    print("\nWorking on within 50-50")
    df.loc["Within 50-50"] = within_train_test(config, train_size=0.5)
    print("\nWorking on within 67-33")
    df.loc["Within 67-33"] = within_train_test(config, train_size=0.67)
    print("\nWorking on within loocv")
    df.loc["Within LOOCV"] = within_loocv(config)
    mean, std = df.mean(axis=1), df.std(axis=1)
    df["Mean"] = mean
    df["SD"] = std

    if save:
        df.to_csv(f"{RESULTS_PATH}/overall/{config['data_set']}.csv", index_label="Data Split")  
    return df


def compile_online_results(save=False):
    config = create_config({"data_set": "evaluation"})
    users = get_users(config)
    df = pd.DataFrame(columns=users)

    print("Working on between users transfer")
    df.loc["Baseline Transfer"], df.loc["EA Transfer"] = between_classification(config, transfer_data_set="training", online=True)
    print("\nWorking on within 67-33")
    df.loc["Within 67-33"] = within_train_test(config, online_sim=True)
    mean, std = df.mean(axis=1), df.std(axis=1)
    df["Mean"] = mean
    df["SD"] = std

    if save:
        df.to_csv(f"{RESULTS_PATH}/overall/online.csv", index_label="Data Split")  
    return df


if __name__ == "__main__":
    for data_set in ["training", "evaluation", "benchmark"]:    
        print(f"Working on {data_set} data")
        config = create_config({"data_set": data_set})
        compile_results(config, save=True)
    compile_online_results(save=True)
