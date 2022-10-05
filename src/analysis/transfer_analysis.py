
import itertools
import numpy as np
from scipy.linalg import inv, sqrtm
from natsort import natsorted
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from seaborn import heatmap

import sys
from pathlib import Path
import os
from dotenv import load_dotenv
from random import sample
load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
RESULTS_PATH = os.environ["RESULTS_PATH"]

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)

from classification.train_test_model import create_config
from data_acquisition.preprocessing import preprocess_openbci as preprocess_training
from analysis.benchmark_analysis import preprocess_openbci as preprocess_benchmark
from feature_extraction.extractors import CSPExtractor
from classification.classifiers import CLASSIFIERS
from transfer_learning.euclidean_alignment import get_align_mat, align

def train_model(X, y, config):
    extractor = CSPExtractor(config)
    X_transformed = extractor.fit_transform(X, y)
    classifier = CLASSIFIERS[config["model_type"]](config)
    classifier.fit(X_transformed, y)
    return extractor, classifier


def test_model(X, y, model):
    extractor, predictor = model
    X_transformed = extractor.transform(X)
    acc = predictor.score(X_transformed, y)
    return acc


def process_data_set(data_set, config):
    if data_set == "training":
        dir = Path(f"{DATA_PATH}/recordings/training_data_collection")
        users = natsorted([path.stem for path in dir.glob("*.hdf5")])
        preprocess_recording = preprocess_training

    elif data_set == "benchmark":
        dir = Path(f"{DATA_PATH}/recordings/competition_IV_2a")
        users = natsorted([path.stem for path in dir.glob("*.gdf") if path.stem[-1] == "T"])
        preprocess_recording = preprocess_benchmark
    X_all, X_all_aligned, y_all = [], [], []

    # Align data
    print("Aligning data...")
    for user in users:
        X, y = preprocess_recording(user, config)
        align_mat = get_align_mat(X)
        X_aligned = align(X, align_mat)
        X_all.append(X)
        X_all_aligned.append(X_aligned)
        y_all.append(y)
    return X_all, X_all_aligned, y_all, users


def exclude_idx(lis, idx):
    return lis[:idx] + lis[idx+1:]

def between_classification(data_set, config, title):
        X_all, X_all_aligned, y_all, users = process_data_set(data_set, config)

        results = np.zeros((2, len(users)+1))
        for u, user in enumerate(users):
            # remove user from training data
            y_combined = list(itertools.chain(*exclude_idx(y_all, u))) 
            X_combined = np.concatenate(exclude_idx(X_all, u), axis=0)
            X_combined_aligned = np.concatenate(exclude_idx(X_all_aligned, u), axis=0)

            # Train
            base_model = train_model(X_combined, y_combined, config)
            tf_model = train_model(X_combined_aligned, y_combined, config)
        
            # Test
            base_acc = test_model(X_all[u], y_all[u], base_model)
            tf_acc = test_model(X_all_aligned[u], y_all[u], tf_model)
            results[0, u] = base_acc
            results[1, u] = tf_acc
            
            print(f"{user}: base acc = {base_acc:.3f}, tf_acc = {tf_acc:.3f}")
        
        results[0, len(users)] = np.mean(results[0, :len(users)])
        results[1, len(users)] = np.mean(results[1, :len(users)])

        print(f"Overall base_acc = {results[0, len(users)]:.3f}, tf_acc = {results[1, len(users)]:.3f}")
        
        with open(f"{RESULTS_PATH}/transfer_learning/{title}.npy", 'wb') as f:
            np.save(f, results)
        plot_between_results(results, users, data_set, title)


def plot_between_results(results, users, data_set, title):
    x_size = 15 if data_set == "training" else 9
    plt.figure(figsize=(x_size, 2))
    heatmap(
        results,
        vmin=np.min(results) - 0.15,
        vmax=np.max(results) + 0.15,
        annot=True,
        xticklabels=users + ["avg"],
        yticklabels=["Baseline", "Transfer"]
    )
    plt.title(title)
    plt.savefig(
        f"{RESULTS_PATH}/transfer_learning/{title}.png", dpi=400, bbox_inches="tight"
    )

def online_simulation(data_set, config, title, align_X=True, oversample=1):
    X_all, X_all_aligned, y_all, users = process_data_set(data_set, config)
    n_trials = X_all[0].shape[0]
    pool_size = 40
    step_size = 4
    repetitions = 10
    train_sizes = range(step_size, pool_size+step_size, step_size)
    
    results = np.zeros((len(users), repetitions, len(train_sizes)))
    for u, user in enumerate(users):
        print(f"User {user}")
        for rep in range(repetitions):
            print(f"Repetition {rep+1}")
            pool_idx = sample(range(n_trials), k=pool_size)
            test_idx = list(set(range(n_trials)).difference(set(pool_idx)))
            for t_idx, train_size in enumerate(train_sizes):
                train_idx = pool_idx[:train_size] 
                X_train_user = np.repeat(X_all[u][train_idx, :, :], oversample, axis=0)
                align_mat = get_align_mat(X_train_user)
                X_train_aligned = align(X_train_user, align_mat) 
                y_train_user = [y_all[u][t] for t in range(n_trials) if t in train_idx] * oversample

                if align_X:
                    X_train = np.concatenate(exclude_idx(X_all_aligned, u) + [X_train_aligned], axis=0)
                    X_test = align(X_all[u][test_idx, :, :], align_mat)
                else:
                    X_train = np.concatenate(exclude_idx(X_all, u) + [X_train_user], axis=0)
                    X_test = X_all[u][test_idx, :, :]

                y_train = list(itertools.chain(*exclude_idx(y_all, u))) + y_train_user


                y_test = [y_all[u][t] for t in range(n_trials) if t in test_idx]

                model = train_model(X_train, y_train, config)
                acc = test_model(X_test, y_test, model)
                results[u, rep, t_idx] = acc

    with open(f"{RESULTS_PATH}/transfer_learning/{title}.npy", 'wb') as f:
        np.save(f, results)


def plot_online_sim_results(title):
    
        with open(f"{RESULTS_PATH}/transfer_learning/{title}.npy", 'rb') as f:
            data = np.load(f)
    
        data = np.mean(data, axis=1) # average across repetitions
        n_users = data.shape[0]
        x = list(range(4, 44, 4))

        plt.clf()
        for i in range(n_users):
            plt.plot(x, data[i, :], marker=".",  markersize=5)
        y = np.mean(data, axis=0)
        plt.plot(x, y, color="black", linestyle="dashed", linewidth=2,marker="o", markersize=6)
        plt.title(title)
        plt.gca().xaxis.set_major_locator(MultipleLocator(4))
        plt.savefig(
            f"{RESULTS_PATH}/transfer_learning/{title}.png", dpi=400, bbox_inches="tight"
        )
def plot_online_sim_comparison(base_results, tf_results, title):
    
        with open(f"{RESULTS_PATH}/transfer_learning/{base_results}.npy", 'rb') as f:
            base_data = np.load(f)
        with open(f"{RESULTS_PATH}/transfer_learning/{tf_results}.npy", 'rb') as f:
            tf_data = np.load(f)
        
        base_data = np.mean(base_data, axis=1)
        tf_data = np.mean(tf_data, axis=1) 
        x = list(range(4, 44, 4))

        n_users = base_data.shape[0]
        
        # for u in range(n_users):
        #     plt.clf()
        #     plt.plot(x, tf_data[u, :], marker="o", color="C1",  markersize=5, label="transfer")
        #     plt.plot(x, base_data[u, :], linestyle="dashed", marker="o", color="C7", markersize=5, label="baseline")
        #     plt.ylim([0.2, 0.85])
        #     plt.title(f"Online simulation for u{u+1}")
        #     plt.legend()
        #     plt.gca().xaxis.set_major_locator(MultipleLocator(4))
        #     plt.show()

        plt.clf()
        plt.plot(x, np.mean(tf_data, axis=0), marker="o", color="C1",  markersize=5, label="transfer")
        plt.plot(x, np.mean(base_data, axis=0), linestyle="dashed", marker="o", color="C7", markersize=5, label="baseline")
        plt.ylim([0.2, 0.85])
        plt.title(title)
        plt.legend()
        plt.gca().xaxis.set_major_locator(MultipleLocator(4))
        plt.savefig(
            f"{RESULTS_PATH}/transfer_learning/{title}.png", dpi=400, bbox_inches="tight"
        )


if __name__ == "__main__":
    data_set = "benchmark"
    config = create_config(clf_specific={"shrinkage": "auto", "solver": "eigen"}, bandpass=(8, 30))
    title = "Between classification (benchmark data, 8-30)"
    between_classification(data_set, config, title)
    #online_simulation(data_set, config, title, align_X=False)
    

    # dir = Path(f"{RESULTS_PATH}/transfer_learning")
    # results =[path.stem for path in dir.glob("*.npy")]  
    # for result in results:
    #      plot_online_sim_results(result)



    # plot_online_sim_comparison(
    #     "Online simulation (benchmark data, 8-30, no alignment)",
    #     "Online simulation (benchmark data, 8-30)",
    #     title="Online simulation averaged across users (benchmark data)")