import itertools
import numpy as np
from random import sample
import sys
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()
RESULTS_PATH = os.environ["RESULTS_PATH"]

src_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(src_dir)

from pipeline.utilities import create_config, train_model, test_model, save_model
from pipeline.preprocessing import preprocess_recording, get_users
from pipeline.transfer_learning import get_align_mat, align
from scipy.stats import ttest_1samp, ttest_rel

def exclude_idx(lis, idx):
    return lis[:idx] + lis[idx+1:]


def process_data(config):
    users = get_users(config)
    X_all, X_all_aligned, y_all = [], [], []

    # Align data
    for user in users:
        X, y = preprocess_recording(user, config)
        align_mat = get_align_mat(X)
        X_aligned = align(X, align_mat)
        X_all.append(X)
        X_all_aligned.append(X_aligned)
        y_all.append(y)
    return X_all, X_all_aligned, y_all, users

def train_full_models(config, title=None):
        X_all, X_all_aligned, y_all, _ = process_data(config)

        y_combined = list(itertools.chain(*y_all)) 
        X_combined = np.concatenate(X_all, axis=0)
        X_combined_aligned = np.concatenate(X_all_aligned, axis=0)

        # Train
        base_model = train_model(X_combined, y_combined, config)
        tf_model = train_model(X_combined_aligned, y_combined, config)
        if not title:
            title = config['data_set']

        save_model(base_model, config, f"{title}_base")
        save_model(tf_model, config, f"{title}_tf")


def between_classification(config, transfer_data_set="within", save=False):
        X_all, X_all_aligned, y_all, users = process_data(config)

        print("Transfer source:", transfer_data_set)
        # Train models for between data sets transfer
        if not transfer_data_set == "within":
            assert transfer_data_set in ["training", "evaluation", "benchmark"]
            transfer_config = config.copy()
            transfer_config["data_set"] = transfer_data_set
            src_X_all, src_X_all_aligned, src_y_all, _ = process_data(transfer_config)

            src_y_combined = list(itertools.chain(*src_y_all)) 
            src_X_combined = np.concatenate(src_X_all, axis=0)
            src_X_combined_aligned = np.concatenate(src_X_all_aligned, axis=0)

            # Train
            base_model = train_model(src_X_combined, src_y_combined, config)
            tf_model = train_model(src_X_combined_aligned, src_y_combined, config)            

        base_accs, tf_accs = [], []
        for u, user in enumerate(users):

            # Train models for within data set, between users transfer
            if transfer_data_set == "within":
                # remove user from training data
                y_combined = list(itertools.chain(*exclude_idx(y_all, u))) 
                X_combined = np.concatenate(exclude_idx(X_all, u), axis=0)
                X_combined_aligned = np.concatenate(exclude_idx(X_all_aligned, u), axis=0)

                # Train
                base_model = train_model(X_combined, y_combined, config)
                tf_model = train_model(X_combined_aligned, y_combined, config)
                
                if save:
                    save_model(base_model, config, f"{user}_base")
                    save_model(tf_model, config, f"{user}_tf")

            # Test
            base_acc = test_model(X_all[u], y_all[u], base_model)
            tf_acc = test_model(X_all_aligned[u], y_all[u], tf_model)
            base_accs.append(base_acc)
            tf_accs.append(tf_acc)
            
            print(f"{user}: base acc = {base_acc:.3f}, tf_acc = {tf_acc:.3f}")
        
        # T-tests
        print(ttest_1samp(tf_accs, popmean=0.3333, alternative="greater"))
        print(ttest_rel(base_accs, tf_accs, alternative="less"))

        print(f"Overall base_acc = {np.mean(base_accs):.3f}, tf_acc = {np.mean(tf_accs):.3f}")
        return base_accs, tf_accs


def online_simulation(config, title, align_X=True, oversample=1, save=False):
    X_all, X_all_aligned, y_all, users = process_data(config)
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
    if save:
        with open(f"{RESULTS_PATH}/transfer_learning/{title}.npy", 'wb') as f:
            np.save(f, results)


if __name__ == "__main__":
    config = create_config({"data_set": "evaluation"})
    #train_full_models(config, title="retrain")
    between_classification(config, transfer_data_set="benchmark"), 
    #online_simulation(config, title, align_X=False)
     