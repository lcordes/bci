import itertools
import numpy as np
from random import sample
import sys
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()
from random import sample
RESULTS_PATH = os.environ["RESULTS_PATH"]

src_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(src_dir)

from pipeline.utilities import create_config, train_model, test_model, save_model
from pipeline.preprocessing import preprocess_recording, get_users, get_not_railed_channels, get_eval_user_id
from pipeline.transfer_learning import get_align_mat, align
from scipy.stats import ttest_1samp, ttest_rel

def exclude_idx(lis, idx):
    return lis[:idx] + lis[idx+1:]


def process_data(config):
    users = get_users(config)

    config_copy = config.copy()
    config_copy["discard_railed"] = False
    X_all, X_all_aligned, y_all = [], [], []

    # Align data
    for user in users:
        X, y = preprocess_recording(user, config_copy)
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



def single_user_between(user, u, X_all, X_all_aligned, y_all, transfer_data_set, config):

    if transfer_data_set == "within":
        # Remove user from training data
        X_combined = np.concatenate(exclude_idx(X_all, u), axis=0)
        X_combined_aligned = np.concatenate(exclude_idx(X_all_aligned, u), axis=0)
        y_combined = list(itertools.chain(*exclude_idx(y_all, u))) 
    else:
        # Get training data from transfer set
        assert transfer_data_set in ["training", "evaluation", "benchmark"]
        transfer_config = config.copy()
        transfer_config["data_set"] = transfer_data_set
        src_X_all, src_X_all_aligned, src_y_all, _ = process_data(transfer_config)

        X_combined = np.concatenate(src_X_all, axis=0)
        X_combined_aligned = np.concatenate(src_X_all_aligned, axis=0)
        y_combined = list(itertools.chain(*src_y_all)) 

    if config["discard_railed"]:
        # Remove railed channels
        not_railed_idx, _ = get_not_railed_channels(user, config)
        X_combined = X_combined[:, not_railed_idx, :]
        X_combined_aligned = X_combined_aligned[:, not_railed_idx, :]
        X_test = X_all[u][:, not_railed_idx, :]
        X_test_aligned = X_all_aligned[u][:, not_railed_idx, :]
    else: 
        X_test = X_all[u]
        X_test_aligned = X_all_aligned[u]
    

    ### Delete, for online reconstruction
    block1_idx = list(range(30))
    block2_idx = list(range(30, 60))
    user_id = int(get_eval_user_id(user, config))


    if user_id % 2 == 0:   
        X_test = X_test[block2_idx, :]
        y_test = y_all[u][block2_idx]
        y_test_aligned = y_all[u][block1_idx]
    else:
        X_test = X_test[block1_idx, :]
        y_test = y_all[u][block1_idx]
        y_test_aligned = y_all[u][block2_idx]
    
    cal_config = config.copy()
    cal_config["discard_railed"] = False
    X_calibration, _ = preprocess_recording(user, cal_config, calibration=True)
    if config["discard_railed"]:
        X_calibration = X_calibration[:, not_railed_idx, :]
    align_mat = get_align_mat(X_calibration)
    X_test_aligned = align(X_test, align_mat)
    ####

    # Train
    base_model = train_model(X_combined, y_combined, config)
    tf_model = train_model(X_combined_aligned, y_combined, config)
    
    # Test
    base_acc = test_model(X_test, y_test, base_model)
    tf_acc = test_model(X_test_aligned, y_test_aligned, tf_model)
    
    if X_combined.shape[1] < len(config["channels"]):
        print(f"{user}: {X_combined.shape[1]} channels")

    return base_acc, tf_acc


def between_classification(config, transfer_data_set="within"):
        
        X_all, X_all_aligned, y_all, users = process_data(config)
        print("Transfer source:", transfer_data_set)        
        base_accs, tf_accs = [], []

        for u, user in enumerate(users):
            base_acc, tf_acc = single_user_between(user, u, X_all, X_all_aligned, y_all, transfer_data_set, config)
            base_accs.append(base_acc)
            tf_accs.append(tf_acc)
            print(f"{user}: base acc = {base_acc:.3f}, tf_acc = {tf_acc:.3f}")
        
        # T-tests
        print("", ttest_1samp(tf_accs, popmean=0.3333, alternative="greater"))
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
    config = create_config({"data_set": "training"})
    #train_full_models(config, title="retrain")
    between_classification(config, transfer_data_set="within"), 
    #online_simulation(config, title, align_X=False)
     