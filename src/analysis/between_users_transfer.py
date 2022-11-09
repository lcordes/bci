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


def get_transfer_set_data(data_set, channels="all"):
    if channels == "all":
        transfer_config = create_config({"data_set": data_set, "discard_railed": False, "n_classes": 2})
    else:
        transfer_config = create_config({"data_set": data_set, "discard_railed": False, "channels": channels, "n_classes": 2})

    users = get_users(transfer_config)

    X_all, X_all_aligned, y_all = [], [], []

    # Align data
    for user in users:
        X, y = preprocess_recording(user, transfer_config)
        align_mat = get_align_mat(X)
        X_aligned = align(X, align_mat)
        X_all.append(X)
        X_all_aligned.append(X_aligned)
        y_all.append(y)
    return X_all, X_all_aligned, y_all


def between_classification(config, transfer_data_set, online=False):
        
        X_all, X_all_aligned, y_all = get_transfer_set_data(transfer_data_set)
        print("Transfer source:", transfer_data_set)        
        baseline_accs, ea_accs = [], []

        users = get_users(config)
        for u, user in enumerate(users):
            baseline_acc, ea_acc = single_user_between(user, u, X_all, X_all_aligned, y_all, transfer_data_set, config, online)
            baseline_accs.append(baseline_acc)
            ea_accs.append(ea_acc)
            print(f"{user}: baseline acc = {baseline_acc:.3f}, ea_acc = {ea_acc:.3f}")
        
        # T-tests
        print("", ttest_1samp(ea_accs, popmean=0.3333, alternative="greater"))
        print(ttest_rel(baseline_accs, ea_accs, alternative="less"))

        print(f"Overall baseline_acc = {np.mean(baseline_accs):.3f}, ea_acc = {np.mean(ea_accs):.3f}")
        return baseline_accs, ea_accs

def single_user_between(user, u, X_all, X_all_aligned, y_all, transfer_data_set, config, online):

    # Re-align tranfer data if test user has reailed channels
    user_channels = get_not_railed_channels(user, config)
    if len(user_channels) < len(config["channels"]):
        print(f"Re-aligning training data for {user}")
        X_all, X_all_aligned, y_all = get_transfer_set_data(transfer_data_set, user_channels)  
    
    # Remove user from transfer set when doing within transfer
    if transfer_data_set == config["data_set"]:
        X_all = exclude_idx(X_all, u)
        X_all_aligned = exclude_idx(X_all_aligned, u)
        y_all = exclude_idx(y_all, u)
 
    X_combined = np.concatenate(X_all, axis=0)
    X_combined_aligned = np.concatenate(X_all_aligned, axis=0)
    y_combined = list(itertools.chain(*y_all)) 

    # get user test data
    X_test, y_test = preprocess_recording(user, config)

    if online:
        assert config["data_set"] == "evaluation", "Online transfer only works for the evaluation data set."

        X_calibration, _ = preprocess_recording(user, config, calibration=True)
        print(X_calibration.shape)
        align_mat = get_align_mat(X_calibration)
        user_id = int(get_eval_user_id(user))

        if user_id % 2 == 0:   
            baseline_idx = list(range(30, 60))
            ea_idx = list(range(30))
        else:
            baseline_idx = list(range(30))
            ea_idx = list(range(30, 60))

        X_test_aligned = X_test[ea_idx, :]
        X_test = X_test[baseline_idx, :]

        y_test_aligned = [val for i, val in enumerate(y_test) if i in list(ea_idx)]
        y_test = [val for i, val in enumerate(y_test) if i in list(baseline_idx)]

    else:        
        y_test_aligned = y_test
        align_mat = get_align_mat(X_test)
        X_test_aligned = align(X_test, align_mat)
    
    # Train
    baseline_model = train_model(X_combined, y_combined, config)
    ea_model = train_model(X_combined_aligned, y_combined, config)
    
    # Test
    baseline_acc = test_model(X_test, y_test, baseline_model)
    ea_acc = test_model(X_test_aligned, y_test_aligned, ea_model)

    return baseline_acc, ea_acc



def train_full_models(config, title=None):
        X_all, X_all_aligned, y_all, _ = get_transfer_set_data(config)

        y_combined = list(itertools.chain(*y_all)) 
        X_combined = np.concatenate(X_all, axis=0)
        X_combined_aligned = np.concatenate(X_all_aligned, axis=0)

        # Train
        baseline_model = train_model(X_combined, y_combined, config)
        ea_model = train_model(X_combined_aligned, y_combined, config)
        if not title:
            title = config['data_set']

        save_model(baseline_model, config, f"{title}_baseline")
        save_model(ea_model, config, f"{title}_ea")


def online_simulation(config, title, align_X=True, oversample=1, save=False):
    X_all, X_all_aligned, y_all, users = get_transfer_set_data(config)
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
    config = create_config({"data_set": "benchmark", "n_classes": 2})
    #train_full_models(config, title="retrain")
    between_classification(config, transfer_data_set="benchmark", online=False) 
    #online_simulation(config, title, align_X=False)
     