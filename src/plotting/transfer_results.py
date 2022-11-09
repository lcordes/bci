import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from seaborn import heatmap
import os
from dotenv import load_dotenv
load_dotenv()
RESULTS_PATH = os.environ["RESULTS_PATH"]


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
def plot_online_sim_comparison(baseline_results, ea_results, title):
    
        with open(f"{RESULTS_PATH}/transfer_learning/{baseline_results}.npy", 'rb') as f:
            baseline_data = np.load(f)
        with open(f"{RESULTS_PATH}/transfer_learning/{ea_results}.npy", 'rb') as f:
            ea_data = np.load(f)
        
        baseline_data = np.mean(baseline_data, axis=1)
        ea_data = np.mean(ea_data, axis=1) 
        x = list(range(4, 44, 4))

        n_users = baseline_data.shape[0]

        plt.clf()
        plt.plot(x, np.mean(ea_data, axis=0), marker="o", color="C1",  markersize=5, label="transfer")
        plt.plot(x, np.mean(baseline_data, axis=0), linestyle="dashed", marker="o", color="C7", markersize=5, label="baseline")
        plt.ylim([0.2, 0.85])
        plt.title(title)
        plt.legend()
        plt.gca().xaxis.set_major_locator(MultipleLocator(4))
        plt.savefig(
            f"{RESULTS_PATH}/transfer_learning/{title}.png", dpi=400, bbox_inches="tight"
        )

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