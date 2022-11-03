import matplotlib.pyplot as plt
from seaborn import barplot
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
RESULTS_PATH = os.environ["RESULTS_PATH"]        
        

save = True

train_dat = pd.read_csv("results/overall/Classification accuracy per user and classifier (training data).csv", index_col=0)
eval_dat = pd.read_csv("results/overall/Classification accuracy per user and classifier (evaluation data).csv", index_col=0)
benchmark_dat = pd.read_csv("results/overall/Classification accuracy per user and classifier (benchmark data).csv", index_col=0)
eval_online = pd.read_csv("results/overall/manual.csv")
train_col = "#0A2E5E"
eval_online_col = "#4C91EC"
eval_offline_col = "#40E0D0"
benchmark_col = "#808080"
sd_col = "#FF5733"

# Train barplot
train_dat = train_dat.drop(columns=["Mean", "SD"])
train_dat = train_dat.transpose().melt()
train_dat = train_dat.rename(columns={"value": "Accuracy", "Data Split": "Classifier"})
barplot(data=train_dat, x="Classifier", y="Accuracy", color=train_col, ci="sd", capsize=.1, errcolor=sd_col)
plt.title("Mean user accuracy per classifier")
if save:
    plt.savefig("barplot_training.png", dpi=400)
    plt.clf()
else:
    plt.show()

# Benchmark barplot
benchmark_dat = benchmark_dat.drop(columns=["Mean", "SD"])
benchmark_dat = benchmark_dat.transpose().melt()
benchmark_dat = benchmark_dat.rename(columns={"value": "Accuracy", "Data Split": "Classifier"})

benchmark_dat["Data Set"] = "Benchmark"
train_dat["Data Set"] = "Training"
combined_dat = pd.concat([train_dat, benchmark_dat])


barplot(data=combined_dat, x="Classifier", y="Accuracy", hue="Data Set",
 ci="sd", errcolor=sd_col, capsize=.1, palette=[train_col, benchmark_col])
plt.title("Mean user accuracy per classifier")
if save:
    plt.savefig("barplot_benchmark.png", dpi=400)
    plt.clf()
else:
    plt.show()

# Evaluation barplot
eval_dat = eval_dat.drop(columns=["Mean", "SD"])
eval_dat = eval_dat.transpose()
eval_dat = eval_dat.drop(["Within LOOCV", "Within 50-50"], axis=1)
eval_dat = eval_dat.melt()
eval_dat = eval_dat.rename(columns={"value": "Accuracy", "Data Split": "Classifier"})

eval_dat["Computation"] = "Offline"
eval_online["Computation"] = "Online"
combined_dat = pd.concat([eval_online, eval_dat])

barplot(data=combined_dat, x="Classifier", y="Accuracy", hue="Computation", palette=[eval_online_col, eval_offline_col], ci="sd", capsize=.1, errcolor=sd_col)
plt.title("Mean user accuracy per classifier")
if save:
    plt.savefig("barplot_evaluation.png", dpi=400)
    plt.clf()
else:
    plt.show()