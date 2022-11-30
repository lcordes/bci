import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from seaborn import barplot
from seaborn import heatmap
import pandas as pd
import os
import sys
from pathlib import Path

src_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(src_dir)

from analysis.overall_results import compile_results
from pipeline.utilities import create_config

from dotenv import load_dotenv
load_dotenv()

DATA_PATH = os.environ["DATA_PATH"]
RESULTS_PATH = os.environ["RESULTS_PATH"]        
TRAINING_COL = "#" + os.environ["TRAINING_COL"]
EVALUATION_COL = "#" + os.environ["EVALUATION_COL"]
BENCHMARK_COL = "#" + os.environ["BENCHMARK_COL"]
ONLINE_COL = "#" + os.environ["ONLINE_COL"]
ERROR_COL = "#" + os.environ["ERROR_COL"]


def load_results(config, overwrite=False):
    try:
        df =  pd.read_csv(f"results/overall/{config['data_set']}.csv", index_col=0)
        df_exists = True
    except:
        df_exists = False
        print(f"No results found for {config['data_set']}, compiling now")
    
    if overwrite or not df_exists:
        compile_results(config, save=True)
        df =  pd.read_csv(f"results/overall/{config['data_set']}.csv", index_col=0)
    return df

def reshape_df(df, data_set):
    df = df.drop(columns=["Mean", "SD"])
    df = df.transpose().melt()
    df = df.rename(columns={"value": "Accuracy", "Data Split": "Classifier"})
    df["Data Set"] = data_set
    return df


def overall_barplot(df, data_set, save=False):
    plt.figure()
    if data_set == "training":
        df = reshape_df(results[data_set], data_set)
        df["Classifier"] = df["Classifier"].str.replace(" ", "\n")
        ax = barplot(data=df, x="Classifier", y="Accuracy", color=TRAINING_COL, ci="sd", capsize=.1, errcolor=ERROR_COL)
    else: 
        if data_set == "benchmark":
            df1 = reshape_df(results["training"], "Training") 
            df2 = reshape_df(results["benchmark"], "Benchmark") 
            palette = [TRAINING_COL, BENCHMARK_COL]
            df = pd.concat([df1, df2])
            df["Classifier"] = df["Classifier"].str.replace(" ", "\n")
        else:
            eval_subset = results["evaluation"].drop(["Within 50-50", "Within LOOCV"])
            df1 = reshape_df(eval_subset, "Offline") 
            df2 = reshape_df(results["online"], "Online") 
            palette = [ONLINE_COL, EVALUATION_COL]
            df = pd.concat([df1, df2])

        ax = barplot(data=df, x="Classifier", y="Accuracy", hue="Data Set",
        ci="sd", errcolor=ERROR_COL, capsize=.1, palette=palette)   
    ax.tick_params(labelsize=11)
    ax.set_xlabel("Classifier",fontsize=15)
    ax.set_ylabel("Accuracy",fontsize=15)

    if save:
        plt.savefig(f"{RESULTS_PATH}/overall/{data_set}_barplot.png", dpi=250, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()
    

def overall_heatmap(results, data_set, save=False):
    df = results[data_set]
    x_size = 15 if df.shape[1] > 10 else 9
    plt.figure(figsize=(x_size, 3))
    heatmap(
        df,
        vmin=df.max().max() + 0.15,
        vmax=df.min().min() - 0.15,
        annot=True,
        fmt='.2f',
    )

    if save:
        plt.savefig(f"{RESULTS_PATH}/overall/{data_set}_heatmap.png", dpi=250, bbox_inches="tight")
        plt.clf()
    else:
        plt.show()


if __name__ == "__main__":
    results = {}
    for data_set in ["training", "evaluation", "benchmark", "online"]:    
        config = create_config({"data_set": data_set})
        results[data_set] = load_results(config)
    
    for data_set in ["training", "evaluation", "benchmark"]:   
        overall_barplot(results, data_set, save=True)        
        overall_heatmap(results, data_set, save=True)
     
