import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from setup_experiments import kl_divergence

filenames = [
    "../../compas_anchors_param_optimization_20210909.csv",
    "../../cc_anchors_param_optimization_20210825.csv",
    "../../german_anchors_param_optimization_20210816.csv",
    "../../compas_lore_param_optimization_20210831.csv",
    "../../cc_lore_param_optimization_20210901.csv",
    "../../german_lore_param_optimization_20210829.csv",
    "../../compas_explan_param_optimization_20210902.csv",
    "../../cc_explan_param_optimization_20210901.csv",
    "../../german_explan_param_optimization_20210901.csv",
]

# load in dataframe from parameter optimization
collected_dfs = []

for filename in filenames:
    df = pd.read_csv(filename, sep=";")
    if "anchors" in filename:
        df = df[["replication_rate", "p"]]
        df["method"] = "anchors"
    elif "lore" in filename:
        df = df[["replication_rate"]]
        df["method"] = "lore"
    else:
        df = df[["replication_rate"]]
        df["method"] = "explan"
    if "compas" in filename:
        df["dataset"] = "compas"
    elif "cc" in filename:
        df["dataset"] = "cc"
    else:
        df["dataset"] = "german"
    collected_dfs.append(df)

sns.set_theme(style="ticks", palette="muted")

sns.boxplot(x="dataset", y="replication_rate", hue="p",
            data=pd.concat(collected_dfs[:3]))
plt.xlabel("Datensatz")
plt.ylabel("Replikationsrate")

plt.tight_layout()

plt.savefig("plot_replication_rate_anchors.png", bbox_inches="tight")
plt.close()

collected_dfs = pd.concat(collected_dfs)

fig = sns.boxplot(x="dataset", y="replication_rate", hue="method",
            data=collected_dfs)

fig.get_legend().set_title("Methode")

plt.xlabel("Datensatz")
plt.ylabel("Replikationsrate")

plt.tight_layout()

plt.savefig("plot_replication_rate_complete.png", bbox_inches="tight")


