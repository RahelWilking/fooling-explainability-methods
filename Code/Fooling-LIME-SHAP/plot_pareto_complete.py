import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from setup_experiments import identify_pareto

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

sns.set_theme(style="ticks", palette="muted")

fig, axs = plt.subplots(ncols=3, nrows=3)
fig.set_size_inches(15,15)

# load in dataframe from parameter optimization
collected_dfs = []
chosen_idxs = [2,2,10,7,1,3,2,0,2]
cols = ["compas", "cc", "german"]
rows = ["anchors", "lore", "explan"]

for idx, filename in enumerate(filenames):
    df = pd.read_csv(filename, sep=";")

    # in case of anchors: first filter to use only the worst combination over the forth and fifth parameter, other metadata not representative afterwards!
    if "anchors" in filename:
        original_df = df.copy()
        res = df.groupby(np.arange(len(df.index)) // 6)[['fooled_heuristic']].agg(['max'])
        df = df.iloc[::6]
        df.reset_index(inplace=True)
        df["fooled_heuristic"] = res

    # only use the two pareto-scores to compute pareto-front indexes
    pareto_idx = identify_pareto(df[["fidelity_error", "fooled_heuristic"]].values)

    # plot scores for pareto indexes
    ax = axs[idx // 3, idx % 3]
    plt.sca(ax)

    plt.xlabel("Fidelity Fehlerrate")
    plt.ylabel("Fooled-Heuristik")

    sns.lineplot(df["fidelity_error"][pareto_idx], df["fooled_heuristic"][pareto_idx], ax=ax, marker="o")
    plt.scatter(df["fidelity_error"][chosen_idxs[idx]], df["fooled_heuristic"][chosen_idxs[idx]], color="r", zorder=10)

pad = 10  # in points

for ax, col in zip(axs[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size=20, ha='center', va='baseline')

for ax, row in zip(axs[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=20, ha='right', va='center', rotation=90)

fig.tight_layout()

fig.subplots_adjust(left=0.15, top=0.8)

plt.savefig("pareto_complete.png", bbox_inches="tight")
