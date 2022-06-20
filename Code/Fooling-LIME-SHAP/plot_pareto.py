import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from setup_experiments import identify_pareto

# load in dataframe from parameter optimization
#filename = "../../german_anchors_param_optimization_20210816.csv"
#filename = "../../cc_anchors_param_optimization_20210825.csv"
filename = "../../compas_anchors_param_optimization_20210909.csv"
#filename = "../../german_lore_param_optimization_20210829.csv"
#filename = "../../cc_lore_param_optimization_20210901.csv"
#filename = "../../compas_lore_param_optimization_20210831.csv"
#filename = "../../german_explan_param_optimization_20210901.csv"
#filename = "../../cc_explan_param_optimization_20210901.csv"
#filename = "../../compas_explan_param_optimization_20210902.csv"

df = pd.read_csv(filename, sep=";")

# print(df.head(5))

# in case of anchors: first filter to use only the worst combination over the forth and fifth parameter, other metadata not representative afterwards!
if "anchors" in filename:
    original_df = df.copy()
    res = df.groupby(np.arange(len(df.index)) // 6)[['fooled_heuristic']].agg(['max'])
    df = df.iloc[::6]
    df.reset_index(inplace=True)
    df["fooled_heuristic"] = res

# only use the two pareto-scores to compute pareto-front indexes
pareto_idx = identify_pareto(df[["fidelity_error", "fooled_heuristic"]].values)

print(pareto_idx)

# plot scores for pareto indexes
ax = plt.axes()
#plt.ylim(0.01, 1)
#plt.xlim(0,1)
#ax.set(yscale="log")

plt.xlabel("Fidelity Error")
plt.ylabel("Fooled Heuristic")

sns.lineplot(df["fidelity_error"][pareto_idx], df["fooled_heuristic"][pareto_idx], ax=ax, style=1, markers=True)
#sns.scatterplot(df["fidelity_error"][pareto_idx], df["fooled_heuristic"][pareto_idx], ax=ax)
for i in range(len(pareto_idx)):
    plt.text(x=df["fidelity_error"][pareto_idx[i]]*1.01 ,y=df["fooled_heuristic"][pareto_idx[i]]*1.01,s=pareto_idx[i])
plt.savefig("pareto_compas_anchors.png")
