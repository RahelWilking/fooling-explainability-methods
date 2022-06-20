import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

filenames = [
    "../../compas_anchors_param_optimization_20210909.csv",
    "../../cc_anchors_param_optimization_20210825.csv",
    "../../german_anchors_param_optimization_20210816.csv"
]

# load in dataframe from parameter optimization
collected_dfs = []

for filename in filenames:
    df = pd.read_csv(filename, sep=";")
    df = df[["fooled_heuristic", "val_percent", "conversion_method"]]
    df["conversion_method"] = df["conversion_method"].replace(0,"Methode 1").replace(1,"Methode 2")
    if "compas" in filename:
        df["dataset"] = "compas"
    elif "cc" in filename:
        df["dataset"] = "cc"
    else:
        df["dataset"] = "german"
    collected_dfs.append(df)

collected_dfs = pd.concat(collected_dfs)

sns.set_theme(style="ticks", palette="muted")

fig = sns.boxplot(x="dataset", y="fooled_heuristic", hue="val_percent",
            data=collected_dfs)
plt.xlabel("Datensatz")
plt.ylabel("Fooled-Heuristik")
fig.get_legend().set_title("Anteil Validierungsdaten")

plt.tight_layout()

plt.savefig("plot_parameter_effect_val_percent.png", bbox_inches="tight")
plt.close()

fig = sns.boxplot(x="dataset", y="fooled_heuristic", hue="conversion_method",
            data=collected_dfs)
plt.xlabel("Datensatz")
plt.ylabel("Fooled-Heuristik")
fig.get_legend().set_title("Anchors-Transformation")

plt.tight_layout()

plt.savefig("plot_parameter_effect_conversion_method.png", bbox_inches="tight")
