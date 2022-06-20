import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

import ast

####### current version: flipped axes ########

#filename = "../../compas_experiment_20210909.csv"
#filename = "../../cc_experiment_20210909.csv"
filename = "../../german_experiment_20210910.csv"

#sensitive_feature = "race"
#sensitive_feature = "racePctWhite numeric"
sensitive_feature = "Gender"

if "german" in filename:
    unrelated_feature = "LoanRateAsPercentOfIncome"
    unrelated_feature_two = None
else:
    unrelated_feature = "unrelated_column_one"
    unrelated_feature_two = "unrelated_column_two"

# i need the data in three different ways, rightmost grey part must be summed total
# then unrelated feature 2 must be summed total
# then unrelated feature 1 must be summed total
#

df = pd.read_csv(filename, sep=";")

# parse out summary
new_df = df[["model", "blackbox_version"]]

summaries = df["summary"].to_list()

plot_values = []
for idx,summary in enumerate(summaries):
    summary_dict = ast.literal_eval(summary)
    for key in summary_dict:
        sens_val = 0
        unrelated_val = 0
        unrelated_val2 = 0
        total_val = 0
        for feature,value in summary_dict[key]:
            if feature == "Nothing shown":
                continue
            total_val += value*100
            if feature == sensitive_feature:
                sens_val = value*100
            if feature == unrelated_feature:
                unrelated_val = value*100
            if feature == unrelated_feature_two:
                unrelated_val2 = value*100
        plot_values.append((idx,str(key),sens_val,sens_val+unrelated_val,sens_val+unrelated_val+unrelated_val2,total_val))

plot_df = pd.DataFrame(plot_values, columns=["idx", "place", "sens_val", "unrelated_val_cum", "unrelated_val2_cum", "total_val"])

result_df = new_df.merge(plot_df, left_index=True, right_on="idx")

#print(result_df)

name_mapping_bb = {
    "baseline" : "Original f",
    "one_unrelated_feature" : "Attack\none Feature",
    "two_unrelated_features" : "Attack\ntwo Features"
}

name_mapping_method = {
        "anchors": "Anchors",
        "explan": "EXPLAN",
        "lore": "LORE",
        "lime": "LIME",
        "shap": "KernelSHAP"
    }

n_cols = 3
cols = ["baseline", "one_unrelated_feature", "two_unrelated_features"]

if "german" in filename:
    n_cols = 2
    cols = ["baseline", "one_unrelated_feature"]

rows = ["lime", "shap", "anchors", "lore", "explan"]

sns.set_theme(style="ticks", palette="muted")
fig, axs = plt.subplots(ncols=5, nrows=n_cols)
fig.set_size_inches(15, 2 * n_cols)
sns.set_context(rc = {'patch.linewidth': 0.0})

for col_idx,method in enumerate(rows):
    for row_idx,blackbox_version in enumerate(cols):
        current_df = result_df.loc[(result_df["model"] == method) & (result_df["blackbox_version"] == blackbox_version)]

        ax = axs[row_idx, col_idx]

        plt.sca(ax)

        plt.xlim(0,100)

        sns.barplot(x="total_val", y="place", data=current_df,
                    label="Other Features", color="lightgrey")
        if "german" in filename:
            sns.barplot(x="unrelated_val_cum", y="place", data=current_df,
                        label=unrelated_feature, color="b")
        else:
            sns.barplot(x="unrelated_val2_cum", y="place", data=current_df,
                    label=unrelated_feature_two, color="b")
            sns.barplot(x="unrelated_val_cum", y="place", data=current_df,
                        label=unrelated_feature, color="c")
        sns.barplot(x="sens_val", y="place", data=current_df,
                    label=sensitive_feature, color="r")
        ax.yaxis.set_tick_params(length=0)
        plt.xlabel("")
        if row_idx == n_cols-1:
            plt.xlabel("Percent Occurance", fontsize=15)
        if col_idx == 0:
            plt.ylabel("Rank in Expl.", fontsize=15)
        else:
            plt.ylabel("")
            ax.set_yticks([])

sns.despine(left=True)
handles, labels = ax.get_legend_handles_labels()
if "german" in filename:
    fig.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0.55, -0.2), loc='lower center', ncol=n_cols+1,fontsize=15)
else:
    fig.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0.55, -0.13), loc='lower center', ncol=n_cols+1, fontsize=15)

pad = 10  # in points

for ax, col in zip(axs[0], rows):
    ax.annotate(name_mapping_method[col], xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size=20, ha='center', va='baseline')

for ax, row in zip(axs[:,0], cols):
    ax.annotate(name_mapping_bb[row], xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=20, ha='right', va='center', ma='center', rotation=90)

fig.tight_layout()

fig.subplots_adjust(left=0.15, top=0.8, bottom=0.05)

plt.savefig("00_plot_german_experiment.eps", bbox_inches="tight", dpi=1200)
