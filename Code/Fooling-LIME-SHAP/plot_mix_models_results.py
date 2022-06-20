import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

import ast

filename = "../../compas_mix_models_20210913.csv"
#filename = "../../cc_mix_models_20210912.csv"
#filename = "../../german_mix_models_20210910.csv"

filename2 = "../../compas_experiment_20210909.csv"
#filename2 = "../../cc_experiment_20210909.csv"
#filename2 = "../../german_experiment_20210910.csv"

sensitive_feature = "race"
#sensitive_feature = "racePctWhite numeric"
#sensitive_feature = "Gender"

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
diagonal_df = pd.read_csv(filename2, sep=";")
diagonal_df = diagonal_df.loc[diagonal_df["blackbox_version"] != "baseline"]
diagonal_df = diagonal_df.replace("one_unrelated_feature", "one_unrelated_features")

# parse out summary
new_df = df[["model_adversary", "model_explainer", "blackbox_version"]]
new_diagonal_df = pd.DataFrame()
new_diagonal_df["model_adversary"] = diagonal_df["model"]
new_diagonal_df["model_explainer"] = diagonal_df["model"]
new_diagonal_df["blackbox_version"] = diagonal_df["blackbox_version"]

new_df = pd.concat([new_df,new_diagonal_df])
new_df = new_df.reset_index()

summaries = df["summary"].to_list() + diagonal_df["summary"].to_list()

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

result_df = result_df.loc[result_df["place"] == "1"]
#print(result_df)

name_mapping_method = {
        "anchors": "Anchors",
        "explan": "EXPLAN",
        "lore": "LORE",
        "lime": "LIME",
        "shap": "KernelSHAP"
    }

cols = ["lime", "shap", "anchors", "lore", "explan"]
rows = ["lime", "shap", "anchors", "lore", "explan"]
n_legend = 3

if not ("german" in filename):
    n_legend += 1

save_name = "00_plot_compas_{}_mix_models.eps"
sns.set_theme(style="ticks", palette="muted")
sns.set_context(rc = {'patch.linewidth': 0.0})

# plot for blackbox_version = one_unrelated_features
blackbox_version = "one_unrelated_features"

fig, axs = plt.subplots(ncols=5, nrows=1)
fig.set_size_inches(10, 2)

for col_idx, adversary in enumerate(cols):
    ax = axs[col_idx]
    plt.sca(ax)

    plt.xlim(0, 100)

    current_df = result_df.loc[(result_df["model_adversary"] == adversary) & (result_df["blackbox_version"] == blackbox_version)]
    #current_df = result_df.loc[(result_df["model_explainer"] == explainer) & (result_df["model_adversary"] == adversary) & (result_df["blackbox_version"] == blackbox_version)]
    #print(current_df)

    sns.barplot(x="total_val", y="model_explainer", data=current_df,
                label="Other Features", color="lightgrey", order=rows)
    if "german" in filename:
        sns.barplot(x="unrelated_val_cum", y="model_explainer", data=current_df,
                    label=unrelated_feature, color="b", order=rows)
    else:
        sns.barplot(x="unrelated_val2_cum", y="model_explainer", data=current_df,
                label=unrelated_feature_two, color="b", order=rows)
        sns.barplot(x="unrelated_val_cum", y="model_explainer", data=current_df,
                    label=unrelated_feature, color="c", order=rows)
    sns.barplot(x="sens_val", y="model_explainer", data=current_df,
                label=sensitive_feature, color="r", order=rows)

    ax.yaxis.set_tick_params(length=0)
    plt.xlabel("")
    #if row_idx == 4:
    plt.xlabel("Percent Occurance")
    ax.set_yticklabels([name_mapping_method[name]+"  " for name in rows], fontsize=15)
    if col_idx == 0:
        plt.ylabel("")
    else:
        plt.ylabel("")
        ax.set_yticks([])

sns.despine(left=True)
handles, labels = ax.get_legend_handles_labels()
fig.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0.55, -0.37), loc='lower center', ncol=n_legend)

pad = 10  # in points

for ax, col in zip(axs, cols):
    ax.annotate(name_mapping_method[col], xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size=15, ha='center', va='baseline')

#for ax, row in zip(axs[:,0], rows):
#    ax.annotate(name_mapping_method[row], xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
#                xycoords=ax.yaxis.label, textcoords='offset points',
#                size=15, ha='right', va='center', rotation=90)

axs[2].annotate("Targeted Method", xy=(0.5, 1), xytext=(0, pad+30), xycoords='axes fraction', textcoords='offset points',
                size=20, ha='center', va='baseline')
ax = axs[0]
ax.annotate("Explaining Method", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=20, ha='right', va='center', rotation=90)


fig.tight_layout()

fig.subplots_adjust(left=0.15, top=0.8, bottom=0.05, wspace=0.15)

plt.savefig(save_name.format(blackbox_version), bbox_inches="tight", dpi=1200)

plt.close()

if not ("german" in filename):
    # do the same with psi_two
    blackbox_version = "two_unrelated_features"

    fig, axs = plt.subplots(ncols=5, nrows=1)
    fig.set_size_inches(10, 2)

    for col_idx, adversary in enumerate(cols):
        ax = axs[col_idx]
        plt.sca(ax)

        plt.xlim(0, 100)

        current_df = result_df.loc[
            (result_df["model_adversary"] == adversary) & (result_df["blackbox_version"] == blackbox_version)]
        # current_df = result_df.loc[(result_df["model_explainer"] == explainer) & (result_df["model_adversary"] == adversary) & (result_df["blackbox_version"] == blackbox_version)]
        # print(current_df)

        sns.barplot(x="total_val", y="model_explainer", data=current_df,
                    label="Other Features", color="lightgrey", order=rows)
        sns.barplot(x="unrelated_val2_cum", y="model_explainer", data=current_df,
                    label=unrelated_feature_two, color="b", order=rows)
        sns.barplot(x="unrelated_val_cum", y="model_explainer", data=current_df,
                    label=unrelated_feature, color="c", order=rows)
        sns.barplot(x="sens_val", y="model_explainer", data=current_df,
                    label=sensitive_feature, color="r", order=rows)

        ax.yaxis.set_tick_params(length=0)
        plt.xlabel("")
        # if row_idx == 4:
        plt.xlabel("Percent Occurance")
        ax.set_yticklabels([name_mapping_method[name]+"  " for name in cols], fontsize=15)
        if col_idx == 0:
            plt.ylabel("")
        else:
            plt.ylabel("")
            ax.set_yticks([])

    sns.despine(left=True)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0.55, -0.37), loc='lower center', ncol=n_legend)

    pad = 10  # in points

    for ax, col in zip(axs, cols):
        ax.annotate(name_mapping_method[col], xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size=15, ha='center', va='baseline')

    # for ax, row in zip(axs[:,0], rows):
    #    ax.annotate(name_mapping_method[row], xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
    #                xycoords=ax.yaxis.label, textcoords='offset points',
    #                size=15, ha='right', va='center', rotation=90)

    axs[2].annotate("Targeted Method", xy=(0.5, 1), xytext=(0, pad + 30), xycoords='axes fraction',
                    textcoords='offset points',
                    size=20, ha='center', va='baseline')
    ax = axs[0]
    ax.annotate("Explaining Method", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=20, ha='right', va='center', rotation=90)

    fig.tight_layout()

    fig.subplots_adjust(left=0.15, top=0.8, bottom=0.05, wspace=0.15)

    plt.savefig(save_name.format(blackbox_version), bbox_inches="tight", dpi=1200)

    plt.close()


