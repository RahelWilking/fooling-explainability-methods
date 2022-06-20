"""
* Methods to create graphs for f1 accuracy on perturbation task graphs.
"""
import pandas as pd
import numpy as np
from scipy.stats import binned_statistic

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from matplotlib import pyplot as plt
import seaborn as sns

filenames = ["../../threshold_results_anchors_20210913.csv",
			 "../../threshold_results_lore_20210923.csv",
			 "../../threshold_results_explan_20210918.csv"]

rets = []
names = ["anchors", "lore", "explan"]
for filename in filenames:
	df = pd.read_csv(filename, index_col=0)
	f1s, fsts, scnds, thrds = [], [], [], []

	for trial in np.unique(df['trial']):
		relevant_runs = df[df.trial == trial]

		yhat = relevant_runs['yhat']
		y = relevant_runs['y']

		# need to flip classes (we interpret 0 as ood in code but refer to it as 1 in paper)
		yhat = 1 - yhat
		y = 1 - y

		pct_first = relevant_runs['pct_occur_first'].values[0]
		pct_second = relevant_runs['pct_occur_second'].values[0]
		pct_third = relevant_runs['pct_occur_third'].values[0]

		f1 = f1_score(y, yhat)

		f1s.append(f1)
		fsts.append(pct_first)
		scnds.append(pct_second)
		thrds.append(pct_third)

	f1s, fsts = zip(*sorted(zip(f1s,fsts)))

	n_bins = 15
	ret = binned_statistic(f1s, [f1s,fsts], "mean", bins=n_bins)
	rets.append(ret)

sns.set_theme(style="ticks", palette="muted")

ax = plt.axes()
plt.ylim(-.05,1.05)
plt.xlim(0,1)
fig = plt.gcf()
fig.set_size_inches(10, 2)
plt.xlabel("F1-Score of the OOD-Classifier")
plt.ylabel("Percent of Explanations\nwith race at Rank 1")

#sns.scatterplot(f1s, fsts, ax=ax)
colors = ["r","g","b"]
for idx,ret in enumerate(rets):
	plt.plot(ret.statistic[0], ret.statistic[1], color=colors[idx])
plt.legend(names)
plt.savefig("00_plot_thresholds.eps", bbox_inches="tight", dpi=1200)
