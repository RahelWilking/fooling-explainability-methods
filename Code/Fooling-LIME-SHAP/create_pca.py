from adversarial_models import * 
from utils import *
from get_data import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

import lime
import lime.lime_tabular
import shap

from copy import deepcopy


params = Params("model_configurations/experiment_params.json")
X, y, cols = get_and_preprocess_compas_data(params)
features = [c for c in X]
np.random.seed(params.seed)
race_indc = features.index('race')

X = X.values
before_scaling = X.copy()
c_cols = [features.index('c_charge_degree_F'), features.index('c_charge_degree_M'), features.index('two_year_recid'), features.index('race'), features.index("sex_Male"), features.index("sex_Female")]

#X = np.delete(X, c_cols, axis=1)

ss = StandardScaler().fit(X)
X = ss.transform(X)


r = []
for _ in range(1):
	p = np.random.normal(0,1,size=X.shape)

	# for row in p:
	#	for c in c_cols:
	#		row[c] = np.random.choice(X[:,c])

	X_p = X + p
	#X_p[c_cols] = X[c_cols]
	r.append(X_p)

r = np.vstack(r)
p = [1 for _ in range(len(r))]
iid = [0 for _ in range(len(X))]

all_x = np.vstack((r,X))
all_y = np.array(p + iid)

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2) 
results = pca.fit_transform(all_x)

print (len(X))
indexes = np.where(before_scaling[:,features.index('c_charge_degree_F')])
print(indexes)
indexes = indexes[0] + len(X)
print(indexes)
test_X = results[indexes,:]
print(before_scaling.shape,results.shape, test_X.shape)

plt.scatter(results[:500,0], results[:500,1], alpha=.2)
plt.scatter(results[-500:,0], results[-500:,1], alpha=.2)
plt.scatter(test_X[:500,0],test_X[:500,1],alpha=.2)
plt.show()



