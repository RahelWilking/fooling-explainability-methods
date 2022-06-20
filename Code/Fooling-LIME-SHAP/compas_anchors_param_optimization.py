def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from adversarial_models import * 
from utils import *
from get_data import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import datetime

import numpy as np
from numpy.random import SeedSequence
import pandas as pd

from copy import deepcopy

from setup_experiments import convert_anchor_explanation, compute_scores, identify_pareto

import multiprocessing_on_dill as multiprocessing

# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")
np.random.seed(params.seed)
X, y, cols = get_and_preprocess_compas_data_modified(params)

# add unrelated columns, setup
X['unrelated_column_one'] = np.random.choice([0,1],size=X.shape[0])
X['unrelated_column_two'] = np.random.choice([0,1],size=X.shape[0])
features = [c for c in X]

race_indc = features.index('race')
unrelated_indcs = features.index('unrelated_column_one')
unrelated_indcs1 = features.index('unrelated_column_two')

X = X.values

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.1)
ss = StandardScaler().fit(xtrain)
xtrain = ss.transform(xtrain)
xtest = ss.transform(xtest)


###
## The models f and psi for COMPAS.  We discriminate based on race for f and concider two RANDOMLY DRAWN features to display in psi
#

# the biased model
class racist_model_f:
	# Decision rule: classify negatively if race is black
	def predict(self, X):
		return np.array([params.negative_outcome if x[race_indc] > 0 else params.positive_outcome for x in X])

	def predict_proba(self, X):
		return one_hot_encode(self.predict(X))

	def score(self, X, y):
		return np.sum(self.predict(X) == y) / len(X)


# the display model with one unrelated feature
class innocuous_model_psi:
	# Decision rule: classify according to randomly drawn column 'unrelated column'
	def predict_proba(self, X):
		return one_hot_encode(
			np.array([params.negative_outcome if x[unrelated_indcs] > 0 else params.positive_outcome for x in X]))

#
##


def experiment_main():

	testing = False

	categorical_features = [features.index('unrelated_column_one'),features.index('unrelated_column_two'), features.index('c_charge_degree'), features.index('two_year_recid'), features.index('race'), features.index("sex")]

	limit = xtest.shape[0]
	if testing:
		limit = min(5, limit)

	# compute label encoder (index-based, for constructors)
	constructor_label_encoder = dict()
	for idx in categorical_features:
		le = LabelEncoder()
		le.fit(xtrain[:, idx])
		constructor_label_encoder[idx] = le

	newXtrain = np.copy(xtrain)
	# Data needs to be label_encoded, since the one_hot_encoder cant handle negative values in the old version it is used at
	# (negative values happen due to the scaling of the data)
	label_encoder = dict()
	# compute categorical names
	categorical_names = dict()
	for idx in categorical_features:
		le = LabelEncoder()
		newXtrain[:, idx] = le.fit_transform(newXtrain[:, idx])
		label_encoder[idx] = le
		categorical_names[idx] = le.classes_

	newXtest = np.copy(xtest)
	for idx in categorical_features:
		le = label_encoder[idx]
		newXtest[:, idx] = le.transform(newXtest[:, idx])

	print('---------------------')
	print("Beginning Anchors COMPAS parameter optimization....")
	print("Start:", datetime.datetime.now())
	print('---------------------')

	#perturbation_multiplier_anchors = 30 # N+
	#n_samples_per_tuple = 5 # 1,...,perturbation_multiplier
	#val_percent = 0.1 # [0,1]
	#p = 0.5 # [0,1]
	#conversion_method = 0 # 0,1

	configurations = []
	summaries = []
	scores = []
	model_metas = []

	for perturbation_multiplier_anchors in [1,1,1,1,1]:
		for n_samples_per_tuple in [1,1,1,1,1]:
			for p in [0.8,0.8,0.8,0.8]:
				adv_anchors = Adversarial_Anchors_Model(racist_model_f(), innocuous_model_psi(),
														constructor_label_encoder, p=p).train(newXtrain,
																							  ytrain,
																							  categorical_features=categorical_features,
																							  feature_names=features,
																							  perturbation_multiplier=perturbation_multiplier_anchors,
																							  n_samples_per_tuple=n_samples_per_tuple,
																							  )
				pred = adv_anchors.perturbation_identifier.predict(xtest)
				pred_probs = adv_anchors.perturbation_identifier.predict_proba(xtest)
				perturbation_preds = (pred_probs[:, 1] >= 0.5)
				print("current config:", perturbation_multiplier_anchors, n_samples_per_tuple, p)
				print("number of test-instances predicted to be real", sum(pred), sum(perturbation_preds), "of", xtest.shape[0])

				for val_percent in [0.1,0.2,0.3]:
					anchors_xtrain, anchors_xval, anchors_ytrain, anchors_yval = train_test_split(newXtrain, ytrain,
																								  test_size=val_percent)

					adv_anchors_explainer = anchor_tabular.AnchorTabularExplainer(np.unique(ytrain), features,
																				  newXtrain,
																				  categorical_names.copy())  # fitting later modifies therefore give copy here
					adv_anchors_explainer.fit(anchors_xtrain, anchors_ytrain, anchors_xval, anchors_yval)

					for conversion_method in [0, 1]:
						print(len(configurations))
						def anchors_exp(i):
							res = convert_anchor_explanation(
								adv_anchors_explainer.explain_instance(newXtest[i], adv_anchors.predict), features,
								conversion_method)
							#if i % 50 == 0:
							#	print(datetime.datetime.now(), i)
							return res

						explanations = list(pool.map(anchors_exp, range(limit)))

						model_metas.append((sum(pred), sum(perturbation_preds), adv_anchors.replication_rate))
						configurations.append(
							(perturbation_multiplier_anchors, n_samples_per_tuple, p, val_percent, conversion_method))
						summary = experiment_summary(explanations, features)
						summaries.append([summary])
						fidelity = round(adv_anchors.fidelity(xtest), 2)
						fooled_heuristic = compute_scores(summary, features[race_indc], [features[unrelated_indcs]])
						scores.append((1-fidelity,fooled_heuristic,1-accuracy_score(*adv_anchors.ood_training_task_ability)))


	print("Ende:", datetime.datetime.now())

	# save results (all, pareto can be computed from this again)
	headers = ["perturbation_multiplier", "n_samples_per_tuple", "p", "val_percent", "conversion_method", "fidelity_error", "fooled_heuristic", "ood_error", "summary", "number_preds", "number_perturbation_preds", "replication_rate"]
	df = pd.DataFrame(data=np.hstack((configurations, scores, summaries, model_metas)), columns=headers)
	#print(df)
	filename = "compas_anchors_variation.csv"
	df.to_csv(filename, sep=";", index=False)

	print("Results on Pareto-Front:")
	pareto_front = identify_pareto(np.array(scores)[:,:-1])
	for idx in pareto_front:
		configuration = configurations[idx]
		summary = summaries[idx]
		scores_of_run = scores[idx]
		# Display Results
		print("perturbation_multiplier_anchors:", configuration[0])
		print("n_samples_per_tuple:", configuration[1])
		print("p:", configuration[2])
		print("val_percent:", configuration[3])
		print("conversion_method:", configuration[4])
		print("---")
		print("Anchors Ranks and Pct Occurances for one unrelated feature:")
		print(summary[0])
		print("Fidelity:", 1 - scores_of_run[0])
		print("Fooled-Heuristic:", scores_of_run[1])
		print("OOD-Accuracy:", 1 - scores_of_run[2])
		print("OOD-Accuracy-test:", model_metas[idx][0]/xtest.shape[0], model_metas[idx][1]/xtest.shape[0])
		print("--------------------------------------")


def seed_process(seed_seq):
	seed = seed_seq.generate_state(1)
	np.random.seed(seed)

if __name__ == "__main__":
	pool = multiprocessing.Pool()
	ss = SeedSequence(params.seed)

	process_seeds = ss.spawn(12)

	pool.map(seed_process, process_seeds)
	experiment_main()
