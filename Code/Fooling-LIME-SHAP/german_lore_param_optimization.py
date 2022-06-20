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

sys.path.insert(1, "../EXPLAN/LORE")
import lore

from setup_experiments import compute_scores, identify_pareto
from pyyadt_feature_importance import mdi_gain_ratio

import multiprocessing_on_dill as multiprocessing

# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")
np.random.seed(params.seed)
X, y, cols = get_and_preprocess_german(params)

features = [c for c in X]

gender_indc = features.index('Gender')
loan_rate_indc = features.index('LoanRateAsPercentOfIncome')

X = X.values

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1)
ss = StandardScaler().fit(xtrain)
xtrain = ss.transform(xtrain)
xtest = ss.transform(xtest)

mean_lrpi = np.mean(xtrain[:, loan_rate_indc])


###
## The models f and psi for GERMAN.  We discriminate based on gender for f and consider loan rate % income for explanation
#

# the biased model 
class racist_model_f:
    # Decision rule: classify negative outcome if female
    def predict(self,X):
        return np.array([params.positive_outcome if x[gender_indc] > 0 else params.negative_outcome for x in X])

    def predict_proba(self, X):
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)

# the display model with one unrelated feature
class innocuous_model_psi:
    # Decision rule: classify according to loan rate indc
    def predict_proba(self, X): 
        return one_hot_encode(np.array([params.negative_outcome if x[loan_rate_indc] > mean_lrpi else params.positive_outcome for x in X]))

##
###


def experiment_main():

	testing = False

	categorical_features = ['Gender', 'ForeignWorker', 'Single', 'HasTelephone', 'CheckingAccountBalance_geq_0',
				   'CheckingAccountBalance_geq_200', 'SavingsAccountBalance_geq_100', 'SavingsAccountBalance_geq_500',
				   'MissedPayments', 'NoCurrentLoan', 'CriticalAccountOrLoansElsewhere', 'OtherLoansAtBank',
				   'OtherLoansAtStore', 'HasCoapplicant', 'HasGuarantor', 'OwnsHouse', 'RentsHouse', 'Unemployed',
				   'YearsAtCurrentJob_lt_1', 'YearsAtCurrentJob_geq_4', 'JobClassIsSkilled']
	categorical_features = [features.index(c) for c in categorical_features]

	limit = xtest.shape[0]
	if testing:
		limit = min(5, limit)

	# compute label encoder (index-based, for constructors)
	constructor_label_encoder = dict()
	for idx in categorical_features:
		le = LabelEncoder()
		le.fit(xtrain[:, idx])
		constructor_label_encoder[idx] = le

	# construct meta-information of dataset
	name = "temp"  # only used for temporary tree files
	feature_names = features[:]
	class_name = "target"
	columns = [class_name] + feature_names
	possible_outcomes = list(np.unique(ytrain))
	discrete = [feature_names[x] for x in categorical_features]
	continuous = []
	dfXy = pd.DataFrame(data=np.concatenate((ytrain.reshape(-1, 1), xtrain), axis=1), columns=columns)
	type_features, features_type = recognize_features_type(dfXy, class_name)
	discrete, continuous = set_discrete_continuous(columns, type_features, class_name, discrete=discrete,
												   continuous=None)
	idx_features = {i: col for i, col in enumerate(feature_names)}
	df_le, label_encoder = label_encode(dfXy, discrete)

	# need to use encoded data or work around later decoding somehow...
	newytrain = df_le.pop(class_name).values
	newXtrain = df_le.values  # after pop

	# dataset in LORE style with only the neccessary field set
	dataset = {
		'name': name,
		'columns': list(columns),
		'class_name': class_name,
		'features_type': features_type,
		'discrete': discrete,
		'continuous': continuous,
		'idx_features': idx_features,
		'label_encoder': label_encoder,
		'possible_outcomes': possible_outcomes
	}

	newXtest = np.copy(xtest)
	for idx in categorical_features:
		le = label_encoder[features[idx]]
		newXtest[:, idx] = le.transform(newXtest[:, idx])

	print('---------------------')
	print("Beginning Lore GERMAN parameter optimization....")
	print("Start:", datetime.datetime.now())
	print('---------------------')

	#perturbation_multiplier_lore = 30

	# potentially add feature_importance_measure?

	configurations = []
	summaries = []
	scores = []
	model_metas = []

	for perturbation_multiplier_lore in [1,2,3,4,5,6,7,8,9,10]:
		adv_lore = Adversarial_LORE_Model(racist_model_f(), innocuous_model_psi(), constructor_label_encoder).train(xtrain, ytrain,
																								   categorical_features=categorical_features,
																								   feature_names=features,
																								   perturbation_multiplier=perturbation_multiplier_lore,
																								   pool=pool)

		pred = adv_lore.perturbation_identifier.predict(xtest)
		pred_probs = adv_lore.perturbation_identifier.predict_proba(xtest)
		perturbation_preds = (pred_probs[:, 1] >= 0.5)
		print("current config:", perturbation_multiplier_lore)
		print("number of test-instances predicted to be real", sum(pred), sum(perturbation_preds), "of", xtest.shape[0])

		print(len(configurations))
		def lore_exp(i):
			trainplusinstance = np.vstack((newXtest[i], newXtrain))

			explanation, infos = lore.explain(0, trainplusinstance, dataset, adv_lore,
											  discrete_use_probabilities=True,
											  returns_infos=True,
											  path="tmp" + str(datetime.datetime.now().timestamp())+str(multiprocessing.current_process().pid))
			dt = infos["dt"]
			neighborhood_data = infos["dfZ"]
			bb_labels = infos["y_pred_bb"]
			if i % 50 == 0:
				print(datetime.datetime.now(), i)
			return mdi_gain_ratio(dt, neighborhood_data, bb_labels, features, discrete, features_type)

		explanations = list(pool.map(lore_exp, range(limit)))

		model_metas.append((sum(pred), sum(perturbation_preds), adv_lore.replication_rate))
		configurations.append(
			[perturbation_multiplier_lore])
		summary = experiment_summary(explanations, features)
		summaries.append([summary])
		fidelity = round(adv_lore.fidelity(xtest), 2)
		fooled_heuristic = compute_scores(summary, features[gender_indc], [features[loan_rate_indc]])
		scores.append((1-fidelity,fooled_heuristic,1-accuracy_score(*adv_lore.ood_training_task_ability)))

	print("Ende:", datetime.datetime.now())

	# save results (all, pareto can be computed from this again)
	headers = ["perturbation_multiplier", "fidelity_error", "fooled_heuristic", "ood_error", "summary", "number_preds", "number_perturbation_preds", "replication_rate"]
	df = pd.DataFrame(data=np.hstack((configurations, scores, summaries, model_metas)), columns=headers)
	#print(df)
	filename = "german_lore_param_optimization.csv"
	df.to_csv(filename, sep=";", index=False)

	print("Results on Pareto-Front:")
	pareto_front = identify_pareto(np.array(scores)[:,:-1])
	for idx in pareto_front:
		configuration = configurations[idx]
		summary = summaries[idx]
		scores_of_run = scores[idx]
		# Display Results
		print("perturbation_multiplier_lore:", configuration[0])
		print("---")
		print("Lore Ranks and Pct Occurances for one unrelated feature:")
		print(summary[0])
		print("Fidelity:", 1 - scores_of_run[0])
		print("Fooled-Heuristic:", scores_of_run[1])
		print("OOD-Accuracy:", 1 - scores_of_run[2])
		print("OOD-Accuracy-test:", model_metas[idx][0] / xtest.shape[0], model_metas[idx][1] / xtest.shape[0])
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
