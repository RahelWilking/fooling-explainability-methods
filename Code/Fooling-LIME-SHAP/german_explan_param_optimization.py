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
from util import *
sys.path.insert(1, "../EXPLAN")
import explan

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
	discrete = [feature_names[x] for x in categorical_features]
	continuous = []
	dfXy = pd.DataFrame(data=np.concatenate((ytrain.reshape(-1, 1), xtrain), axis=1), columns=columns)
	type_features, features_type = recognize_features_type(dfXy, class_name)
	discrete, continuous = set_discrete_continuous(columns, type_features, class_name, discrete=discrete,
												   continuous=None)
	df_le, label_encoder = label_encode(dfXy, discrete)

	# need to use encoded data or work around later decoding somehow...
	newytrain = df_le.pop(class_name).values
	newXtrain = df_le.values  # after pop

	discrete_indices = list()
	for idx, col in enumerate(feature_names):
		if col == class_name or col in continuous:
			continue
		discrete_indices.append(idx)

	dataset = {
		'X': newXtrain,  # TODO: currently uses only xtrain, but original Explan uses full dataset
		'discrete_indices': discrete_indices,
		'name': name,
		'class_name': class_name,
		'columns': columns,
		'discrete': discrete,
		'continuous': continuous,
		'features_type': features_type,
		'label_encoder': label_encoder
	}

	newXtest = np.copy(xtest)
	for idx in categorical_features:
		le = label_encoder[features[idx]]
		newXtest[:, idx] = le.transform(newXtest[:, idx])

	print('---------------------')
	print("Beginning Explan GERMAN parameter optimization....")
	print("Start:", datetime.datetime.now())
	print('---------------------')

	#perturbation_multiplier_explan = 30
	#n_samples = 3000
	#tau = 250

	# potentially add feature_importance_measure?

	configurations = []
	summaries = []
	scores = []
	model_metas = []

	for perturbation_multiplier_explan in [1,2,3,4,5,6,7,8,9,10]:
		n_samples = 3000
		tau = 250
		adv_explan = Adversarial_EXPLAN_Model(racist_model_f(), innocuous_model_psi(), constructor_label_encoder).train(xtrain,
																									   ytrain,
																									   categorical_features=categorical_features,
																									   feature_names=features,
																									   perturbation_multiplier=perturbation_multiplier_explan,
																									   n_samples=n_samples,
																									   tau=tau,
																									   pool=pool)

		pred = adv_explan.perturbation_identifier.predict(xtest)
		pred_probs = adv_explan.perturbation_identifier.predict_proba(xtest)
		perturbation_preds = (pred_probs[:, 1] >= 0.5)
		print("current config:", perturbation_multiplier_explan)
		print("number of test-instances predicted to be real", sum(pred), sum(perturbation_preds), "of", xtest.shape[0])

		print(len(configurations))
		def explan_exp(i):
			exp_EXPLAN, info_EXPLAN = explan.Explainer(newXtest[i],
													   adv_explan,
													   dataset) # could give n_samples and tau here...but should probably be default?
			dt = info_EXPLAN["C"]
			neighborhood_data = info_EXPLAN["dfX"]
			bb_labels = info_EXPLAN["y_X_bb"]
			if i % 50 == 0:
				print(datetime.datetime.now(), i)
			return mdi_gain_ratio(dt, neighborhood_data, bb_labels, features, discrete,
                                  features_type)

		explanations = list(pool.map(explan_exp, range(limit)))

		model_metas.append((sum(pred), sum(perturbation_preds), adv_explan.replication_rate))
		configurations.append(
			[perturbation_multiplier_explan])
		summary = experiment_summary(explanations, features)
		summaries.append([summary])
		fidelity = round(adv_explan.fidelity(xtest), 2)
		fooled_heuristic = compute_scores(summary, features[gender_indc], [features[loan_rate_indc]])
		scores.append((1-fidelity,fooled_heuristic,1-accuracy_score(*adv_explan.ood_training_task_ability)))

	print("Ende:", datetime.datetime.now())

	# save results (all, pareto can be computed from this again)
	headers = ["perturbation_multiplier", "fidelity_error", "fooled_heuristic", "ood_error", "summary", "number_preds", "number_perturbation_preds", "replication_rate"]
	df = pd.DataFrame(data=np.hstack((configurations, scores, summaries, model_metas)), columns=headers)
	#print(df)
	filename = "german_explan_param_optimization.csv"
	df.to_csv(filename, sep=";", index=False)

	print("Results on Pareto-Front:")
	pareto_front = identify_pareto(np.array(scores)[:,:-1])
	for idx in pareto_front:
		configuration = configurations[idx]
		summary = summaries[idx]
		scores_of_run = scores[idx]
		# Display Results
		print("perturbation_multiplier_explan:", configuration[0])
		print("---")
		print("Explan Ranks and Pct Occurances for one unrelated feature:")
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
