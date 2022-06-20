"""
The experiment MAIN for GERMAN.
"""
import warnings
warnings.filterwarnings('ignore') 

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

from setup_experiments import get_adv_model_and_exp_func, compute_scores

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
	"""
	Run through experiments for all models on GERMAN.
	* This may take some time given that we iterate through every point in the test set
	* We print out the rate at which features occur in the top three features
	"""

	models = [
		"anchors",
		"explan",
		"lore",
		"lime",
		"shap"
	]
	testing = False
	compute_baseline = True

	categorical_features = ['Gender', 'ForeignWorker', 'Single', 'HasTelephone', 'CheckingAccountBalance_geq_0',
				   'CheckingAccountBalance_geq_200', 'SavingsAccountBalance_geq_100', 'SavingsAccountBalance_geq_500',
				   'MissedPayments', 'NoCurrentLoan', 'CriticalAccountOrLoansElsewhere', 'OtherLoansAtBank',
				   'OtherLoansAtStore', 'HasCoapplicant', 'HasGuarantor', 'OwnsHouse', 'RentsHouse', 'Unemployed',
				   'YearsAtCurrentJob_lt_1', 'YearsAtCurrentJob_geq_4', 'JobClassIsSkilled']
	categorical_features = [features.index(c) for c in categorical_features]

	limit = xtest.shape[0]
	if testing:
		limit = min(5, limit)

	results = []
	headers = ["model", "blackbox_version", "fidelity", "fooled_heuristic", "ood_accuracy", "summary"]

	if "anchors" in models:
		print('---------------------')
		print("Beginning Anchors GERMAN Experiments....")
		print("(These take some time to run because we have to generate explanations for every point in the test set) ")
		print('---------------------')

		adv_anchors, anchors_explain = get_adv_model_and_exp_func("anchors", "german", xtrain, ytrain, features, racist_model_f(),
																  innocuous_model_psi(),
																  categorical_features=categorical_features)

		if compute_baseline:
			def anchors_exp(i):
				tmp = anchors_explain(xtest[i], racist_model_f())
				if i % 50 == 0:
					print(datetime.datetime.now(), i)
				return tmp

			explanations = list(pool.map(anchors_exp, range(limit)))

			# Display Results
			print("Anchors Ranks and Pct Occurances for original blackbox:")
			summary = experiment_summary(explanations, features)
			print(summary)
			results.append(("anchors", "baseline", np.NaN, np.NaN, np.NaN, summary))

		def anchors_exp(i):
			tmp = anchors_explain(xtest[i], adv_anchors)
			if i % 50 == 0:
				print(datetime.datetime.now(), i)
			return tmp

		explanations = list(pool.map(anchors_exp, range(limit)))

		# Display Results
		print("Anchors Ranks and Pct Occurances (1 corresponds to most important feature) for one unrelated feature:")
		summary = experiment_summary(explanations, features)
		print(summary)
		fidelity = round(adv_anchors.fidelity(xtest), 2)
		print("Fidelity:", fidelity)
		fooled_heuristic = compute_scores(summary, features[gender_indc], [features[loan_rate_indc]])
		results.append(("anchors", "one_unrelated_feature", fidelity, fooled_heuristic, accuracy_score(*adv_anchors.ood_training_task_ability), summary))

	if "explan" in models:
		print('---------------------')
		print("Beginning EXPLAN GERMAN Experiments....")
		print("(These take some time to run because we have to generate explanations for every point in the test set) ")
		print('---------------------')

		adv_explan, explan_explain = get_adv_model_and_exp_func("explan", "german", xtrain, ytrain, features, racist_model_f(),
																innocuous_model_psi(),
																categorical_features=categorical_features)
		
		if compute_baseline:
			def explan_exp(i):
				tmp = explan_explain(xtest[i], racist_model_f())
				if i % 50 == 0:
					print(datetime.datetime.now(), i)
				return tmp

			explanations = list(pool.map(explan_exp, range(limit)))

			# Display Results
			print("EXPLAN Ranks and Pct Occurances for original blackbox:")
			summary = experiment_summary(explanations, features)
			print(summary)
			results.append(("explan", "baseline", np.NaN, np.NaN, np.NaN, summary))

		def explan_exp(i):
			tmp = explan_explain(xtest[i], adv_explan)
			if i % 50 == 0:
				print(datetime.datetime.now(), i)
			return tmp

		explanations = list(pool.map(explan_exp, range(limit)))

		# Display Results
		print("EXPLAN Ranks and Pct Occurances (1 corresponds to most important feature) for one unrelated feature:")
		summary = experiment_summary(explanations, features)
		print(summary)
		fidelity = round(adv_explan.fidelity(xtest), 2)
		print("Fidelity:", fidelity)
		fooled_heuristic = compute_scores(summary, features[gender_indc], [features[loan_rate_indc]])
		results.append(("explan", "one_unrelated_feature", fidelity, fooled_heuristic,
						accuracy_score(*adv_explan.ood_training_task_ability), summary))

	if "lore" in models:
		print('---------------------')
		print("Beginning LORE GERMAN Experiments....")
		print("(These take some time to run because we have to generate explanations for every point in the test set) ")
		print('---------------------')

		adv_lore, lore_explain = get_adv_model_and_exp_func("lore", "german", xtrain, ytrain, features, racist_model_f(),
															innocuous_model_psi(),
															categorical_features=categorical_features, pool=pool)

		if compute_baseline:
			def lore_exp(i):
				tmp = lore_explain(xtest[i], racist_model_f())
				if i % 50 == 0:
					print(datetime.datetime.now(), i)
				return tmp

			explanations = list(pool.map(lore_exp, range(limit)))

			# Display Results
			print("LORE Ranks and Pct Occurances for original blackbox:")
			summary = experiment_summary(explanations, features)
			print(summary)
			results.append(("lore", "baseline", np.NaN, np.NaN, np.NaN, summary))

		def lore_exp(i):
			tmp = lore_explain(xtest[i], adv_lore)
			if i % 50 == 0:
				print(datetime.datetime.now(), i)
			return tmp

		explanations = list(pool.map(lore_exp, range(limit)))

		# Display Results
		print("LORE Ranks and Pct Occurances (1 corresponds to most important feature) for one unrelated feature:")
		summary = experiment_summary(explanations, features)
		print(summary)
		fidelity = round(adv_lore.fidelity(xtest), 2)
		print("Fidelity:", fidelity)
		fooled_heuristic = compute_scores(summary, features[gender_indc], [features[loan_rate_indc]])
		results.append(("lore", "one_unrelated_feature", fidelity, fooled_heuristic,
						accuracy_score(*adv_lore.ood_training_task_ability), summary))

	if "lime" in models:
		print('---------------------')
		print("Beginning LIME GERMAN Experiments....")
		print("(These take some time to run because we have to generate explanations for every point in the test set) ")
		print('---------------------')

		adv_lime, lime_explain = get_adv_model_and_exp_func("lime", "german", xtrain, ytrain, features, racist_model_f(),
															innocuous_model_psi(),
															categorical_features=categorical_features)
		
		if compute_baseline:
			explanations = []
			for i in range(limit):
				explanations.append(lime_explain(xtest[i], racist_model_f()))
				if i % 50 == 0:
					print(datetime.datetime.now(), i)
			
			# Display Results
			print("LIME Ranks and Pct Occurances for original blackbox:")
			summary = experiment_summary(explanations, features)
			print(summary)
			results.append(("lime", "baseline", np.NaN, np.NaN, np.NaN, summary))

		explanations = []
		for i in range(limit):
			explanations.append(lime_explain(xtest[i], adv_lime))
			if i % 50 == 0:
				print(datetime.datetime.now(), i)

		# Display Results
		print("LIME Ranks and Pct Occurances (1 corresponds to most important feature) for one unrelated feature:")
		summary = experiment_summary(explanations, features)
		print(summary)
		fidelity = round(adv_lime.fidelity(xtest), 2)
		print("Fidelity:", fidelity)
		fooled_heuristic = compute_scores(summary, features[gender_indc], [features[loan_rate_indc]])
		results.append(("lime", "one_unrelated_feature", fidelity, fooled_heuristic,
						accuracy_score(*adv_lime.ood_training_task_ability), summary))

	if "shap" in models:
		print('---------------------')
		print('Beginning SHAP GERMAN Experiments....')
		print('---------------------')

		adv_shap, shap_explain = get_adv_model_and_exp_func("shap", "german", xtrain, ytrain, features, racist_model_f(),
															innocuous_model_psi(),
															categorical_features=categorical_features)

		if compute_baseline:
			explanations = []
			for i in range(limit):
				explanations.append(shap_explain(xtest[i], racist_model_f()))
				if i % 50 == 0:
					print(datetime.datetime.now(), i)

			# Display Results
			print("SHAP Ranks and Pct Occurances for original blackbox:")
			summary = experiment_summary(explanations, features)
			print(summary)
			results.append(("shap", "baseline", np.NaN, np.NaN, np.NaN, summary))

		explanations = []
		for i in range(limit):
			explanations.append(shap_explain(xtest[i], adv_shap))
			if i % 50 == 0:
				print(datetime.datetime.now(), i)

		print("SHAP Ranks and Pct Occurances one unrelated features:")
		summary = experiment_summary(explanations, features)
		print(summary)
		fidelity = round(adv_shap.fidelity(xtest), 2)
		print("Fidelity:", fidelity)
		fooled_heuristic = compute_scores(summary, features[gender_indc], [features[loan_rate_indc]])
		results.append(("shap", "one_unrelated_feature", fidelity, fooled_heuristic,
						accuracy_score(*adv_shap.ood_training_task_ability), summary))

	df = pd.DataFrame(data=results, columns=headers)
	filename = "german_experiment.csv"
	df.to_csv(filename, sep=";", index=False)

def seed_process(seed_seq):
	seed = seed_seq.generate_state(1)
	np.random.seed(seed)

if __name__ == "__main__":
	pool = multiprocessing.Pool()
	ss = SeedSequence(params.seed)

	process_seeds = ss.spawn(12)

	pool.map(seed_process, process_seeds)
	experiment_main()
