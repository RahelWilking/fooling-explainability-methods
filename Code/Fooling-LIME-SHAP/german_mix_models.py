import warnings

warnings.filterwarnings('ignore')

from utils import *
from get_data import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import numpy as np
from numpy.random import SeedSequence
import pandas as pd

import datetime

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
    def predict(self, X):
        return np.array([params.positive_outcome if x[gender_indc] > 0 else params.negative_outcome for x in X])

    def predict_proba(self, X):
        return one_hot_encode(self.predict(X))

    def score(self, X, y):
        return np.sum(self.predict(X) == y) / len(X)


# the display model with one unrelated feature
class innocuous_model_psi:
    # Decision rule: classify according to loan rate indc
    def predict_proba(self, X):
        return one_hot_encode(np.array(
            [params.negative_outcome if x[loan_rate_indc] > mean_lrpi else params.positive_outcome for x in X]))


##
###

def experiment_main():

    display_names = {
        "anchors": "Anchors",
        "explan": "EXPLAN",
        "lore": "LORE",
        "lime": "LIME",
        "shap": "SHAP"
    }

    models = [
         "anchors",
         "explan",
         "lore",
         "lime",
         "shap"
    ]

    testing = False

    limit = xtest.shape[0]
    if testing:
        limit = min(5, limit)

    categorical_features = ['Gender', 'ForeignWorker', 'Single', 'HasTelephone', 'CheckingAccountBalance_geq_0',
                            'CheckingAccountBalance_geq_200', 'SavingsAccountBalance_geq_100',
                            'SavingsAccountBalance_geq_500',
                            'MissedPayments', 'NoCurrentLoan', 'CriticalAccountOrLoansElsewhere', 'OtherLoansAtBank',
                            'OtherLoansAtStore', 'HasCoapplicant', 'HasGuarantor', 'OwnsHouse', 'RentsHouse',
                            'Unemployed',
                            'YearsAtCurrentJob_lt_1', 'YearsAtCurrentJob_geq_4', 'JobClassIsSkilled']
    categorical_features = [features.index(c) for c in categorical_features]

    results = []
    headers = ["model_adversary", "model_explainer", "blackbox_version", "fidelity", "fooled_heuristic", "ood_accuracy", "summary"]

    # get adv_model + explain method from every explainer for both psi
    adv_models_one = dict()
    explain_functions_one = dict()
    for model in models:
        # get adv_model + explain method and add them to dict
        if model == "anchors":
            adv_model, explain_func = get_adv_model_and_exp_func(model, "german", xtrain, ytrain, features,
                                                                racist_model_f(),
                                                                innocuous_model_psi(),
                                                                categorical_features=categorical_features)
        else:
            adv_model, explain_func = get_adv_model_and_exp_func(model, "german", xtrain, ytrain, features, racist_model_f(),
                                                             innocuous_model_psi(),
                                                             categorical_features=categorical_features, pool=pool)
        adv_models_one[model] = adv_model
        explain_functions_one[model] = explain_func

    for model_adv in models:
        for model_exp in models:
            if model_exp == model_adv:
                continue
            print('---------------------')
            print("Beginning " + display_names[model_exp] + " explanations on " + display_names[model_adv] + " attack German Experiments....")
            print(
                "(These take some time to run because we have to generate explanations for every point in the test set) ")
            print('---------------------')
            # run experiment with psi
            adv_model = adv_models_one[model_adv]
            explain_func = explain_functions_one[model_exp]

            def model_explain(i):
                tmp = explain_func(xtest[i], adv_model)
                if i % 50 == 0:
                    print(datetime.datetime.now(), i)
                return tmp

            explanations = list(pool.map(model_explain, range(limit)))

            # Display Results
            print(display_names[model_exp] + " explanations on " + display_names[model_adv] + " attack"
                  + " Ranks and Pct Occurances (1 corresponds to most important feature) for one unrelated feature:")
            summary = experiment_summary(explanations, features)
            print(summary)
            fidelity = round(adv_model.fidelity(xtest), 2)
            print("Fidelity:", fidelity)
            fooled_heuristic = compute_scores(summary, features[gender_indc], [features[loan_rate_indc]])
            results.append((model_adv, model_exp, "one_unrelated_features", fidelity, fooled_heuristic,
                            accuracy_score(*adv_model.ood_training_task_ability), summary))

    df = pd.DataFrame(data=results, columns=headers)
    filename = "german_mix_models.csv"
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
