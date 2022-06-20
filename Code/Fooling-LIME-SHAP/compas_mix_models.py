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
X, y, cols = get_and_preprocess_compas_data_modified(params)

# add unrelated columns, setup
X['unrelated_column_one'] = np.random.choice([0, 1], size=X.shape[0])
X['unrelated_column_two'] = np.random.choice([0, 1], size=X.shape[0])
features = [c for c in X]

race_indc = features.index('race')
unrelated_indcs = features.index('unrelated_column_one')
unrelated_indcs1 = features.index('unrelated_column_two')

X = X.values

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1)
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


# the display model with two unrelated features
class innocuous_model_psi_two:
    def predict_proba(self, X):
        A = np.where(X[:, unrelated_indcs] > 0, params.positive_outcome, params.negative_outcome)
        B = np.where(X[:, unrelated_indcs1] > 0, params.positive_outcome, params.negative_outcome)
        preds = np.logical_xor(A, B).astype(int)
        return one_hot_encode(preds)


#
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

    categorical_features = [features.index('unrelated_column_one'),features.index('unrelated_column_two'), features.index('c_charge_degree'), features.index('two_year_recid'), features.index('race'), features.index("sex")]

    results = []
    headers = ["model_adversary", "model_explainer", "blackbox_version", "fidelity", "fooled_heuristic", "ood_accuracy", "summary"]

    # get adv_model + explain method from every explainer for both psi
    adv_models_one = dict()
    explain_functions_one = dict()
    adv_models_two = dict()
    explain_functions_two = dict()
    for model in models:
        # get adv_model + explain method and add them to dict
        if model == "anchors":
            adv_model, explain_func = get_adv_model_and_exp_func(model, "compas", xtrain, ytrain, features,
                                                                racist_model_f(),
                                                                innocuous_model_psi(),
                                                                categorical_features=categorical_features)
        else:
            adv_model, explain_func = get_adv_model_and_exp_func(model, "compas", xtrain, ytrain, features, racist_model_f(),
                                                             innocuous_model_psi(),
                                                             categorical_features=categorical_features, pool=pool)
        adv_models_one[model] = adv_model
        explain_functions_one[model] = explain_func
        if model == "anchors":
            adv_model, explain_func = get_adv_model_and_exp_func(model, "compas", xtrain, ytrain, features,
                                                                racist_model_f(),
                                                                innocuous_model_psi_two(),
                                                                categorical_features=categorical_features)
        else:
            adv_model, explain_func = get_adv_model_and_exp_func(model, "compas", xtrain, ytrain, features, racist_model_f(),
                                                             innocuous_model_psi_two(),
                                                             categorical_features=categorical_features, pool=pool)
        adv_models_two[model] = adv_model
        explain_functions_two[model] = explain_func

    for model_adv in models:
        for model_exp in models:
            if model_exp == model_adv:
                continue
            print('---------------------')
            print("Beginning " + display_names[model_exp] + " explanations on " + display_names[model_adv] + " attack COMPAS Experiments....")
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
            fooled_heuristic = compute_scores(summary, features[race_indc], [features[unrelated_indcs]])
            results.append((model_adv, model_exp, "one_unrelated_features", fidelity, fooled_heuristic,
                            accuracy_score(*adv_model.ood_training_task_ability), summary))


            # run experiment with psi two
            adv_model = adv_models_two[model_adv]
            explain_func = explain_functions_two[model_exp]

            def model_explain(i):
                tmp = explain_func(xtest[i], adv_model)
                if i % 50 == 0:
                    print(datetime.datetime.now(), i)
                return tmp

            explanations = list(pool.map(model_explain, range(limit)))

            # Display Results
            print(display_names[model_exp] + " explanations on " + display_names[model_adv] + " attack"
                  + " Ranks and Pct Occurances (1 corresponds to most important feature) for two unrelated features:")
            summary = experiment_summary(explanations, features)
            print(summary)
            fidelity = round(adv_model.fidelity(xtest), 2)
            print("Fidelity:", fidelity)
            fooled_heuristic = compute_scores(summary, features[race_indc], [features[unrelated_indcs], features[unrelated_indcs1]])
            results.append((model_adv, model_exp, "two_unrelated_features", fidelity, fooled_heuristic,
                            accuracy_score(*adv_model.ood_training_task_ability), summary))

    df = pd.DataFrame(data=results, columns=headers)
    filename = "compas_mix_models.csv"
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
