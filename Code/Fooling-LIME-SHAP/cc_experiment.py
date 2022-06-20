"""
The experiment MAIN for Communities and Crime.
 * Run the file and the CC experiments will complete
 * See compas experiment file for more details on how to read results.
"""
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
X, y, cols = get_and_preprocess_cc(params)

# add unrelated columns, setup
X['unrelated_column_one'] = np.random.choice([0, 1], size=X.shape[0])
X['unrelated_column_two'] = np.random.choice([0, 1], size=X.shape[0])
features = [c for c in X]

race_indc = features.index('racePctWhite numeric')
unrelated_indcs = features.index('unrelated_column_one')
unrelated_indcs1 = features.index('unrelated_column_two')

X = X.values

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1)
ss = StandardScaler().fit(xtrain)
xtrain = ss.transform(xtrain)
xtest = ss.transform(xtest)

###
## The models f and psi for CC.  We discriminate based on race for f and concider two RANDOMLY DRAWN features to display in psi
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
        # Using 0.5 to make it easier to detect decision boundary on perturbation
        A = np.where(X[:, unrelated_indcs] > .5, params.positive_outcome, params.negative_outcome)
        B = np.where(X[:, unrelated_indcs1] < -.5, params.positive_outcome, params.negative_outcome)
        preds = np.logical_xor(A, B).astype(int)
        return one_hot_encode(preds)


#
##
###

def experiment_main():
    """
    Run through experiments for chosen methods on CC using both one and two unrelated features.
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

    limit = xtest.shape[0]
    if testing:
        limit = min(5,limit)

    categorical_features = [features.index('unrelated_column_one'), features.index('unrelated_column_two')]

    results = []
    headers = ["model", "blackbox_version", "fidelity", "fooled_heuristic", "ood_accuracy", "summary"]

    if "anchors" in models:
        print('---------------------')
        print("Beginning Anchors CC Experiments....")
        print("(These take some time to run because we have to generate explanations for every point in the test set) ")
        print('---------------------')

        adv_anchors, anchors_explain = get_adv_model_and_exp_func("anchors", "cc", xtrain, ytrain, features, racist_model_f(), innocuous_model_psi(), categorical_features=categorical_features)

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
        fooled_heuristic = compute_scores(summary, features[race_indc], [features[unrelated_indcs]])
        results.append(("anchors", "one_unrelated_feature", fidelity, fooled_heuristic,
                        accuracy_score(*adv_anchors.ood_training_task_ability), summary))

        # Repeat the same thing for two features
        adv_anchors, anchors_explain = get_adv_model_and_exp_func("anchors", "cc", xtrain, ytrain, features, racist_model_f(),
                                                                  innocuous_model_psi_two(),
                                                                  categorical_features=categorical_features)

        def anchors_exp(i):
            tmp = anchors_explain(xtest[i], adv_anchors)
            if i % 50 == 0:
                print(datetime.datetime.now(), i)
            return tmp

        explanations = list(pool.map(anchors_exp, range(limit)))

        print("Anchors Ranks and Pct Occurances for two unrelated features:")
        summary = experiment_summary(explanations, features)
        print(summary)
        fidelity = round(adv_anchors.fidelity(xtest), 2)
        print("Fidelity:", fidelity)
        fooled_heuristic = compute_scores(summary, features[race_indc], [features[unrelated_indcs], features[unrelated_indcs1]])
        results.append(("anchors", "two_unrelated_features", fidelity, fooled_heuristic,
                        accuracy_score(*adv_anchors.ood_training_task_ability), summary))

    if "explan" in models:
        print('---------------------')
        print("Beginning EXPLAN CC Experiments....")
        print("(These take some time to run because we have to generate explanations for every point in the test set) ")
        print('---------------------')

        adv_explan, explan_explain = get_adv_model_and_exp_func("explan", "cc", xtrain, ytrain, features, racist_model_f(),
                                                                  innocuous_model_psi(),
                                                                  categorical_features=categorical_features, pool=pool)

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
        fooled_heuristic = compute_scores(summary, features[race_indc], [features[unrelated_indcs]])
        results.append(("explan", "one_unrelated_feature", fidelity, fooled_heuristic,
                        accuracy_score(*adv_explan.ood_training_task_ability), summary))

        # Repeat the same thing for two features
        adv_explan, explan_explain = get_adv_model_and_exp_func("explan", "cc", xtrain, ytrain, features, racist_model_f(),
                                                                  innocuous_model_psi_two(),
                                                                  categorical_features=categorical_features, pool=pool)

        def explan_exp(i):
            tmp = explan_explain(xtest[i], adv_explan)
            if i % 50 == 0:
                print(datetime.datetime.now(), i)
            return tmp

        explanations = list(pool.map(explan_exp, range(limit)))

        # Display Results
        print("EXPLAN Ranks and Pct Occurances for two unrelated features:")
        summary = experiment_summary(explanations, features)
        print(summary)
        fidelity = round(adv_explan.fidelity(xtest), 2)
        print("Fidelity:", fidelity)
        fooled_heuristic = compute_scores(summary, features[race_indc], [features[unrelated_indcs], features[unrelated_indcs1]])
        results.append(("explan", "two_unrelated_features", fidelity, fooled_heuristic,
                        accuracy_score(*adv_explan.ood_training_task_ability), summary))

    if "lore" in models:
        print('---------------------')
        print("Beginning LORE CC Experiments....")
        print("(These take some time to run because we have to generate explanations for every point in the test set) ")
        print('---------------------')

        adv_lore, lore_explain = get_adv_model_and_exp_func("lore", "cc", xtrain, ytrain, features, racist_model_f(),
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
        fooled_heuristic = compute_scores(summary, features[race_indc], [features[unrelated_indcs]])
        results.append(("lore", "one_unrelated_feature", fidelity, fooled_heuristic,
                        accuracy_score(*adv_lore.ood_training_task_ability), summary))

        # Repeat the same thing for two features
        adv_lore, lore_explain = get_adv_model_and_exp_func("lore", "cc", xtrain, ytrain, features, racist_model_f(),
                                                            innocuous_model_psi_two(),
                                                            categorical_features=categorical_features, pool=pool)

        def lore_exp(i):
            tmp = lore_explain(xtest[i], adv_lore)
            if i % 50 == 0:
                print(datetime.datetime.now(), i)
            return tmp

        explanations = list(pool.map(lore_exp, range(limit)))

        # Display Results
        print("LORE Ranks and Pct Occurances for two unrelated features:")
        summary = experiment_summary(explanations, features)
        print(summary)
        fidelity = round(adv_lore.fidelity(xtest), 2)
        print("Fidelity:", fidelity)
        fooled_heuristic = compute_scores(summary, features[race_indc], [features[unrelated_indcs], features[unrelated_indcs1]])
        results.append(("lore", "two_unrelated_features", fidelity, fooled_heuristic,
                        accuracy_score(*adv_lore.ood_training_task_ability), summary))

    if "lime" in models:
        print('---------------------')
        print("Beginning LIME CC Experiments....")
        print("(These take some time to run because we have to generate explanations for every point in the test set) ")
        print('---------------------')

        adv_lime, lime_explain = get_adv_model_and_exp_func("lime", "cc", xtrain, ytrain, features, racist_model_f(),
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
        fooled_heuristic = compute_scores(summary, features[race_indc], [features[unrelated_indcs]])
        results.append(("lime", "one_unrelated_feature", fidelity, fooled_heuristic,
                        accuracy_score(*adv_lime.ood_training_task_ability), summary))

        # Repeat the same thing for two features
        adv_lime, lime_explain = get_adv_model_and_exp_func("lime", "cc", xtrain, ytrain, features, racist_model_f(),
                                                            innocuous_model_psi_two(),
                                                            categorical_features=categorical_features)

        explanations = []
        for i in range(limit):
            explanations.append(lime_explain(xtest[i], adv_lime))
            if i % 50 == 0:
                print(datetime.datetime.now(), i)

        print("LIME Ranks and Pct Occurances two unrelated features:")
        summary = experiment_summary(explanations, features)
        print(summary)
        fidelity = round(adv_lime.fidelity(xtest), 2)
        print("Fidelity:", fidelity)
        fooled_heuristic = compute_scores(summary, features[race_indc], [features[unrelated_indcs], features[unrelated_indcs1]])
        results.append(("lime", "two_unrelated_features", fidelity, fooled_heuristic,
                        accuracy_score(*adv_lime.ood_training_task_ability), summary))

    if "shap" in models:
        print('---------------------')
        print('Beginning SHAP CC Experiments....')
        print('---------------------')

        adv_shap, shap_explain = get_adv_model_and_exp_func("shap", "cc", xtrain, ytrain, features, racist_model_f(),
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
        fooled_heuristic = compute_scores(summary, features[race_indc], [features[unrelated_indcs]])
        results.append(("shap", "one_unrelated_feature", fidelity, fooled_heuristic,
                        accuracy_score(*adv_shap.ood_training_task_ability), summary))

        adv_shap, shap_explain = get_adv_model_and_exp_func("shap", "cc", xtrain, ytrain, features, racist_model_f(),
                                                            innocuous_model_psi_two(),
                                                            categorical_features=categorical_features)

        explanations = []
        for i in range(limit):
            explanations.append(shap_explain(xtest[i], adv_shap))
            if i % 50 == 0:
                print(datetime.datetime.now(), i)

        print("SHAP Ranks and Pct Occurances two unrelated features:")
        summary = experiment_summary(explanations, features)
        print(summary)
        fidelity = round(adv_shap.fidelity(xtest), 2)
        print("Fidelity:", fidelity)
        fooled_heuristic = compute_scores(summary, features[race_indc], [features[unrelated_indcs], features[unrelated_indcs1]])
        results.append(("shap", "two_unrelated_features", fidelity, fooled_heuristic,
                        accuracy_score(*adv_shap.ood_training_task_ability), summary))
        print('---------------------')

    df = pd.DataFrame(data=results, columns=headers)
    filename = "cc_experiment.csv"
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
