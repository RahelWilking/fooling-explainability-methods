"""
Train a bunch of models to create metric vs perturbation task score graphs.
"""
from adversarial_models import *
from utils import *
from get_data import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from numpy.random import SeedSequence
import pandas as pd

import datetime

from setup_experiments import get_adv_model_and_exp_func, compute_scores

import multiprocessing_on_dill as multiprocessing

from copy import deepcopy

# Set model
# model = "anchors"
model = "lore"
# model = "explan"

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

categorical_features = [i for i, f in enumerate(features) if f not in ['age', 'length_of_stay', 'priors_count']]


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
###

limit = min(100, xtest.shape[0])


def experiment_main():
    data_dict = {'trial': [], 'yhat': [], 'y': [], 'pct_occur_first': [], 'pct_occur_second': [], 'pct_occur_third': [],
                 'fidelity': [], 'fooled_heuristic': [], 'number_preds': [], 'number_perturbation_preds': [],
                 'replication_rate': []}

    trial = 0
    for n_estimators in [1, 2, 4, 8, 16, 32, 64]:
        for max_depth in [1, 2, 4, 8, None]:
            for min_samples_split in [2, 4, 8, 16, 32, 64]:
                estimator = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                   min_samples_split=min_samples_split)

                if model == "anchors":
                    adv_model, explain_func = get_adv_model_and_exp_func(model, "compas", xtrain, ytrain, features,
                                                                         racist_model_f(),
                                                                         innocuous_model_psi(),
                                                                         categorical_features=categorical_features,
                                                                         estimator=estimator)
                else:
                    adv_model, explain_func = get_adv_model_and_exp_func(model, "compas", xtrain, ytrain, features,
                                                                         racist_model_f(),
                                                                         innocuous_model_psi(),
                                                                         categorical_features=categorical_features,
                                                                         estimator=estimator,
                                                                         pool=pool)

                def model_explain(i):
                    tmp = explain_func(xtest[i], adv_model)
                    # if i % 50 == 0:
                    # 	print(datetime.datetime.now(), i)
                    return tmp

                explanations = list(pool.map(model_explain, range(limit)))

                summary = experiment_summary(explanations, features)
                fidelity = round(adv_model.fidelity(xtest[:limit]), 2)
                fooled_heuristic = compute_scores(summary, features[race_indc], [features[unrelated_indcs]])

                pred = adv_model.perturbation_identifier.predict(xtest[:100])
                pred_probs = adv_model.perturbation_identifier.predict_proba(xtest[:100])
                perturbation_preds = (pred_probs[:, 1] >= 0.5)

                pct_occur = [0]
                for indc in [1, 2, 3]:
                    found = False
                    for tup in summary[indc]:
                        if tup[0] == 'race':
                            pct_occur.append(sum([pct_occur[-1], tup[1]]))
                            found = True

                    if not found:
                        pct_occur.append(pct_occur[-1])

                pct_occur = pct_occur[1:]

                y = adv_model.ood_training_task_ability[0]
                yhat = adv_model.ood_training_task_ability[1]
                trial_df = np.array([trial for _ in range(y.shape[0])])

                data_dict['trial'] = np.concatenate((data_dict['trial'], trial_df))
                data_dict['yhat'] = np.concatenate((data_dict['yhat'], yhat))
                data_dict['y'] = np.concatenate((data_dict['y'], y))
                data_dict['pct_occur_first'] = np.concatenate(
                    (data_dict['pct_occur_first'], [pct_occur[0] for _ in range(y.shape[0])]))
                data_dict['pct_occur_second'] = np.concatenate(
                    (data_dict['pct_occur_second'], [pct_occur[1] for _ in range(y.shape[0])]))
                data_dict['pct_occur_third'] = np.concatenate(
                    (data_dict['pct_occur_third'], [pct_occur[2] for _ in range(y.shape[0])]))
                data_dict['fidelity'] = np.concatenate((data_dict['fidelity'], [fidelity for _ in range(y.shape[0])]))
                data_dict['fooled_heuristic'] = np.concatenate(
                    (data_dict['fooled_heuristic'], [fooled_heuristic for _ in range(y.shape[0])]))
                data_dict['number_preds'] = np.concatenate(
                    (data_dict['number_preds'], [sum(pred) for _ in range(y.shape[0])]))
                data_dict['number_perturbation_preds'] = np.concatenate(
                    (data_dict['number_perturbation_preds'], [sum(perturbation_preds) for _ in range(y.shape[0])]))
                data_dict['replication_rate'] = np.concatenate(
                    (data_dict['replication_rate'], [adv_model.replication_rate for _ in range(y.shape[0])]))

                trial += 1

                if trial % 10 == 0:
                    print("Complete {}".format(trial + 1), datetime.datetime.now())

    df = pd.DataFrame(data_dict)

    df.to_csv('data/threshold_results_{}.csv'.format(model))


def seed_process(seed_seq):
    seed = seed_seq.generate_state(1)
    np.random.seed(seed)


if __name__ == "__main__":
    pool = multiprocessing.Pool()
    ss = SeedSequence(params.seed)

    process_seeds = ss.spawn(12)

    pool.map(seed_process, process_seeds)
    experiment_main()
