import multiprocessing

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

import statistics

from adversarial_models import *

import lime
import lime.lime_tabular
import shap

sys.path.insert(1, "../EXPLAN/LORE")
import lore
from anchor import anchor_tabular
sys.path.insert(1, "../EXPLAN")
import explan

from pyyadt_feature_importance import mdi_gain_ratio

import multiprocessing_on_dill as multiprocessing


def convert_anchor_explanation(exp, features, method=0):
    features_in_order = exp[0].features()  # features should be the format that goes into categorical_names, so the indexes
    # TODO: add more approaches?

    # possible approaches to duplicates:
    #   add together ranks <- currently chosen (two variants)
    #   only choose top rank

    tuple_length = len(features_in_order)
    result = []
    for f in features:
        if method == 0:  # add decreasing ranks together
            result.append((f, sum([tuple_length - i for i, x in enumerate(features_in_order) if x == features.index(f)])))
        elif method == 1:  # add inverted ranks together (1,1/2,1/3,1/4,...)
            result.append((f, sum([1.0/(i+1) for i, x in enumerate(features_in_order) if x == features.index(f)])))

    return result


def get_adv_model_and_exp_func(model, dataset, xtrain, ytrain, features, f_obscure, psi_display, categorical_features=[], estimator=None, pool=None):
    """
    TODO: Beschreibung hinzufügen

    :param model: string, one of lime, shap, anchors, lore or explan
    :param dataset: string, one of compas, cc or german
    :param xtrain: ndarray, training data
    :param ytrain: ndarray, training labels
    :param features: list, feature names
    :param f_obscure: blackbox model
    :param psi_display: model to display
    :param categorical_features: list, indexes of categorical features.
    :param estimator: estimator to be trained for ood-decision
    :param pool: process pool for multiprocessing
    :return: Adversarial_Model, explain
    """

    # compute label encoder (index-based, for constructors)
    constructor_label_encoder = dict()
    for idx in categorical_features:
        le = LabelEncoder()
        le.fit(xtrain[:, idx])
        constructor_label_encoder[idx] = le

    if model == "lime":
        # TODO: potentionally add the sample around instance somehow?
        adv_lime = Adversarial_Lime_Model(f_obscure, psi_display, constructor_label_encoder).train(xtrain, ytrain,
                                                                                         categorical_features=categorical_features,
                                                                                         feature_names=features,
                                                                                         perturbation_multiplier=30, estimator=estimator)

        adv_lime_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, feature_names=adv_lime.get_column_names(),
                                                               discretize_continuous=False, categorical_features=categorical_features)

        def lime_explain(X, bb):
            if len(X.shape) == 1:  # single instance
                return adv_lime_explainer.explain_instance(X, bb.predict_proba).as_list()
            elif len(X.shape) == 2:  # multiple instances
                return [adv_lime_explainer.explain_instance(X[i], bb.predict_proba).as_list() for i in range(X.shape[0])]
            else:
                print("X has wrong format")

        return adv_lime, lime_explain

    elif model == "shap":
        if dataset == "german":
            background_distribution = KMeans(n_clusters=10, random_state=0).fit(xtrain).cluster_centers_
            adv_shap = Adversarial_Kernel_SHAP_Model(f_obscure, psi_display, constructor_label_encoder)\
                .train(xtrain, ytrain, feature_names=features, background_distribution=background_distribution, n_samples=5e4, estimator=estimator)
        elif dataset == "compas" or dataset == "cc":
            background_distribution = shap.kmeans(xtrain, 10)
            adv_shap = Adversarial_Kernel_SHAP_Model(f_obscure, psi_display, constructor_label_encoder)\
                .train(xtrain, ytrain, feature_names=features)
        else:
            print("Not supported dataset:", dataset)

        def shap_explain(X, bb):
            adv_kernel_explainer = shap.KernelExplainer(bb.predict, background_distribution)
            explanations = adv_kernel_explainer.shap_values(X)
            formatted_explanations = []
            if len(explanations.shape) == 1:
                formatted_explanations = [(features[i], explanations[i]) for i in range(len(explanations))]
            else:
                for exp in explanations:
                    formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])

            return formatted_explanations

        return adv_shap, shap_explain

    elif model == "anchors":
        if dataset == "compas":
            perturbation_multiplier_anchors = 1
            n_samples_per_tuple = 1
            val_percent = 0.2
            p = 0.8
            conversion_method = 0
        elif dataset == "cc":
            perturbation_multiplier_anchors = 1
            n_samples_per_tuple = 1
            val_percent = 0.2
            p = 0.8
            conversion_method = 0
        elif dataset == "german":
            perturbation_multiplier_anchors = 2
            n_samples_per_tuple = 1
            val_percent = 0.2
            p = 0.6
            conversion_method = 0
        else:
            print("Not supported dataset:", dataset)

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

        anchors_xtrain, anchors_xval, anchors_ytrain, anchors_yval = train_test_split(newXtrain, ytrain,
                                                                                      test_size=val_percent)

        adv_anchors = Adversarial_Anchors_Model(f_obscure, psi_display, constructor_label_encoder, p=p).train(newXtrain, ytrain,
                                                                                  categorical_features=categorical_features,
                                                                                  feature_names=features,
                                                                                  perturbation_multiplier=perturbation_multiplier_anchors,
                                                                                  n_samples_per_tuple=n_samples_per_tuple,
                                                                                  estimator=estimator,
                                                                                  pool=pool)

        adv_anchors_explainer = anchor_tabular.AnchorTabularExplainer(np.unique(ytrain), features, newXtrain,
                                                              categorical_names.copy())  # fitting later modifiers therefore give copy here
        adv_anchors_explainer.fit(anchors_xtrain, anchors_ytrain, anchors_xval, anchors_yval)

        def anchors_explain(X, bb):
            if len(X.shape) == 1:  # single instance
                # prep X
                newX = np.copy(X)
                for idx in categorical_features:
                    le = label_encoder[idx]
                    newX[idx] = le.transform([newX[idx]])[0]  # needs to be "array-like"
                return convert_anchor_explanation(adv_anchors_explainer.explain_instance(newX, bb.predict), features, conversion_method)
            elif len(X.shape) == 2:  # multiple instances
                # prep X
                newX = np.copy(X)
                for idx in categorical_features:
                    le = label_encoder[idx]
                    newX[:, idx] = le.transform(newX[:, idx])
                return [convert_anchor_explanation(adv_anchors_explainer.explain_instance(newX[i], bb.predict), features, conversion_method) for i in range(X.shape[0])]
            else:
                print("X has wrong format")

        return adv_anchors, anchors_explain
    elif model == "lore":
        if dataset == "compas":
            perturbation_multiplier_lore = 8
        elif dataset == "cc":
            perturbation_multiplier_lore = 2
        elif dataset == "german":
            perturbation_multiplier_lore = 4
        else:
            print("Not supported dataset:", dataset)

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

        # Train the adversarial model for LORE with f and psi
        adv_lore = Adversarial_LORE_Model(f_obscure, psi_display, constructor_label_encoder).train(xtrain, ytrain,
                                                                                         categorical_features=categorical_features,
                                                                                         feature_names=features,
                                                                                         perturbation_multiplier=perturbation_multiplier_lore,
                                                                                         estimator=estimator,
                                                                                         pool=pool)

        def lore_explain(X, bb, feature_importance_measure=mdi_gain_ratio):
            if len(X.shape) == 1:  # single instance
                # prep X
                newX = np.copy(X)
                for idx in categorical_features:
                    le = label_encoder[features[idx]]
                    newX[idx] = le.transform([newX[idx]])[0]  # needs to be "array-like"

                # create train + instance dataset
                trainplusinstance = np.vstack((newX, newXtrain))

                explanation, infos = lore.explain(0, trainplusinstance, dataset, bb,
                                                  discrete_use_probabilities=True,
                                                  returns_infos=True,
                                                  path="tmp"+str(datetime.datetime.now().timestamp())+str(id(multiprocessing.current_process())))
                dt = infos["dt"]
                neighborhood_data = infos["dfZ"]
                bb_labels = infos["y_pred_bb"]
                return feature_importance_measure(dt, neighborhood_data, bb_labels, features, discrete, features_type)
            elif len(X.shape) == 2:  # multiple instances
                # prep X
                newX = np.copy(X)
                for idx in categorical_features:
                    le = label_encoder[features[idx]]
                    newX[:, idx] = le.transform(newX[:, idx])

                explanations = []
                for i in range(newX.shape[0]):
                    # create train + instance dataset
                    trainplusinstance = np.vstack((newX[i], newXtrain))

                    explanation, infos = lore.explain(0, trainplusinstance, dataset, bb,
                                                      discrete_use_probabilities=True,
                                                      returns_infos=True,
                                                      path="tmp"+str(datetime.datetime.now().timestamp())+str(id(multiprocessing.current_process())))
                    dt = infos["dt"]
                    neighborhood_data = infos["Z"]
                    bb_labels = infos["y_pred_bb"]
                    explanations.append(
                        feature_importance_measure(dt, neighborhood_data, bb_labels, features, discrete, features_type))
                return explanations
            else:
                print("X has wrong format")

        return adv_lore, lore_explain
    elif model == "explan":
        if dataset == "compas":  # seems not yet acceptable
            perturbation_multiplier_explan = 3
            n_samples = 3000
            tau = 250
        elif dataset == "cc":  # seems to be already acceptable
            perturbation_multiplier_explan = 1
            n_samples = 3000
            tau = 250
        elif dataset == "german":
            perturbation_multiplier_explan = 3
            n_samples = 3000
            tau = 250
        else:
            print("Not supported dataset:", dataset)

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

        # Train the adversarial model for EXPLAN with f and psi
        adv_explan = Adversarial_EXPLAN_Model(f_obscure, psi_display, constructor_label_encoder).train(xtrain, ytrain,
                                                                                             categorical_features=categorical_features,
                                                                                             feature_names=features,
                                                                                             perturbation_multiplier=perturbation_multiplier_explan,
                                                                                             n_samples=n_samples,
                                                                                             tau=tau,
                                                                                             estimator=estimator,
                                                                                             pool=pool)

        def explan_explain(X, bb, feature_importance_measure=mdi_gain_ratio):
            if len(X.shape) == 1:  # single instance
                # prep X
                newX = np.copy(X)
                for idx in categorical_features:
                    le = label_encoder[features[idx]]
                    newX[idx] = le.transform([newX[idx]])[0]  # needs to be "array-like"

                exp_EXPLAN, info_EXPLAN = explan.Explainer(newX,
                                                           bb,
                                                           dataset) # could give n_samples and tau here...but should probably be default?
                dt = info_EXPLAN["C"]
                neighborhood_data = info_EXPLAN["dfX"]
                bb_labels = info_EXPLAN["y_X_bb"]
                return feature_importance_measure(dt, neighborhood_data, bb_labels, features, discrete, features_type)
            elif len(X.shape) == 2:  # multiple instances
                # prep X
                newX = np.copy(X)
                for idx in categorical_features:
                    le = label_encoder[features[idx]]
                    newX[:, idx] = le.transform(newX[:, idx])

                explanations = []
                for i in range(newX.shape[0]):
                    exp_EXPLAN, info_EXPLAN = explan.Explainer(newX[i],
                                                               bb,
                                                               dataset,
                                                               N_samples=n_samples,
                                                               tau=tau)
                    dt = info_EXPLAN["C"]
                    neighborhood_data = info_EXPLAN["X"]
                    bb_labels = info_EXPLAN["y_X_bb"]
                    explanations.append(
                        feature_importance_measure(dt, neighborhood_data, bb_labels, features, discrete, features_type))
                return explanations
            else:
                print("X has wrong format")

        return adv_explan, explan_explain
    else:
        print("Model needs to be one of lime, shap, anchors, lore or explan.")
        return None, None


# from https://pythonhealthcare.org/tag/pareto-front/
def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] <= scores[i]) and any(scores[j] < scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]


def kl_divergence(p, q):
    # Do not divide by zero
    eps = 1e-9
    p[np.where(p == 0)] = eps
    q[np.where(q == 0)] = eps
    return np.sum(np.log(p / q) * p)


def compute_scores(summary, sensitive_feature, relevant_features):
    #print(summary)
    sensitive_dist = [[val for feat,val in rank if feat == sensitive_feature][0] if sensitive_feature in [feat for feat,val in rank] else 0.0 for rank in summary.values()]
    sensitive_dist.append(1-sum(sensitive_dist)) # add percentage for not in ranking
    #print(sensitive_dist)
    relevant_dists = []
    for relevant_feat in relevant_features:
        relevant_dist = [[val for feat,val in rank if feat == relevant_feat][0] if relevant_feat in [feat for feat,val in rank] else 0.0 for rank in summary.values()]
        relevant_dist.append(1 - sum(relevant_dist))
        #print(relevant_dist)
        relevant_dists.append(relevant_dist)

    hidden_dist = [0, 0, 0, 1.0]
    shown_dist = [1.0, 0, 0, 0]
    divergences = [kl_divergence(np.array(hidden_dist), np.array(sensitive_dist))]
    for dist in relevant_dists:
        divergences.append(kl_divergence(np.array(shown_dist), np.array(dist)))

    return statistics.harmonic_mean(divergences)








