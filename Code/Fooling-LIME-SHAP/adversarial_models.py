import numpy as np
import pandas as pd
import datetime

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from copy import deepcopy

import random

import shap

import sys

#might not work?
sys.path.insert(1, "../EXPLAN/LORE")
sys.path.insert(1, "../EXPLAN/treeinterpreter")
from treeinterpreter import treeinterpreter as ti
from util import *
from neighbor_generator import genetic_neighborhood
from gpdatagenerator import calculate_feature_values
sys.path.insert(1, "../EXPLAN")
from explan import *
#sys.path.insert(1, "../EXPLAN/LORE/anchor")
from anchor import anchor_tabular

from sklearn.preprocessing import LabelEncoder

class Adversarial_Model(object):
    """	A scikit-learn style adversarial explainer base class for adversarial models.  This accepts
	a scikit learn style function f_obscure that serves as the _true classification rule_ for in distribution
	data.  Also, it accepts, psi_display: the classification rule you wish to display by explainers (e.g. LIME/SHAP).
	Ideally, f_obscure will classify individual instances but psi_display will be shown by the explainer.

	Parameters
	----------
	f_obscure : function
	psi_display : function
	"""

    def __init__(self, f_obscure, psi_display, label_encoder):
        self.f_obscure = f_obscure
        self.psi_display = psi_display

        self.label_encoder = label_encoder

        self.cols = None
        self.scaler = None
        self.numerical_cols = None

        self.test_mode = False

    def predict_proba(self, X, threshold=0.5):
        """ Scikit-learn style probability prediction for the adversarial model.

        Parameters
        ----------
        X : np.ndarray

        Returns
        ----------
        A numpy array of the class probability predictions of the adversarial model.
        """
        if self.perturbation_identifier is None:
            raise NameError("Model is not trained yet, can't perform predictions.")

        if len(self.label_encoder.keys()) > 0: # keys exist
            key = next(iter(self.label_encoder)) # first key (any key is sufficient)
            # print("decode?",X.shape, X[0,key])
            if  int(X[0,key]) == X[0,key]:
                # this means the data is encoded
                X = self.decode(X)

        # generate the "true" predictions on the data using the "bad" model -- this is f in the paper
        predictions_to_obscure = self.f_obscure.predict_proba(X)

        # generate the "explain" predictions -- this is psi in the paper

        predictions_to_explain_by = self.psi_display.predict_proba(X)

        # in the case that we're only considering numerical columns
        if self.numerical_cols:
            X = X[:,self.numerical_cols]

        # allow thresholding for finetuned control over psi_display and f_obscure
        pred_probs = self.perturbation_identifier.predict_proba(X)
        perturbation_preds = (pred_probs[:,1] >= threshold)

        sol = np.where(np.array([perturbation_preds == 1,perturbation_preds==1]).transpose(), predictions_to_obscure, predictions_to_explain_by)

        return sol

    def predict(self, X):
        """	Scikit-learn style prediction. Follows from predict_proba.

        Parameters
        ----------
        X : np.ndarray

        Returns
        ----------
        A numpy array containing the binary class predictions.
        """
        pred_probs = self.predict_proba(X)
        return np.argmax(pred_probs,axis=1)

    def score(self, X_test, y_test):
        """ Scikit-learn style accuracy scoring.

        Parameters:
        ----------
        X_test : X_test
        y_test : y_test

        Returns:
        ----------
        A scalar value of the accuracy score on the task.
        """

        return np.sum(self.predict(X_test)==y_test) / y_test.size

    def get_column_names(self):
        """ Access column names."""

        if self.cols is None:
            raise NameError("Train model with pandas data frame to get column names.")

        return self.cols

    def fidelity(self, X):
        """ Get the fidelity of the adversarial model to the original predictions.  High fidelity means that
        we're predicting f along the in distribution data.

        Parameters:
        ----------
        X : np.ndarray

        Returns:
        ----------
        The fidelity score of the adversarial model's predictions to the model you're trying to obscure's predictions.
        """

        return (np.sum(self.predict(X) == self.f_obscure.predict(X)) / X.shape[0])

    def decode(self, X): # assumes ndarray, might need to fix for df, also assumes index as key
        """
        TODO: Beschreibung hinzufügen
        :param self:
        :param X:
        :return:
        """
        X = np.copy(X)
        for key in self.label_encoder:
            le = self.label_encoder[key]
            if len(X.shape) == 1: # single instance
                X[key] = le.inverse_transform([X[key]].astype(int))[0]
            else:
                X[:,key] = le.inverse_transform(X[:,key].astype(int))
        return X

class Adversarial_Lime_Model(Adversarial_Model):
	""" Lime adversarial model.  Generates an adversarial model for LIME style explainers using the Adversarial Model
	base class.

	Parameters:
	----------
	f_obscure : function
	psi_display : function
	perturbation_std : float
	"""
	def __init__(self, f_obscure, psi_display, label_encoder, perturbation_std=0.3):
		super(Adversarial_Lime_Model, self).__init__(f_obscure, psi_display, label_encoder)
		self.perturbation_std = perturbation_std

	def train(self, X, y, feature_names, perturbation_multiplier=30, categorical_features=[], rf_estimators=100, estimator=None):
		""" Trains the adversarial LIME model.  This method trains the perturbation detection classifier to detect instances
		that are either in the manifold or not if no estimator is provided.

		Parameters:
		----------
		X : np.ndarray or pd.DataFrame
		y : np.ndarray
		perturbation_multiplier : int
		cols : list
		categorical_columns : list
		rf_estimators : integer
		estimaor : func
		"""
		if isinstance(X, pd.DataFrame):
			cols = [c for c in X]
			X = X.values
		elif not isinstance(X, np.ndarray):
			raise NameError("X of type {} is not accepted. Only pandas dataframes or numpy arrays allowed".format(type(X)))

		self.cols = feature_names
		all_x, all_y = [], []

		# loop over perturbation data to create larger data set
		for _ in range(perturbation_multiplier):
			perturbed_xtrain = np.random.normal(0,self.perturbation_std,size=X.shape)
			p_train_x = np.vstack((X, X + perturbed_xtrain))
			p_train_y = np.concatenate((np.ones(X.shape[0]), np.zeros(X.shape[0])))

			all_x.append(p_train_x)
			all_y.append(p_train_y)

		all_x = np.vstack(all_x)
		all_y = np.concatenate(all_y)

		# it's easier to just work with numerical columns, so focus on them for exploiting LIME
		self.numerical_cols = [feature_names.index(c) for c in feature_names if feature_names.index(c) not in categorical_features]

		if self.numerical_cols == []:
			raise NotImplementedError("We currently only support numerical column data. If your data set is all categorical, consider using SHAP adversarial model.")

		# generate perturbation detection model as RF
		xtrain = all_x[:,self.numerical_cols]
		xtrain, xtest, ytrain, ytest = train_test_split(xtrain, all_y, test_size=0.2)

		if estimator is not None:
			self.perturbation_identifier = estimator.fit(xtrain, ytrain)
		else:
			self.perturbation_identifier = RandomForestClassifier(n_estimators=rf_estimators).fit(xtrain, ytrain)

		ypred = self.perturbation_identifier.predict(xtest)
		self.ood_training_task_ability = (ytest, ypred)

		return self

class Adversarial_Kernel_SHAP_Model(Adversarial_Model):
	""" SHAP adversarial model.  Generates an adversarial model for SHAP style perturbations.

	Parameters:
	----------
	f_obscure : function
	psi_display : function
	"""
	def __init__(self, f_obscure, psi_display, label_encoder):
		super(Adversarial_Kernel_SHAP_Model, self).__init__(f_obscure, psi_display, label_encoder)

	def train(self, X, y, feature_names, background_distribution=None, perturbation_multiplier=10, n_samples=2e4, rf_estimators=100, n_kmeans=10, estimator=None):
		""" Trains the adversarial SHAP model. This method perturbs the shap training distribution by sampling from
		its kmeans and randomly adding features.  These points get substituted into a test set.  We also check to make
		sure that the instance isn't in the test set before adding it to the out of distribution set. If an estimator is
		provided this is used.

		Parameters:
		----------
		X : np.ndarray
		y : np.ndarray
		features_names : list
		perturbation_multiplier : int
		n_samples : int or float
		rf_estimators : int
		n_kmeans : int
		estimator : func

		Returns:
		----------
		The model itself.
		"""

		if isinstance(X, pd.DataFrame):
			X = X.values
		elif not isinstance(X, np.ndarray):
			raise NameError("X of type {} is not accepted. Only pandas dataframes or numpy arrays allowed".format(type(X)))

		self.cols = feature_names

		# This is the mock background distribution we'll pull from to create substitutions
		if background_distribution is None:
			background_distribution = shap.kmeans(X,n_kmeans).data
		repeated_X = np.repeat(X, perturbation_multiplier, axis=0)

		new_instances = []
		equal = []

		# We generate n_samples number of substutions
		for _ in range(int(n_samples)):
			i = np.random.choice(X.shape[0])
			point = deepcopy(X[i, :])

			# iterate over points, sampling and updating
			for _ in range(X.shape[1]):
				j = np.random.choice(X.shape[1])
				point[j] = deepcopy(background_distribution[np.random.choice(background_distribution.shape[0]),j])

			new_instances.append(point)

		substituted_training_data = np.vstack(new_instances)
		all_instances_x = np.vstack((repeated_X, substituted_training_data))

		# make sure feature truly is out of distribution before labeling it
		xlist = X.tolist()
		ys = np.array([1 if substituted_training_data[val,:].tolist() in xlist else 0\
						 for val in range(substituted_training_data.shape[0])])

		all_instances_y = np.concatenate((np.ones(repeated_X.shape[0]),ys))

		xtrain,xtest,ytrain,ytest = train_test_split(all_instances_x, all_instances_y, test_size=0.2)

		if estimator is not None:
			self.perturbation_identifier = estimator.fit(xtrain,ytrain)
		else:
			self.perturbation_identifier = RandomForestClassifier(n_estimators=rf_estimators).fit(xtrain,ytrain)

		ypred = self.perturbation_identifier.predict(xtest)
		self.ood_training_task_ability = (ytest, ypred)

		return self

class Adversarial_Anchors_Model(Adversarial_Model):
    """ Anchors adversarial model.  Generates an adversarial model for Anchor style perturbations.

        Parameters:
        ----------
        f_obscure : function
        psi_display : function
        p : parameter for gemoetric distribution, chance of success
        """
    def __init__(self, f_obscure, psi_display, label_encoder, p=0.5):
        super(Adversarial_Anchors_Model, self).__init__(f_obscure, psi_display, label_encoder)
        self.p = p

    def train(self, X, y, feature_names, categorical_features=[], perturbation_multiplier=30, n_samples_per_tuple=5,
              rf_estimators=100, estimator=None, pool=None):
        """ Trains the adversarial Anchors model. Samples n_samples_per_tuple data points from
        perturbation_multiplier/n_samples_per_tuple number of tuples (minimum 1) per instance. For each tuple a random
        tuple length is drawn of a geometric distribution using p as parameter. A tuple of constraints ist randomly
        drawn of all possible constraints of the current instance and a sample adhering to the constraints ist drawn.
        It is also made sure that the instance isn't in the test set before adding it to the out of distribution
        set. If an estimator is provided this is used.

        Parameters:
        ----------
        X : np.ndarray
        y : np.ndarray
        features_names : list
        perturbation_multiplier : int
        n_samples : int or float
        rf_estimators : int
        estimator : func

        Returns:
        ----------
        The model itself.
        """

        if isinstance(X, pd.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            raise NameError(
                "X of type {} is not accepted. Only pandas dataframes or numpy arrays allowed".format(type(X)))

        self.cols = feature_names

        new_X = np.copy(X)
        # Data needs to be label_encoded, since the one_hot_encoder cant handle negative values in the old version it is used at
        # (negative values happen due to the scaling of the data)
        label_encoder = dict()
        # compute categorical names
        categorical_names = dict()
        for idx in categorical_features:
            le = LabelEncoder()
            new_X[:,idx] = le.fit_transform(new_X[:,idx])
            label_encoder[idx] = le
            categorical_names[idx] = le.classes_

        # multiply original data
        repeated_X = np.repeat(X, perturbation_multiplier, axis=0)

        # set up and fit anchortabularexplainer
        # use extra test set?
        explainer = anchor_tabular.AnchorTabularExplainer(class_names=np.unique(y), feature_names=feature_names, data=new_X, categorical_names=categorical_names)
        explainer.fit(new_X,y,new_X,y)

        n_tuples = np.ceil(perturbation_multiplier / n_samples_per_tuple).astype('int')

        num_iterations = new_X.shape[0]
        if self.test_mode:
            num_iterations = min(num_iterations,5)

        def anchors_iter(idx):
            #if idx % 100 == 0:
                #print(datetime.datetime.now(), '%d - %.2f' % (idx, idx / len(new_X)))
            sample_fn, mapping = explainer.get_sample_fn(new_X[idx],
                                                         self.f_obscure.predict)  # is using the f_obscure here a problem? but i dont really have a choice
            n_conditions = len(mapping)
            tuple_sizes = np.random.geometric(self.p, n_tuples)  # geometric distribution for tuple sizes
            sample_data = []
            for size in tuple_sizes:
                #size = size + 1
                if size > n_conditions:  # catch (hopefully rare?) case of to large values of geometric distribution
                    size = n_conditions
                    print("adjusted tuple size down to max value of", n_conditions)
                # sample tuple
                tuple = np.random.choice(n_conditions, size, replace=False)

                tuple_data, _, _ = sample_fn(tuple, n_samples_per_tuple)
                sample_data.extend(tuple_data)
            return sample_data

        perturbed_instances = []
        if pool is None:
            perturbed_instances = list(map(anchors_iter, range(num_iterations)))
        else:
            perturbed_instances = list(pool.map(anchors_iter, range(num_iterations)))

        perturbed_instances = np.vstack(perturbed_instances)

        # decode

        for idx in categorical_features:
            le = label_encoder[idx]
            perturbed_instances[:,idx] = le.inverse_transform(perturbed_instances[:,idx].astype(int))

        # make sure feature truly is out of distribution before labeling it
        xlist = X.tolist()
        ys = np.array([1 if perturbed_instances[val, :].tolist() in xlist else 0 \
                       for val in range(perturbed_instances.shape[0])])
        print("Number of train-instances replicated through perturbations:", sum(ys), sum(ys) / ys.shape[0])
        self.replication_rate = sum(ys) / ys.shape[0]

        prior_count = perturbed_instances.shape[0] + repeated_X.shape[0]

        perturbed_instances = perturbed_instances[np.where(ys == 0)]
        repeated_X = repeated_X[np.random.choice(repeated_X.shape[0], perturbed_instances.shape[0])]

        print("gone from", prior_count, "samples to", perturbed_instances.shape[0] + repeated_X.shape[0])

        all_instances_x = np.vstack((repeated_X, perturbed_instances))
        all_instances_y = np.concatenate((np.ones(repeated_X.shape[0]), np.zeros(perturbed_instances.shape[0])))

        xtrain, xtest, ytrain, ytest = train_test_split(all_instances_x, all_instances_y, test_size=0.2)

        if estimator is not None:
            self.perturbation_identifier = estimator.fit(xtrain, ytrain)
        else:
            self.perturbation_identifier = RandomForestClassifier(n_estimators=rf_estimators).fit(xtrain, ytrain)

        ypred = self.perturbation_identifier.predict(xtest)
        self.ood_training_task_ability = (ytest, ypred)
        return self

class Adversarial_LORE_Model(Adversarial_Model):
    """ LORE adversarial model.  Generates an adversarial model for LORE style perturbations.

        Parameters:
        ----------
        f_obscure : function
        psi_display : function
        """
    def __init__(self, f_obscure, psi_display, label_encoder):
        super(Adversarial_LORE_Model, self).__init__(f_obscure, psi_display, label_encoder)

    def train(self, X, y, feature_names, categorical_features=[], perturbation_multiplier=30,
              rf_estimators=100, estimator=None, pool=None):
        """ Trains the adversarial LORE model. Constructs needed meta-data and then computes genetic neighborhood for
        each point. Subsamples from that neighborhood perturbation_multiplier points. The original data is repeated
        perturbation_multiplier many times to create a balanced dataset. It is also made sure that the instance isn't in
        the test set before adding it to the out of distribution set. If an estimator is provided this is used.

        Parameters:
        ----------
        X : np.ndarray
        y : np.ndarray
        features_names : list
        categorical_features : list
        perturbation_multiplier : int
        rf_estimators : int
        estimator : func

        Returns:
        ----------
        The model itself.
        """

        if isinstance(X, pd.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            raise NameError(
                "X of type {} is not accepted. Only pandas dataframes or numpy arrays allowed".format(type(X)))

        self.cols = feature_names

        # construct meta-information of dataset
        class_name = "target"
        columns = [class_name] + feature_names
        discrete = [feature_names[x] for x in categorical_features]
        continuous = []
        dfXy = pd.DataFrame(data=np.concatenate((y.reshape(-1, 1), X), axis=1), columns=columns)
        type_features, features_type = recognize_features_type(dfXy, class_name)
        discrete, continuous = set_discrete_continuous(columns, type_features, class_name, discrete=discrete, continuous=None)
        idx_features = {i: col for i, col in enumerate(feature_names)}
        df_le, label_encoder = label_encode(dfXy, discrete)

        # need to use encoded data or work around later decoding somehow...
        new_y = df_le.pop(class_name).values
        new_X = df_le.values # after pop

        # dataset in LORE style with only the neccessary field set
        dataset = {
            'columns': list(columns),
            'class_name': class_name,
            'features_type': features_type,
            'discrete': discrete,
            'continuous': continuous,
            'idx_features': idx_features,
            'label_encoder': label_encoder
        }

        # 1000 is the sample size but I'm afraid messing with it might imbalance the genetic algorithm
        dataset['feature_values'] = calculate_feature_values(new_X, columns, class_name, discrete, continuous, 1000,
                                                             discrete_use_probabilities=True, continuous_function_estimation=False)

        repeated_X = np.repeat(X, perturbation_multiplier, axis=0)

        num_iterations = new_X.shape[0]
        if self.test_mode:
            num_iterations = min(num_iterations, 5)

        def lore_iter(idx):
            random.seed(0)  # orientiert an lore.py TODO: check?
            if idx % 100 == 0:
                print(datetime.datetime.now(), '%d - %.2f' % (idx, idx / len(new_X)))
            dfZ, x = dataframe2explain(new_X, dataset, idx, self.f_obscure)
            dfZ, Z = genetic_neighborhood(dfZ, x, self.f_obscure, dataset)

            decoded = dfZ.drop(class_name, axis=1).values
            # subsample
            subsample_idxs = np.random.choice(decoded.shape[0], perturbation_multiplier)
            neighborhood_instances = decoded[subsample_idxs, :]

            return neighborhood_instances

        perturbed_instances = []
        if pool is None:
            perturbed_instances = list(map(lore_iter, range(num_iterations)))
        else:
            perturbed_instances = list(pool.map(lore_iter, range(num_iterations)))

        perturbed_instances = np.vstack(perturbed_instances)
        all_instances_x = np.vstack((repeated_X, perturbed_instances))

        # make sure feature truly is out of distribution before labeling it
        xlist = X.tolist()
        ys = np.array([1 if perturbed_instances[val, :].tolist() in xlist else 0 \
                       for val in range(perturbed_instances.shape[0])])
        print("Number of train-instances replicated through perturbations:", sum(ys), sum(ys) / ys.shape[0])
        self.replication_rate = sum(ys) / ys.shape[0]

        all_instances_y = np.concatenate((np.ones(repeated_X.shape[0]), ys))

        xtrain, xtest, ytrain, ytest = train_test_split(all_instances_x, all_instances_y, test_size=0.2)

        if estimator is not None:
            self.perturbation_identifier = estimator.fit(xtrain, ytrain)
        else:
            self.perturbation_identifier = RandomForestClassifier(n_estimators=rf_estimators).fit(xtrain, ytrain)

        ypred = self.perturbation_identifier.predict(xtest)
        self.ood_training_task_ability = (ytest, ypred)

        return self

class Adversarial_EXPLAN_Model(Adversarial_Model):
    """ EXPLAN adversarial model.  Generates an adversarial model for EXPLAN style perturbations.

        Parameters:
        ----------
        f_obscure : function
        psi_display : function
        """
    def __init__(self, f_obscure, psi_display, label_encoder):
        super(Adversarial_EXPLAN_Model, self).__init__(f_obscure, psi_display, label_encoder)

    def train(self, X, y, feature_names, categorical_features=[], perturbation_multiplier=30, n_samples=3000, tau=250,
              rf_estimators=100, estimator=None, pool=None):
        """ Trains the adversarial EXPLAN model. Constructs needed meta-data and then computes neighborhood generation steps
        for each point. Subsamples from that neighborhood perturbation_multiplier points. The original data is repeated
        perturbation_multiplier many times to create a balanced dataset. It is also made sure that the instance isn't in
        the test set before adding it to the out of distribution set. If an estimator is provided this is used.

        Parameters:
        ----------
        X : np.ndarray
        y : np.ndarray
        features_names : list
        categorical_features : list
        perturbation_multiplier : int
        n_samples : int or float
        tau : int
        rf_estimators : int
        estimator : func

        Returns:
        ----------
        The model itself.
        """

        if isinstance(X, pd.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            raise NameError(
                "X of type {} is not accepted. Only pandas dataframes or numpy arrays allowed".format(type(X)))

        self.cols = feature_names

        # construct meta-information of dataset
        class_name = "target"
        columns = [class_name] + feature_names
        discrete = [feature_names[x] for x in categorical_features]
        continuous = []
        dfXy = pd.DataFrame(data=np.concatenate((y.reshape(-1, 1), X), axis=1), columns=columns)
        type_features, features_type = recognize_features_type(dfXy, class_name)
        discrete, continuous = set_discrete_continuous(columns, type_features, class_name, discrete=discrete,
                                                       continuous=None)
        df_le, label_encoder = label_encode(dfXy, discrete)

        # need to use encoded data or work around later decoding somehow...
        new_y = df_le.pop(class_name).values
        new_X = df_le.values  # after pop

        discrete_indices = list()
        for idx, col in enumerate(feature_names):
            if col == class_name or col in continuous:
                continue
            discrete_indices.append(idx)

        dataset = {
            'X' : new_X,
            'discrete_indices' : discrete_indices
        }

        repeated_X = np.repeat(X, perturbation_multiplier, axis=0)

        num_iterations = new_X.shape[0]
        if self.test_mode:
            num_iterations = min(num_iterations, 5)

        def explan_iter(idx):
            if idx % 100 == 0:
                print(datetime.datetime.now(), '%d - %.2f' % (idx, idx / len(new_X)))
            instance2explain = new_X[idx]

            # Dense data generation step
            dense_samples = DataGeneration(instance2explain, self.f_obscure, dataset, n_samples)

            # Representative data selection step
            representative_samples = DataSelection(instance2explain, self.f_obscure, dense_samples, tau)

            # Data balancing step
            neighborhood_data = DataBalancing(self.f_obscure, representative_samples, dataset)

            # subsample
            subsample_idxs = np.random.choice(neighborhood_data.shape[0], perturbation_multiplier)
            neighborhood_instances = neighborhood_data[subsample_idxs, :]

            return neighborhood_instances

        perturbed_instances = []
        if pool is None:
            perturbed_instances = list(map(explan_iter, range(num_iterations)))
        else:
            perturbed_instances = list(pool.map(explan_iter, range(num_iterations)))

        perturbed_instances = np.vstack(perturbed_instances)

        # decode
        for feat in discrete:
            if feat == class_name:
                continue
            le = label_encoder[feat]
            idx = feature_names.index(feat)
            perturbed_instances[:, idx] = le.inverse_transform(perturbed_instances[:, idx].astype(int))

        all_instances_x = np.vstack((repeated_X, perturbed_instances))

        # make sure feature truly is out of distribution before labeling it
        xlist = X.tolist()
        ys = np.array([1 if perturbed_instances[val, :].tolist() in xlist else 0 \
                       for val in range(perturbed_instances.shape[0])])
        print("Number of train-instances replicated through perturbations:", sum(ys), sum(ys) / ys.shape[0])
        self.replication_rate = sum(ys) / ys.shape[0]

        all_instances_y = np.concatenate((np.ones(repeated_X.shape[0]), ys))

        xtrain, xtest, ytrain, ytest = train_test_split(all_instances_x, all_instances_y, test_size=0.2)

        if estimator is not None:
            self.perturbation_identifier = estimator.fit(xtrain, ytrain)
        else:
            self.perturbation_identifier = RandomForestClassifier(n_estimators=rf_estimators).fit(xtrain, ytrain)

        ypred = self.perturbation_identifier.predict(xtest)
        self.ood_training_task_ability = (ytest, ypred)

        return self