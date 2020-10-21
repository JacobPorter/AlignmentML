#!/usr/bin/env python
import datetime
import gzip
import itertools
import math
import optparse
import os
import pickle as pickle
import sys
#import seaborn as sns
from collections import Counter

import numpy as np
from sklearn import preprocessing
#from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, make_scorer)
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
#from skopt import gp_minimize
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
"""
TODO: Use Bayesian optimization to optimize hyperparameters
"""

verbose = 8
SAVE_DIR = "./"


class RandomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, seed=None):
        self.seed = seed

    def fit(self, X, y=None):
        if len(y.shape) == 1:
            y_list = list(y)
        else:
            y_list = list(y[0])
        total = len(y_list) + 0.0
        counts = Counter(y_list)
        freq = {key: counts[key] / total for key in counts}
        keys = list(freq.keys())
        keys.sort()
        self.freq_dict = {}
        s = 0.0
        for k in keys:
            self.freq_dict[k] = s + freq[k]
            s += freq[k]
        print("self.freq_dict {}".format(self.freq_dict))
        return self

    def predict(self, X, y=None):
        def random_class(rand_n):
            s = 0.0
            labels = list(self.freq_dict.keys())
            labels.sort()
            for k in labels:
                if rand_n >= s and rand_n < self.freq_dict[k]:
                    return k
                s = self.freq_dict[k]

        if self.seed != None:
            np.random.seed(self.seed)
        samples = np.random.rand(X.shape[0])
        vfunc = np.vectorize(random_class)
        return vfunc(samples)

    def score(self, X, y=None):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / (y.size + 0.0)


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 max_depth=15,
                 max_features=0.5,
                 C_lr=0.5,
                 activation='relu',
                 alpha=0.999,
                 hidden_layer_sizes=(30, 20, 15, 10),
                 k=10):
        self.max_depth = max_depth
        self.max_features = max_features
        self.C_lr = C_lr
        self.alpha = alpha
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activiation = activation
        self.k = k

    def fit(self, X, y=None):
        self.lr = LogisticRegression(C=self.C_lr)
        self.rf = RandomForestClassifier(max_depth=self.max_depth,
                                         max_features=self.max_features)
        self.mlp = MLPClassifier(solver='sgd',
                                 learning_rate='invscaling',
                                 activation=self.activiation,
                                 alpha=self.alpha,
                                 hidden_layer_sizes=self.hidden_layer_sizes,
                                 random_state=1)
        #         gpc_estimator = [('anova', SelectKBest(f_classif, k=k)),
        #                   ('gpc', GaussianProcessClassifier())]
        #         self.gpc = Pipeline(gpc_estimator)
        self.stacker = RandomForestClassifier(max_features=None)
        tick1 = datetime.datetime.now()
        self.rf.fit(X, y)
        tick2 = datetime.datetime.now()
        sys.stderr.write("RF took time {0} to fit.\n".format(tick2 - tick1))
        sys.stderr.flush()
        self.lr.fit(X, y)
        tick3 = datetime.datetime.now()
        sys.stderr.write("LR took time {0} to fit.\n".format(tick3 - tick2))
        sys.stderr.flush()
        self.mlp.fit(X, y)
        tick4 = datetime.datetime.now()
        sys.stderr.write("MLP took time {0} to fit.\n".format(tick4 - tick3))
        sys.stderr.flush()
        #         self.gpc.fit(X_scaled, y)
        y_rf = self.rf.predict(X)
        y_lr = self.lr.predict(X)
        y_mlp = self.mlp.predict(X)
        tick5 = datetime.datetime.now()
        sys.stderr.write("Prediction took time {0} to fit.\n".format(tick5 -
                                                                     tick4))
        sys.stderr.flush()
        #         y_gpc = self.gpc.predict(X_scaled)
        sys.stderr.write(
            "Predictions dimensions {0}, {1}, {2}.  y dimensions {3}\n".format(
                y_rf.shape, y_lr.shape, y_mlp.shape, y.shape))
        sys.stderr.flush()
        X_stacked = np.concatenate(
            (np.array([y_rf]).T, np.array([y_lr]).T, np.array([y_mlp]).T),
            axis=1)
        sys.stderr.write(
            "X_stacked dimensions {0}.  y dimensions {1}\n".format(
                X_stacked.shape, y.shape))
        sys.stderr.flush()
        self.stacker.fit(X_stacked, y)
        return self

    def predict(self, X, y=None):
        y_rf = self.rf.predict(X)
        y_lr = self.lr.predict(X)
        y_mlp = self.mlp.predict(X)
        #         y_gpc = self.gpc.predict(X_scaled)
        return self.stacker.predict(
            np.concatenate(
                (np.array([y_rf]).T, np.array([y_lr]).T, np.array([y_mlp]).T),
                axis=1))

    def score(self, X, y=None):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / (y.size + 0.0)


kappa_scorer = make_scorer(cohen_kappa_score)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Selects columns for keeping or deleting in a numpy 2D array.
    """
    def __init__(self, columns, delete=True):
        self.columns = columns
        self.delete = delete

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.delete:
            return np.delete(X, self.columns, axis=1)
        else:
            return np.delete(
                X, [i for i in range(X.shape[1]) if i not in self.columns],
                axis=1)


def getScaler(X_train):
    return preprocessing.StandardScaler(copy=True,
                                        with_mean=True,
                                        with_std=True).fit(X_train)


def randomize(X, y, seed=321):
    # Generate the permutation index array.
    if seed != None:
        np.random.seed(seed)
    permutation = np.random.permutation(X.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = X[permutation]
    shuffled_b = y[permutation]
    return shuffled_a, shuffled_b


def getPipeLR():
    lr = LogisticRegression()
    estimators = [('scaler',
                   preprocessing.StandardScaler(copy=True,
                                                with_mean=True,
                                                with_std=True)), ('lr', lr)]
    return Pipeline(estimators)


def getPipeRF():
    rf = RandomForestClassifier(max_depth=15, max_features=0.5)
    estimators = [('scaler',
                   preprocessing.StandardScaler(copy=True,
                                                with_mean=True,
                                                with_std=True)), ('rf', rf)]
    return Pipeline(estimators)


def getPipeGB():
    gb = GradientBoostingClassifier()
    estimators = [('scaler',
                   preprocessing.StandardScaler(copy=True,
                                                with_mean=True,
                                                with_std=True)), ('gb', gb)]
    return Pipeline(estimators)


def getPipeEnsemble():
    ens = EnsembleClassifier()
    estimators = [('scaler',
                   preprocessing.StandardScaler(copy=True,
                                                with_mean=True,
                                                with_std=True)), ('ens', ens)]
    return Pipeline(estimators)


def getPipeSVC():
    #svc = svm.SVC()
    svc = LinearSVC(max_iter=5000)
    #anova_filter = SelectKBest(f_classif, k=10)
    estimators = [
        ('scaler',
         preprocessing.StandardScaler(copy=True, with_mean=True,
                                      with_std=True)),
        #('anova', anova_filter),
        ('svc', svc)
    ]
    return Pipeline(estimators)


def getPipeGPC():
    gpc = GaussianProcessClassifier()
    anova_filter = SelectKBest(f_classif, k=10)
    estimators = [('scaler',
                   preprocessing.StandardScaler(copy=True,
                                                with_mean=True,
                                                with_std=True)),
                  ('anova', anova_filter), ('gpc', gpc)]
    return Pipeline(estimators)


def getPipeMLP():
    mlp = MLPClassifier(solver='sgd',
                        learning_rate='invscaling',
                        activation='relu',
                        alpha=1e-5,
                        hidden_layer_sizes=(20, 15),
                        random_state=1)
    estimators = [('scaler',
                   preprocessing.StandardScaler(copy=True,
                                                with_mean=True,
                                                with_std=True)), ('mlp', mlp)]
    return Pipeline(estimators)


def DoAnova(X, y):
    scaler = preprocessing.StandardScaler(copy=True,
                                          with_mean=True,
                                          with_std=True)
    scaler.fit(X)
    return f_classif(scaler.transform(X), y)


# columns_to_remove = [1, 22, 34, 46, 58]
features_to_remove = [
    ' length', ' qual_kurt', ' qual_kurt_1', ' qual_kurt_2', ' qual_kurt_3'
]


def remove_columns_index(feature_labels, features_to_remove):
    columns_to_remove = []
    for feature in features_to_remove:
        feature = feature.strip()
        for i, label in enumerate(feature_labels):
            if feature in label:
                columns_to_remove.append(i)
                break
    print("Columns to remove: {}".format(columns_to_remove), file=sys.stderr)
    return columns_to_remove


#22(' qual_kurt', nan, nan)
#34(' qual_kurt_1', nan, nan)
#1(' length', -64.657361834803481, nan)
#58(' qual_kurt_3', nan, nan)
#46(' qual_kurt_2', 0.0018265588955874504)


def checkNumericMatrix(m):
    # max_float = 3.402823 * math.pow(10, 37)
    max_float_1 = 3.402823 * math.pow(10, 37)
    replace_value = 3.402823 * math.pow(10, 15)
    boolean = True
    for i in range(len(m)):
        for j in range(len(m[i])):
            if m[i][j] >= max_float_1:
                boolean = False
                #sys.stderr.write("Found element that was too large:\t%d\t%d\t%d\n" % (i, j, m[i][j]))
                m[i][j] = replace_value
            if m[i][j] <= -max_float_1:
                boolean = False
                m[i][j] = -replace_value
    return boolean


def checkNumPyMatrix(mat):
    sys.stderr.write("isnan: " + str(np.any(np.isnan(mat))) + " isfinite: " +
                     str(np.isfinite(mat.all())) + "\n")


def checkYMatrix(y):
    for item in y:
        if item >= 4 or item < 0:
            sys.stderr.write("Something is wrong with y:\t%s\n" % (str(item)))


def doML(X,
         y_dict,
         n_jobs,
         classifier,
         datasize=None,
         testsize=100000,
         write_data=None,
         bayesOpt=False,
         acqFunc="",
         num_iter=25,
         save_dir="./"):
    global SAVE_DIR
    SAVE_DIR = save_dir
    class_labels = ['Unique', 'Ambig', 'Unmap', 'Filt']
    classifier = classifier.upper()
    sys.stdout.write(
        "Finished with making the features and labels matrices.\n")
    sys.stdout.flush()
    X, X_ids, feature_labels = X
    sys.stdout.write("Checking the X matrx: {0}\n".format(X.shape))
    y = [float(y_dict[ide][0]) for ide in X_ids]
    checkNumPyMatrix(X)
    y = np.array(y, dtype='float32')
    sys.stdout.write("Checking the y matrix: {0}\n".format(y.shape))
    sys.stderr.write(str(y[0:3]) + "\n")
    checkYMatrix(y)
    checkNumPyMatrix(y)
    sys.stdout.write("Dividing the data into random test and training sets.")
    X, y = randomize(X, y)
    remove_columns = ColumnSelector(remove_columns_index(
        feature_labels, features_to_remove),
                                    delete=True)
    X = remove_columns.transform(X)
    for label in features_to_remove:
        feature_labels.remove(label)
    if datasize != None and datasize <= X.shape[0]:
        X_train = X[:datasize]
        y_train = y[:datasize]
    sys.stdout.write(
        "The size of X_train is %s, and the size of y_train is %s.\n" %
        (str(X_train.shape), str(y_train.shape)))
    if testsize != None:
        X_test = X[datasize:datasize + testsize]
        y_test = y[datasize:datasize + testsize]
    sys.stdout.write(
        "The size of X_test is %s, and the size of y_test is %s.\n" %
        (str(X_test.shape), str(y_test.shape)))
    if write_data != None:
        scaler = preprocessing.StandardScaler(copy=True,
                                              with_mean=True,
                                              with_std=True)
        scaler.fit(X_train)
        np.save(write_data + '_' + str(datasize) + '_' + 'X_train_scaled.npy',
                scaler.transform(X_train))
        np.save(write_data + '_' + str(testsize) + '_' + 'X_test_scaled.npy',
                scaler.transform(X_test))
        np.save(write_data + '_' + str(datasize) + '_' + 'y_train.npy',
                y_train)
        np.save(write_data + '_' + str(testsize) + '_' + 'y_test.npy', y_test)
    sys.stdout.write("Now doing machine learning.\n")
    sys.stdout.flush()
    i = 0
    for label in feature_labels:
        print(i, label)
        i += 1
    f_test, pval = DoAnova(X_train, y_train)
    f_test_features = sorted(zip(feature_labels, f_test, pval),
                             key=lambda x: x[1],
                             reverse=True)
    print("")
    print("")
    print("Features and F-Test with pval")
    i = 1
    for item in f_test_features:
        print(i, str(item))
        i += 1
    print("")
    sys.stdout.flush()
    if 'RAND' in classifier:
        result_rand = doRand(X_train, y_train, X_test, y_test, n_jobs,
                             feature_labels, class_labels, bayesOpt, acqFunc)
        sys.stdout.write("Finished with Random.\n")
        sys.stdout.flush()
    if 'RF' in classifier:
        result_rf = doRandomForest(X_train, y_train, X_test, y_test, n_jobs,
                                   feature_labels, class_labels, bayesOpt,
                                   acqFunc, num_iter)
        sys.stdout.write("Finished with random forest.\n")
        sys.stdout.flush()
    if 'GB' in classifier:
        result_gb = doGradBoost(X_train, y_train, X_test, y_test, n_jobs,
                                feature_labels, class_labels, bayesOpt,
                                acqFunc, num_iter)
        sys.stdout.write("Finished with grad boost.\n")
        sys.stdout.flush()
    if 'MLP' in classifier:
        result_mlp = doMLP(X_train, y_train, X_test, y_test, n_jobs,
                           feature_labels, class_labels, bayesOpt, acqFunc,
                           num_iter)
        sys.stdout.write("Finished with MLP.\n")
        sys.stdout.flush()
    if 'LR' in classifier:
        result_lr = doLogisticRegression(X_train, y_train, X_test, y_test,
                                         n_jobs, feature_labels, class_labels,
                                         bayesOpt, acqFunc, num_iter)
        sys.stdout.write("Finished with logistic regression.\n")
        sys.stdout.flush()
    if 'ENSEMBLE' in classifier:
        result_ensemble = doEnsemble(X_train, y_train, X_test, y_test, n_jobs,
                                     feature_labels, class_labels, bayesOpt,
                                     acqFunc, num_iter)
        sys.stdout.write("Finished with ensemble.\n")
        sys.stdout.flush()
    if 'SVC' in classifier:
        result_svc = doSVC(X_train, y_train, X_test, y_test, n_jobs,
                           feature_labels, class_labels, bayesOpt, acqFunc,
                           num_iter)
        sys.stdout.write("Finished with svc.\n")
        sys.stdout.flush()


#    GPC runs out of memory
#     if 'GPC' in classifier:
#     result_gpc = doGPC(X_train, y_train, X_test, y_test, n_jobs, feature_labels, class_labels)
#     sys.stdout.write("Finished with gpc.\n")
#     sys.stdout.flush()
# return (result_rf, result_svc, result_lr, result_ensemble, result_mlp, result_rand)


def executeML(X,
              y,
              X_test,
              y_test,
              n_jobs,
              feature_labels,
              class_labels,
              pipe,
              parameters,
              ml_type,
              bayesOpt=False,
              search_space=None,
              n_iter=32,
              acq_func=""):
    if bayesOpt:
        # Example search space: { 'C': Real(1e-6, 1e+6, prior='log-uniform'), 'gamma': Real(1e-6, 1e+1, prior='log-uniform'), 'degree': Integer(1,8), 'kernel': Categorical(['linear', 'poly', 'rbf']), }
        acq_funcs = ("LCB", "EI", "PI", "gp_hedge")
        for i in range(len(acq_funcs)):
            if acq_func in acq_funcs[i]:
                break
        if i == len(acq_funcs):
            i = i - 1
        optimizer_kwargs = {'acq_func': acq_funcs[i]}
        cv = BayesSearchCV(pipe,
                           search_space,
                           verbose=verbose,
                           n_iter=n_iter,
                           n_jobs=n_jobs,
                           optimizer_kwargs=optimizer_kwargs,
                           scoring=kappa_scorer)
    else:
        cv = GridSearchCV(pipe,
                          parameters,
                          verbose=verbose,
                          n_jobs=n_jobs,
                          scoring=kappa_scorer)
    tick1 = datetime.datetime.now()
    cv.fit(X, y)
    tick2 = datetime.datetime.now()
    print("\n{0} fitting time: {1}".format(ml_type, tick2 - tick1))
    print("{0} best params {1}".format(ml_type, cv.best_params_))
    print("{0} best score {1}".format(ml_type, cv.best_score_))
    tick3 = datetime.datetime.now()
    y_predict = cv.predict(X_test)
    tick4 = datetime.datetime.now()
    print("{0} test set accuracy: {1}".format(
        ml_type, accuracy_score(y_test, y_predict)))
    print("{0} test set cohen kappa: {1}".format(
        ml_type, cohen_kappa_score(y_test, y_predict)))
    print("{0} prediction time: {1}".format(ml_type, tick4 - tick3))
    print(
        classification_report(y_test,
                              y_predict,
                              target_names=class_labels,
                              digits=8))
    cnf_matrix = confusion_matrix(y_test, y_predict)
    print("")
    print(cnf_matrix)
    pickle.dump(
        cnf_matrix,
        open(os.path.join(SAVE_DIR, "cnf_matrix.best." + ml_type + ".pck"),
             'wb'))
    pickle.dump(
        cv.best_estimator_,
        open(os.path.join(SAVE_DIR, "classifier.best." + ml_type + ".pck"),
             'wb'))
    return cv.best_estimator_


def doRand(X,
           y,
           X_test,
           y_test,
           n_jobs,
           feature_labels,
           class_labels,
           bayesOpt=False,
           acqFunc='',
           num_iter=25):
    # Best: 'lr__C': 0.74
    random_classifier = RandomClassifier(seed=42)
    tick1 = datetime.datetime.now()
    random_classifier.fit(X, y)
    tick2 = datetime.datetime.now()
    ml_type = "Random_classifier"
    print("\n{0} fitting time: {1}".format(ml_type, tick2 - tick1))
    tick3 = datetime.datetime.now()
    y_predict = random_classifier.predict(X_test)
    tick4 = datetime.datetime.now()
    print("{0} test set accuracy: {1}".format(
        ml_type, accuracy_score(y_test, y_predict)))
    print("{0} test set cohen kappa: {1}".format(
        ml_type, cohen_kappa_score(y_test, y_predict)))
    print("{0} prediction time: {1}".format(ml_type, tick4 - tick3))
    print(
        classification_report(y_test,
                              y_predict,
                              target_names=class_labels,
                              digits=8))
    cnf_matrix = confusion_matrix(y_test, y_predict)
    print("")
    print(cnf_matrix)
    pickle.dump(
        cnf_matrix,
        open(os.path.join(SAVE_DIR, "cnf_matrix.best." + ml_type + ".pck"),
             'wb'))
    pickle.dump(
        random_classifier,
        open(os.path.join(SAVE_DIR, "classifier.best." + ml_type + ".pck"),
             'wb'))
    return random_classifier


def doRandomForest(X,
                   y,
                   X_test,
                   y_test,
                   n_jobs,
                   feature_labels,
                   class_labels,
                   bayesOpt=False,
                   acqFunc='',
                   num_iter=25):
    # Best parameters: max_depth 15, max_features 0.6
    #     parameters = {"rf__max_depth": [5, 15, 25, 35, 45, 55, 65],
    #                   "rf__max_features": ['sqrt', None, 0.5]}
    search_space = {
        'rf__max_depth': Integer(1, 55),
        "rf__max_features": Real(.05, 1.0, prior='log-uniform')
    }
    parameters = {
        "rf__max_depth": [i for i in range(5, 60, 5)],
        "rf__max_features": [i / 10.0 for i in range(1, 11)]
    }
    best_estimator = executeML(X,
                               y,
                               X_test,
                               y_test,
                               n_jobs,
                               feature_labels,
                               class_labels,
                               getPipeRF(),
                               parameters,
                               "RF" + acqFunc,
                               bayesOpt=bayesOpt,
                               search_space=search_space,
                               n_iter=num_iter,
                               acq_func=acqFunc)
    try:
        imp = best_estimator.named_steps['rf'].feature_importances_
        feature_importances = sorted(zip(feature_labels, imp),
                                     key=lambda fi: fi[1],
                                     reverse=True)
        print("")
        print("Random Forest feature importances.")
        i = 1
        for item in feature_importances:
            print(i, str(item))
            i += 1
        print("")
        sys.stdout.flush()
    except AttributeError:
        print("How to get the feature importances for RF?")
    return best_estimator


def doGradBoost(X,
                y,
                X_test,
                y_test,
                n_jobs,
                feature_labels,
                class_labels,
                bayesOpt=False,
                acqFunc='',
                num_iter=25):
    #     MIN_SEARCH = 2e-12
    #     num_features = 67
    #     search_space = {
    #         'gb__n_estimators': Integer(50, 150),
    #         'gb__max_features': Real(MIN_SEARCH, 1.0, prior='uniform'),
    #         'gb__criterion': Categorical(['friedman_mse',
    #                                              'mse']),  # mae is very slow.
    #         'gb__min_samples_split': Real(MIN_SEARCH,
    #                                              1.0,
    #                                              prior='log-uniform'),
    #         'gb__min_samples_leaf': Real(MIN_SEARCH,
    #                                             0.5,
    #                                             prior='log-uniform'),
    #         'gb__max_depth': Integer(2, num_features),
    #     }
    search_space = {
        'rf__max_depth': Integer(1, 55),
        "rf__max_features": Real(.05, 1.0, prior='log-uniform')
    }
    parameters = {
        'gb__max_depth': [i for i in range(5, 60, 5)],
        "gb__max_features": [i / 10.0 for i in range(1, 11)]
    }
    best_estimator = executeML(X,
                               y,
                               X_test,
                               y_test,
                               n_jobs,
                               feature_labels,
                               class_labels,
                               getPipeRF(),
                               parameters,
                               "GB" + acqFunc,
                               bayesOpt=bayesOpt,
                               search_space=search_space,
                               n_iter=num_iter,
                               acq_func=acqFunc)
    try:
        imp = best_estimator.named_steps['gb'].feature_importances_
        feature_importances = sorted(zip(feature_labels, imp),
                                     key=lambda fi: fi[1],
                                     reverse=True)
        print("")
        print("Grad boost feature importances.")
        i = 1
        for item in feature_importances:
            print(i, str(item))
            i += 1
        print("")
        sys.stdout.flush()
    except AttributeError:
        print("How to get the feature importances for GB?")
    return best_estimator


def doLogisticRegression(X,
                         y,
                         X_test,
                         y_test,
                         n_jobs,
                         feature_labels,
                         class_labels,
                         bayesOpt=False,
                         acqFunc='',
                         num_iter=25):
    # Best: 'lr__C': 0.74
    search_space = {"lr__C": Real(.001, 1.0, prior='log-uniform')}
    # [0.01, 0.25, 0.5, 0.75, 1.0]
    parameters = {"lr__C": [i / 50.0 for i in range(1, 51)]}
    best_estimator = executeML(X,
                               y,
                               X_test,
                               y_test,
                               n_jobs,
                               feature_labels,
                               class_labels,
                               getPipeLR(),
                               parameters,
                               "LR",
                               bayesOpt=bayesOpt,
                               search_space=search_space,
                               n_iter=num_iter,
                               acq_func=acqFunc)
    try:
        imp = best_estimator.named_steps['lr'].coef_
        print(str(imp))
    except AttributeError:
        print("How to get the feature importances for LR?")
    return best_estimator


def doSVC(X,
          y,
          X_test,
          y_test,
          n_jobs,
          feature_labels,
          class_labels,
          bayesOpt=False,
          acqFunc='',
          num_iter=5):
    """
    From the API documentation:
    The implementation is based on libsvm.
    The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset with more than a couple of 10000 samples.
    """
    search_space = {"svc__C": Real(.001, 1.0, prior='log-uniform')}
    parameters = {
        'svc__C': [0.01, 0.25, 0.5, 0.75,
                   1.0],  # [i/50.0 for i in range(1, 51)]
    }
    #     parameters = {'svc__kernel':['poly'],
    #                   'svc__max_iter':[-1],
    #                   'svc__tol':[1e-2],
    #                   #'anova__k':(10),
    #                   'svc__C':[0.5],
    #                   } #'C':[1, 0, 0.1, ]
    best_estimator = executeML(X,
                               y,
                               X_test,
                               y_test,
                               n_jobs,
                               feature_labels,
                               class_labels,
                               getPipeSVC(),
                               parameters,
                               "SVC",
                               bayesOpt=bayesOpt,
                               search_space=search_space,
                               n_iter=num_iter,
                               acq_func=acqFunc)
    try:
        imp = best_estimator.named_steps['svc'].coef_
        print(str(imp))
    except AttributeError:
        print("How to get the feature importances for SVC?")
    print("SVC number of support vectors: {0}".format(
        best_estimator.named_steps['svc'].n_support_))
    return best_estimator


# def doGPC(X, y, X_test, y_test, n_jobs, feature_labels, class_labels, bayesOpt = False, acqFunc = ''):
#     parameters = {'anova__k':[5, 10, 15, 20],
#                   }
#     return executeML(X, y, X_test, y_test, n_jobs, feature_labels, class_labels, getPipeGPC(), parameters, "GPC", bayesOpt = False, acqFunc = "")


def doEnsemble(X,
               y,
               X_test,
               y_test,
               n_jobs,
               feature_labels,
               class_labels,
               bayesOpt=False,
               acqFunc='',
               num_iter=25):
    search_space = {}
    parameters = {
        'ens__max_depth': [15],
        'ens__max_features': [0.5],
        'ens__C_lr': [0.5],
        'ens__C_svc': [0.5],
        'ens__kernel': ['poly'],
        'ens__max_iter': [10000]
    }
    return executeML(X,
                     y,
                     X_test,
                     y_test,
                     n_jobs,
                     feature_labels,
                     class_labels,
                     getPipeEnsemble(),
                     parameters,
                     "Ensemble",
                     bayesOpt=bayesOpt,
                     search_space=search_space,
                     n_iter=num_iter,
                     acq_func=acqFunc)


def doMLP(X,
          y,
          X_test,
          y_test,
          n_jobs,
          feature_labels,
          class_labels,
          bayesOpt=False,
          acqFunc='',
          num_iter=25):
    # Best parameters: alpha=0.999, hidden_layer_sizes=(30, 20, 15, 10)
    search_space = {
        "mlp__alpha": Real(.001, 1.0, prior='log-uniform'),
        "mlp__hidden_layer_sizes": Categorical([(30, 20, 15, 10)])
    }
    parameters = {
        'mlp__alpha':
        [i / 10.0 for i in range(1, 11)],  # [1e-5, 0.25, 0.5, 0.999],
        # [(30, 20, 15, 10), (30, 20, 15, 10, 5), (40, 30, 20, 15, 10, 5), (40, 40, 40, 40, 40, 40)],#[(20, 15), (20, 20), (20, 15, 5), (30, 20), (30, 20, 15, 10)],
        'mlp__hidden_layer_sizes': [(30, 20, 15, 10)],
    }
    best_estimator = executeML(X,
                               y,
                               X_test,
                               y_test,
                               n_jobs,
                               feature_labels,
                               class_labels,
                               getPipeMLP(),
                               parameters,
                               "MLP",
                               bayesOpt=bayesOpt,
                               search_space=search_space,
                               n_iter=num_iter,
                               acq_func=acqFunc)
    try:
        imp = best_estimator.named_steps['mlp'].coefs_
        print(str(imp))
    except AttributeError:
        print("How to get the feature importances for MLP?")
    return best_estimator


def convertLine(line, func, first=True, last=False):
    line = line.replace("]", "")
    line = line.replace("'", "")
    line = line.replace("[", "")
    line = line.replace("\n", "")
    line = line.split(",")
    features = line
    first_line = None
    last_line = None
    if first:
        first_line = str(features[0])
        features = features[1:len(features)]
    if last:
        features = features[0:len(features) - 1]
        last_line = str(features[len(features) - 1])
    features = [func(x) for x in features]
    return (features, first_line, last_line)


def extractY(line, Y_dict):
    line = line.replace("]", "")
    line = line.replace("[", "")
    line = line.replace("\n", "")
    line = line.split(",")
    Y_dict[str(line[0])] = (line[1], line[2])


# Need to process X and Y at the same time to match them up.
# Missing ids in y should be marked as unmapped.


def getX(filelocation):
    if filelocation.endswith('.gz'):
        X_fd = gzip.open(filelocation, 'rb')
    else:
        X_fd = open(filelocation, 'r')
    feature_labels = X_fd.readline().strip().replace("]", "").replace(
        "[", "").replace("'", "").split(",")
    feature_labels = feature_labels[1:len(feature_labels)]
    X = [convertLine(line, float, first=True, last=False) for line in X_fd]
    X_ids = []
    X_features = []
    for item in X:
        X_features.append(item[0])
        X_ids.append(item[1])
    sys.stderr.write(str(X_ids[0:3]) + "\n")
    sys.stderr.write(str(X_features[0:3]) + "\n")
    sys.stdout.write("Checking the X matrix.\n")
    sys.stdout.write(str(checkNumericMatrix(X_features)) + "\n")
    sys.stdout.write(str(checkNumericMatrix(X_features)) + "\n")
    return np.array(X_features, dtype='float32'), X_ids, feature_labels


def getY(filelocation):
    Y_fd = open(filelocation, 'r')
    Y_dict = {}
    for line in Y_fd:
        extractY(line, Y_dict)
    #Y = [convertLine(line, int, first = True, last = True) for line in Y_fd]
    #Y_mod = [label[0] for label in Y]
    sys.stderr.write(str(list(Y_dict.keys())[0:3]) + "\n")
    return Y_dict


def main():
    now = datetime.datetime.now()
    usage = "usage: %prog [options] <feature_file> <labels_file> "
    description = ""
    p = optparse.OptionParser(usage=usage, description=description)
    sys.stdout.write("The AlignmentTypeML was started at %s\n" % (str(now)))
    p.add_option(
        '--n_jobs',
        '-n',
        help=
        'The number of jobs (parallel processes) to use to do machine learning. [default: %default]',
        default='4')
    p.add_option(
        '--datasize',
        '-s',
        help=
        'The number of rows of the data to do machine learning. [default: %default]',
        default=None)
    p.add_option(
        '--testsize',
        '-t',
        help=
        'The number of rows of the data to calculate test set error. [default: %default]',
        default=100000)
    p.add_option(
        '--classifier',
        '-c',
        help=
        'Specify the classifier to use.  Choices are "RF", "GB", "LR", "MLP", "SVC", "Ensemble", "Random".  [default: %default]',
        default='RF')
    p.add_option(
        '--write_data',
        '-w',
        help=
        'Write the training and the test data to a file with prefix given in this option. [default: %default]',
        default=None)
    p.add_option('--bayesOpt',
                 '-b',
                 help='Use Bayesian optimization.',
                 action='store_true',
                 default=False)
    p.add_option(
        '--acqFunc',
        '-a',
        help=
        'Choose the aquisition function for Bayesian optimization.  The bayesOpt parameter must be True. Choices are "LCB", "EI", "PI", "gp_hedge".  [default: %default]',
        default="gp_hedge")
    p.add_option(
        '--numIter',
        '-i',
        help=
        'The number of iterations (samples) for Bayesian optimization.  [default: %default]',
        default=25)
    p.add_option('--save_dir',
                 '-d',
                 help='The directory to save models too.',
                 default='./')
    options, args = p.parse_args()
    if len(args) == 0:
        p.print_help()
        return
    if len(args) != 2:
        p.error("There must be two files given in the arguments.")
    if not os.path.exists(args[0]) or not os.path.exists(args[1]):
        p.error("One of the files in the arguments could not be found.")
    sys.stdout.write(
        "Executing AlignmentTypeML on feature file {0} and label file {1}.  There will be {2} processes.  The training data size is {3}, and the test size is {4}.\n"
        .format(args[0], args[1], options.n_jobs, options.datasize,
                options.testsize))
    sys.stdout.write(
        "Classifier: {0}\nBayesian Optimization: {1}, {2}, {3}.\n".format(
            options.classifier, options.bayesOpt, options.acqFunc,
            options.numIter))
    sys.stdout.flush()
    doML(getX(args[0]), getY(args[1]), int(options.n_jobs), options.classifier,
         int(options.datasize), int(options.testsize),
         options.write_data, options.bayesOpt, options.acqFunc,
         int(options.numIter), options.save_dir)
    later = datetime.datetime.now()
    sys.stdout.write(
        "The AlignmentTypeML was started at %s and took %s time.\n" %
        (str(now), str(later - now)))


if __name__ == "__main__":
    main()
