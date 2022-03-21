import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Data management
import pandas as pd
import pickle

# Data preprocessing and trasformation (ETL)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, FunctionTransformer, Binarizer, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_iris, make_moons, make_classification


# Math and Stat modules
import numpy as np
from scipy.stats import sem, randint
from random import choice

# Supervised Learning
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, StratifiedKFold, RepeatedKFold, ShuffleSplit, StratifiedShuffleSplit, learning_curve, validation_curve, cross_validate
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve
from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# Hyperparameter Optimization
#from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils.fixes import loguniform

# Unsupervised Learning
# Clustering algorithms and evaluation metrics
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram
from sklearn.neighbors import NearestNeighbors


# Visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

import seaborn as sns
from sklearn.tree import export_graphviz


from library import *


class c_Perceptron:
  def __init__(self, X_train, X_test, y_train, y_test):
    perceptron = Perceptron()
    self.X_train, self.X_test = X_train, X_test
    self.y_train, self.y_test = y_train, y_test
    # perceptron.fit(X_train, y_train) # accuracy = np.sum(predicted_test == y_test) / len(y_test)
    #v_accuracy = cross_val_score(perceptron, X_train, y_train, cv = 5) # perceptron_score     # accuracy = np.mean(v_accuracy)
    #print('v_accuracy', v_accuracy, '    accuracy', accuracy)

    y_train_predicted = cross_val_predict(perceptron, X_train, y_train, cv = 10) # 15326 len(y_train_predicted)
    self.train_accuracy, self.train_precision, self.train_recall, self.train_F1_score = compute_model_stats(y_train, y_train_predicted, 'Perceptron (Train)')
    #C = confusion_matrix(y_train, y_train_predicted) #self.accuracy = round((C[0][0]+C[1][1]) / (sum(C[0])+sum(C[1])),2)

    y_test_predicted = cross_val_predict(perceptron, X_test, y_test, cv = 10) # 3832 len(y_test_predicted)
    self.test_accuracy, self.test_precision, self.test_recall, self.test_F1_score = compute_model_stats(y_test, y_test_predicted, 'Perceptron (Test) ')

  
  def plot_learning_curves(self):
    Cs = [0.01, 0.1, 1, 10] # definire un insieme di valori di C tenendo in considerazione le precedenti osservazioni sul suo effetto 
    fig = plt.figure(figsize=(18,6))
    for i, c in enumerate(Cs):
      print('Training n°', i)
      PERC = Perceptron()

      train_sizes, train_scores, test_scores = learning_curve(PERC, X = self.X_test, y = self.y_test, train_sizes=np.linspace(0.1,1,10), cv = 5, n_jobs=-1, shuffle = True)

      train_mean = np.mean(train_scores, axis=1)
      train_std = np.std(train_scores, axis=1)
      test_mean = np.mean(test_scores, axis=1)
      test_std = np.std(test_scores, axis=1)

      ax = fig.add_subplot(150+(i+1))
      ax.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
      ax.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
      ax.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
      ax.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
      ax.grid()
      ax.set_ylim((0.2,1))
      ax.set_xlabel('Dimensione del training set')
      ax.set_ylabel('Accuracy')# C = ' + str(c))
      ax.legend(loc='lower right')
    plt.show()





class c_Logistic_Regression:
  def __init__(self, X_train, X_test, y_train, y_test, treshold=0.9, max_iter=1000):
    self.X_train, self.X_test = X_train, X_test
    self.y_train, self.y_test = y_train, y_test
    
    y_train_predicted = self.predict_Y(X_train, y_train, treshold)
    self.y_train_predicted = y_train_predicted
    # print('y_train_predicted', y_train_predicted) # len: 15326       [False False  True ... False  True False]
    self.train_accuracy, self.train_precision, self.train_recall, self.train_F1_score = compute_model_stats(y_train, y_train_predicted, 'Log.Reg. (Train)')

    y_test_predicted = self.predict_Y(X_test, y_test, treshold)
    self.y_test_predicted = y_test_predicted
    # print('len  y_test_predicted', len(y_test_predicted)) # len: 3832
    self.test_accuracy, self.test_precision, self.test_recall, self.test_F1_score = compute_model_stats(y_test, y_test_predicted, 'Log.Reg. (Test) ')

    '''
    train_sizes, train_scores, test_scores = learning_curve(logit_cls,
                                                            X=X,
                                                            y=Y,
                                                            train_sizes= [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                                            cv = 10,
                                                            n_jobs = -1,
                                                            shuffle = True
                                                        )
    y_pred = logreg.predict(X_test)
    '''
    # NOTE COSA è prec? (ha 3918 valori, come X_train)
    # print('prec', len(prec)) # len: 15318         [0.24959217 0.2495432  0.24955949 ... 1.         1.         1.        ]
    # print('recall', len(recall)) # len: 15318     [1.00000000e+00 9.99738562e-01 9.99738562e-01 ... 5.22875817e-04   2.61437908e-04 0.00000000e+00]
    # print('soglia', len(soglia)) # len: 15318       [-3.17510098 -3.15179604 -3.14803856 ...  1.84557772  2.04469964   2.240481  ]
    

  def predict_Y(self, X, Y, treshold):
    logit_cls = LogisticRegression(max_iter = 1000)
    y_predicted_score = cross_val_predict(logit_cls, X, Y, cv = 10, method='decision_function') # [-4.53289277 -3.72660414 -2.6409855  ... -3.53140
    prec, recall, soglia = precision_recall_curve(Y, y_predicted_score)   #for i in range(len(prec)):  print(soglia[i], '   p:', prec[i], '   r:', recall[i])
    soglia_prec = soglia[np.argmax(prec >= treshold)] #   SOGLIA=0.9   #  soglia_prec 0.23674 ---- massimo valore dell'array soglia   #  
    y_predicted_score = y_predicted_score >= soglia_prec # [False False False ... False False False]
    return y_predicted_score



  def plot_learning_curves(self):
    Cs = [0.01, 0.1, 1, 10] # definire un insieme di valori di C tenendo in considerazione le precedenti osservazioni sul suo effetto 
    fig = plt.figure(figsize=(18,6))
    for i, c in enumerate(Cs):
      print('Training n°', i)

      logit_cls = LogisticRegression(max_iter = 1000)
      train_sizes, train_scores, test_scores = learning_curve(logit_cls, X = self.X_test, y = self.y_test, train_sizes=np.linspace(0.1,1,10), cv = 5, n_jobs=-1, shuffle = True)

      train_mean = np.mean(train_scores, axis=1)
      train_std = np.std(train_scores, axis=1)
      test_mean = np.mean(test_scores, axis=1)
      test_std = np.std(test_scores, axis=1)

      ax = fig.add_subplot(150+(i+1))
      ax.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
      ax.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
      ax.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
      ax.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
      ax.grid()
      ax.set_ylim((0.2,1))
      ax.set_xlabel('Dimensione del training set')
      ax.set_ylabel('Accuracy')# C = ' + str(c))
      ax.legend(loc='lower right')
    plt.show()





class c_linear_SVM:
  def __init__(self, X, X_test, Y, Y_test):
    self.X = X
    self.Y = Y
    svm_cls = LinearSVC(C=1, max_iter=50000)
    svm_cls.fit(X, Y)
    y_train_predicted = svm_cls.predict(X)
    # print('y_train_predicted', y_train_predicted) # len: 15326    [0. 0. 0. ... 0. 1. 0.]
    self.train_accuracy, self.train_precision, self.train_recall, self.train_F1_score = compute_model_stats(Y, y_train_predicted, 'SVM linear (Train)')

    y_test_predicted = svm_cls.predict(X_test)
    self.test_accuracy, self.test_precision, self.test_recall, self.test_F1_score = compute_model_stats(Y_test, y_test_predicted, 'SVM linear (test) ')


    #train_sizes, train_scores, test_scores = learning_curve(svm_cls, X = self.X, y = self.Y, train_sizes=np.linspace(0.1,1,10), cv = 5, n_jobs=-1, shuffle = True)
    # print('train_sizes', train_sizes) # len:10   [ 1226  2452  3678  4904  6130  7356  8582  9808 11034 12260]
    # print('train_scores', train_scores) # len: 10x5 (sizes * cv) [[0.77732463 0.76508972 0.78874388 0.7634584  0.75448613], [0.77039152 0.77161501 0.76468189 0.76386623 0.76060359], ...]
    # print('test_scores', test_scores) # uguale a train_scores



  def plot_learning_curves(self):
    Cs = [0.01, 0.1, 1, 10] # definire un insieme di valori di C tenendo in considerazione le precedenti osservazioni sul suo effetto 
    fig = plt.figure(figsize=(18,6))
    for i, c in enumerate(Cs):
      print('Training SVM per C =', c)
      svm_cls = LinearSVC(C = c, max_iter=50000)

      train_sizes, train_scores, test_scores = learning_curve(svm_cls, X = self.X, y = self.Y, train_sizes=np.linspace(0.1,1,10), cv = 5, n_jobs=-1, shuffle = True)

      train_mean = np.mean(train_scores, axis=1)
      train_std = np.std(train_scores, axis=1)
      test_mean = np.mean(test_scores, axis=1)
      test_std = np.std(test_scores, axis=1)

      ax = fig.add_subplot(150+(i+1))
      ax.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
      ax.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
      ax.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
      ax.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
      ax.grid()
      ax.set_ylim((0.2,1))
      ax.set_xlabel('Dimensione del training set')
      ax.set_ylabel('Accuracy C = ' + str(c))
      ax.legend(loc='lower right')
    plt.show()













'''
  Funzioni per plottare
'''
def plot_dataset(X, Y, axes):
  plt.plot(X[:, 0][Y==0], X[:, 1][Y==0], "bs")
  plt.plot(X[:, 0][Y==1], X[:, 1][Y==1], "g^")
  plt.axis(axes)
  plt.grid(True, which='both')
  plt.xlabel(r"$x_1$", fontsize=20)
  plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
  plt.figure(figsize=(12, 4))


def plot_predictions(clf, axes):
  x0s = np.linspace(axes[0], axes[1], 100)
  x1s = np.linspace(axes[2], axes[3], 100)
  x0, x1 = np.meshgrid(x0s, x1s)
  X = np.c_[x0.ravel(), x1.ravel()]
  y_pred = clf.predict(X).reshape(x0.shape)
  y_decision = clf.decision_function(X).reshape(x0.shape)
  plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
  plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)






class c_non_linear_SVM:
  def __init__(self, X, X_test, Y, Y_test, kernel, D):
    assert kernel in ['poly', 'rbf'], "Wrong kernel modality"

    if kernel == 'poly':
      SVM = SVC(kernel="poly", degree=D['degree'], coef0=D['coef0'])
    elif kernel == 'rbf':
      SVM = SVC(kernel="rbf", gamma=D['gamma'], C=D['C'])
    '''param_grid = [
          {'kernel': ['poly'], 'degree': [1, 2, 3], 'coef0': [1, 10, 50]}
          {'kernel': ['rbf'], 'gamma': [.1, 5, 10], 'C': [0.1, 1, 1000]},
      ]'''

    self.SVM = SVM
    self.X = X
    self.Y = Y

    SVM.fit(X, Y)
    y_train_predicted = SVM.predict(X)
    self.train_accuracy, self.train_precision, self.train_recall, self.train_F1_score = compute_model_stats(Y, y_train_predicted, 'SVM non-linear('+kernel+') ' + str(D) +' (Train)')
    y_test_predicted = SVM.predict(X_test)
    self.test_accuracy, self.test_precision, self.test_recall, self.test_F1_score = compute_model_stats(Y_test, y_test_predicted, 'SVM non-linear('+kernel+')' + str(D) +' (test) ')


  def do_grid_search(self, X, Y):
    param_grid = [
        {'kernel': ['rbf'], 'gamma': [.1, 5, 10], 'C': [0.1, 1, 1000]},
        {'kernel': ['poly'], 'degree': [1, 3, 5, 10, 20], 'coef0': [1, 10, 50]}
    ]
    '''
            {'kernel': ['rbf'], 'gamma': [.1, 5, 10], 'C': [0.1, 1, 1000]},
            {'kernel': ['poly'], 'degree': [1, 2, 3], 'coef0': [1, 10, 50]}

    '''
    svm_clf = SVC()

    grid_search = GridSearchCV(estimator=svm_clf,
                              param_grid = param_grid,
                              cv = 5,
                              scoring = 'f1',
                              n_jobs = -1
    )

    grid_search.fit(X, Y)
    print('best params:', grid_search.best_params_,'   Best: {}'.format(grid_search.best_score_), '   best_estimator', grid_search.best_estimator_)
    results = grid_search.cv_results_
    results = pd.DataFrame(results)[['mean_test_score','params']]
    results.sort_values(by='mean_test_score',ascending=False,inplace=True)
    # The regressor.best_score_ is the average of r2 scores on left-out test folds for the best parameter combination.
    #print(results)



  def plot_learning_curves_RBF(self):
    gamma1, gamma2 = 0.1, 2
    C1, C2 = 0.01, 5
    hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

    train_sizes, train_means, test_means, test_stds, train_stds = [],[],[],[],[]
    for gamma, C in hyperparams:
      rbf_kernel_svm_clf = SVC(kernel="rbf", gamma = gamma, C = C)
      train_size, train_scores, test_scores = learning_curve(rbf_kernel_svm_clf,
                                                        X=self.X,
                                                        y=self.Y,
                                                        train_sizes=np.linspace(0.1,1.0,10),
                                                        cv=5,
                                                        n_jobs=-1)
      print('fatto {}, {}'.format(gamma,C))
      train_means.append(np.mean(train_scores, axis=1))
      train_stds.append(np.std(train_scores, axis=1))
      test_means.append(np.mean(test_scores, axis=1))
      test_stds.append(np.std(test_scores, axis=1))
      train_sizes.append(train_size)

    fig= plt.figure(figsize=(12, 8))
    for i in range(4):
      ax = fig.add_subplot(221+i)
      ax.plot(train_sizes[i], train_means[i], color='blue', marker='o', markersize=5, label='Training accuracy')
      ax.fill_between(train_sizes[i], train_means[i] + train_stds[i], train_means[i] - train_stds[i], alpha=0.15, color='blue')
      ax.plot(train_sizes[i], test_means[i], color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
      ax.fill_between(train_sizes[i], test_means[i] + test_stds[i], test_means[i] - test_stds[i], alpha=0.15, color='green')
      ax.grid()
      ax.set_ylim((0.4,1))
      ax.set_ylabel('Accuracy')
      ax.legend(loc='lower right')
      ax.set_title(r"$\gamma={}, C={}$".format(*hyperparams[i]), fontsize=18)
    plt.show()



  def plot_learning_curves_POLY(self):
    dg1, dg2 = 1, 5 # degree
    C1, C2 = 1, 10 # 0.01, 5 # coeff0
    hyperparams = (dg1, C1), (dg1, C2), (dg2, C1), (dg2, C2)

    train_sizes, train_means, test_means, test_stds, train_stds = [],[],[],[],[]
    for Degree, Coef0 in hyperparams:
      kernel_svm_clf = SVC(kernel="poly", degree = Degree, coef0 = Coef0)
      train_size, train_scores, test_scores = learning_curve(kernel_svm_clf,
                                                        X=self.X,
                                                        y=self.Y,
                                                        train_sizes=np.linspace(0.1,1.0,10),
                                                        cv=5,
                                                        n_jobs=-1)
      print('fatto {}, {}'.format(Degree, Coef0))
      train_means.append(np.mean(train_scores, axis=1))
      train_stds.append(np.std(train_scores, axis=1))
      test_means.append(np.mean(test_scores, axis=1))
      test_stds.append(np.std(test_scores, axis=1))
      train_sizes.append(train_size)

    fig= plt.figure(figsize=(12, 8))
    for i in range(4):
      ax = fig.add_subplot(221+i)
      ax.plot(train_sizes[i], train_means[i], color='blue', marker='o', markersize=5, label='Training accuracy')
      ax.fill_between(train_sizes[i], train_means[i] + train_stds[i], train_means[i] - train_stds[i], alpha=0.15, color='blue')
      ax.plot(train_sizes[i], test_means[i], color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
      ax.fill_between(train_sizes[i], test_means[i] + test_stds[i], test_means[i] - test_stds[i], alpha=0.15, color='green')
      ax.grid()
      ax.set_ylim((0.4,1))
      ax.set_ylabel('Accuracy')
      ax.legend(loc='lower right')
      ax.set_title('Degree: ' + str(hyperparams[i][0]) + ' |  Coef0: ' + str(hyperparams[i][1]))
      #ax.set_title(r"$\Degree={}, coef0={}$".format(*hyperparams[i]), fontsize=18)
    plt.show()






class c_decision_tree:
  def __init__(self, X, X_test, Y, Y_test):
    self.X = X
    self.Y = Y
    tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42) # min_samples_leaf
    tree_clf.fit(X, Y)
    self.tree_clf = tree_clf
    
    y_train_predicted = tree_clf.predict(X)
    self.train_accuracy, self.train_precision, self.train_recall, self.train_F1_score = compute_model_stats(Y, y_train_predicted, 'Tree (Train)')

    y_test_predicted = tree_clf.predict(X_test)
    self.test_accuracy, self.test_precision, self.test_recall, self.test_F1_score = compute_model_stats(Y_test, y_test_predicted, 'Tree (test) ')



  def plot_learning_curve(self):
    min_leaf = [5, 10, 100, 200, 350]

    train_sizes, train_means, test_means, test_stds, train_stds = [],[],[],[],[]
    for mlf in min_leaf:
      dt_mlf = DecisionTreeClassifier(min_samples_leaf=mlf, random_state=42, max_depth=15)
      train_size, train_scores, test_scores = learning_curve(dt_mlf,
                                                          X=self.X,
                                                          y=self.Y,
                                                          train_sizes=np.linspace(0.1,1.0,10),
                                                          cv=10,
                                                          n_jobs=-1)
      print('fatto {}'.format(mlf))
      train_means.append(np.mean(train_scores, axis=1))
      train_stds.append(np.std(train_scores, axis=1))
      test_means.append(np.mean(test_scores, axis=1))
      test_stds.append(np.std(test_scores, axis=1))
      train_sizes.append(train_size)

    fig= plt.figure(figsize=(12, 8))
    for i in range(5):
      ax = fig.add_subplot(231+i)
      ax.plot(train_sizes[i], train_means[i], color='blue', marker='o', markersize=5, label='Training accuracy')
      ax.fill_between(train_sizes[i], train_means[i] + train_stds[i], train_means[i] - train_stds[i], alpha=0.15, color='blue')
      ax.plot(train_sizes[i], test_means[i], color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
      ax.fill_between(train_sizes[i], test_means[i] + test_stds[i], test_means[i] - test_stds[i], alpha=0.15, color='green')
      ax.grid()
      ax.set_ylim((0.4,1))
      ax.set_ylabel('Accuracy')
      ax.legend(loc='lower right')
      ax.set_title(r"min_sam_leaf: {}".format(min_leaf[i]), fontsize=12)
    plt.show()





class c_random_forest:
  def __init__(self, X, Y):
    '''
    BAGGING + DT
      E' un ensamble di Decision Trees utilizzando metodo Bagging con max_samples = len(training set)
    SOTTO-INSIEME FEATURES x crescita
      Nella crescita dell'albero non vengono considerate tutte le feature per cercare lo splitting migliore ma un sottoinsieme casuale. 
      In questo modo aumento la diversita' degli alberi.

    '''
    self.X = X
    self.Y = Y
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.2, stratify=Y)
    self.RF_clf = RandomForestClassifier(n_estimators=250, max_leaf_nodes=64, max_depth=50, n_jobs=-1)#, max_features=10)
    self.RF_clf.fit(self.X_train, self.y_train)

    y_train_predicted = self.RF_clf.predict(X)
    compute_model_stats(Y, y_train_predicted, 'Random Forest (Train)')

    y_test_predicted = self.RF_clf.predict(self.X_test)
    compute_model_stats(self.y_test, y_test_predicted, 'Random Forest (test) ')


    self.et_clf = self.et_clf = ExtraTreesClassifier(n_estimators=250, max_leaf_nodes=64, n_jobs=-1)#, max_features=10)
    self.et_clf.fit(self.X_train, self.y_train)
    y_train_predicted = self.et_clf.predict(X)
    compute_model_stats(Y, y_train_predicted, 'Extra Trees (Train)')
    y_test_predicted = self.et_clf.predict(self.X_test)
    compute_model_stats(self.y_test, y_test_predicted, 'Extra Trees (test) ')


    self.ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=6),
        n_estimators=100,
        algorithm='SAMME.R',
        learning_rate=0.5
    )
    # best params: {'base_estimator__max_depth': 5, 'base_estimator__min_samples_leaf': 20, 'learning_rate': 0.1, 'n_estimators': 250}    Best: 0.4690804542716961    best_estimator AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5,
    '''
      self.ada_clf = AdaBoostClassifier(
          DecisionTreeClassifier(max_depth=6),
          n_estimators=100,
          algorithm='SAMME.R',
          learning_rate=0.5
      )
    '''
    y_train_predicted = self.et_clf.predict(X)
    compute_model_stats(Y, y_train_predicted, 'ADA Boosting (Train)')
    y_test_predicted = self.et_clf.predict(self.X_test)
    compute_model_stats(self.y_test, y_test_predicted, 'ADA Boosting (test) ')

    # named_feat_importance = dict(zip(columns_name, self.RF_clf.feature_importances_)) # l'importanza delle feature   # sorted(named_feat_importance.items(), key=lambda x:x[1], reverse=True)[:10]


  def plot_RF_ET(self):
    scores_rnf = cross_val_score(self.RF_clf, self.X_train, self.y_train, cv=5, scoring='f1', n_jobs=-1)
    scores_et = cross_val_score(self.et_clf, self.X_train, self.y_train, cv=5, scoring='f1',n_jobs=-1)
    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot()
    sns.boxplot(ax = ax,
                data = [scores_rnf, scores_et],
                palette = 'vlag',
                orient = 'h'
              )
    ax.set(yticklabels=['RF','ET'])
    plt.show()




  def plot_learning_curve(self):
    OPT = [None, 2, 5, 10, 30]

    train_sizes, train_means, test_means, test_stds, train_stds = [],[],[],[],[]
    for opt in OPT:
      dt_mlf = RandomForestClassifier(n_estimators=250, max_leaf_nodes=64, max_depth=opt, n_jobs=-1)  # n_estimators=250, max_leaf_nodes=64  max_features=10)   # min_samples_leaf=mlf, random_state=42)
      train_size, train_scores, test_scores = learning_curve(dt_mlf,
                                                          X=self.X,
                                                          y=self.Y,
                                                          train_sizes=np.linspace(0.1,1.0,10),
                                                          cv=10,
                                                          n_jobs=-1)
      print('fatto {}'.format(str(opt)))
      train_means.append(np.mean(train_scores, axis=1))
      train_stds.append(np.std(train_scores, axis=1))
      test_means.append(np.mean(test_scores, axis=1))
      test_stds.append(np.std(test_scores, axis=1))
      train_sizes.append(train_size)

    fig= plt.figure(figsize=(12, 8))
    for i in range(5):
      ax = fig.add_subplot(231+i)
      ax.plot(train_sizes[i], train_means[i], color='blue', marker='o', markersize=5, label='Training accuracy')
      ax.fill_between(train_sizes[i], train_means[i] + train_stds[i], train_means[i] - train_stds[i], alpha=0.15, color='blue')
      ax.plot(train_sizes[i], test_means[i], color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
      ax.fill_between(train_sizes[i], test_means[i] + test_stds[i], test_means[i] - test_stds[i], alpha=0.15, color='green')
      ax.grid()
      ax.set_ylim((0.4,1))
      ax.set_ylabel('Accuracy')
      ax.legend(loc='lower right')
      ax.set_title(r"max_depth: {}".format(OPT[i]), fontsize=12)
    plt.show()


  def ada_boosting(self):
    '''
      ## Boosting
          Sono metodi di ensemble in cui i classificatori sono addestrati in modo sequenziale, ed ogni classificatori corregge il classificatore precedente. 
          I metodi piu' importanti sono *AdaBoost* e *Gradient Boosting*.
      ### AdaBoost
          Un modo per correggere un classificatore e' focalizzari sulle istanze che il predecessore non classifica correttamente, cioe' sulla istanze piu' difficili.
          Per dare piu' importanza alle istanze non classificate correttamente aumento il peso di quelle istanze e addestro un classificatore con le istanze ri-pesate. Il processo viene ripetuto per il numero di classificatori nel pool.
    '''
    scores_ada = cross_val_score(self.ada_clf, self.X_train, self.y_train, cv=5, scoring='f1', n_jobs=-1)
    scores_rnf = cross_val_score(self.RF_clf, self.X_train, self.y_train, cv=5, scoring='f1', n_jobs=-1)
    scores_et = cross_val_score(self.et_clf, self.X_train, self.y_train, cv=5, scoring='f1',n_jobs=-1)
    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot()

    sns.boxplot(ax = ax,
                data = [scores_rnf, scores_et, scores_ada],
                palette = 'vlag',
                orient = 'h'
              )
    ax.set(yticklabels=['RF','ET','ADA'])
    plt.show()



  def ada_boosting_grid_search(self):

    ADA = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

    parameters = {'base_estimator__max_depth':[i for i in range(5,10,2)],
                  'base_estimator__min_samples_leaf':[20],
                  'n_estimators':[250,1000],
                  'learning_rate':[0.1]}

    grid_search = GridSearchCV(estimator=ADA,
                              param_grid = parameters,
                              cv = 5,
                              scoring = 'f1',
                              n_jobs = -1
    )

    grid_search.fit(self.X_train, self.y_train)
    print('best params:', grid_search.best_params_,'   Best: {}'.format(grid_search.best_score_), '   best_estimator', grid_search.best_estimator_)
    results = grid_search.cv_results_
    results = pd.DataFrame(results)[['mean_test_score','params']]
    results.sort_values(by='mean_test_score',ascending=False,inplace=True)
    print(results)



  def gradient_boosting(self):
    '''
    (Non prende dati della classe, è solo un esempio)
        Gradient Boosting
            Similmente ad AdaBoost, Gradient Boosting agisce in maniera sequenziale ma in ogni step 
            il classificatore apprende sugli errori residui del classificatore precedente.
    '''


    def plot_predicts(X_rnd, y_rnd, tree_reg1, tree_reg2, tree_reg3, y2, y3):
      def supp_plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
        x1 = np.linspace(axes[0], axes[1], 500)
        y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
        plt.plot(X[:, 0], y, data_style, label=data_label)
        plt.plot(x1, y_pred, style, linewidth=2, label=label)
        if label or data_label:
          plt.legend(loc="upper center", fontsize=16)
        plt.axis(axes)


      plt.figure(figsize=(11,11))
      plt.subplot(321)
      supp_plot_predictions([tree_reg1], X_rnd, y_rnd, axes=[-0.5, 0.5, -0.1, 0.8], label="$h_1(x_1)$", style="g-", data_label="Training set")
      plt.ylabel("$y$", fontsize=16, rotation=0)
      plt.title("Residui e DT", fontsize=16)

      plt.subplot(322)
      supp_plot_predictions([tree_reg1], X_rnd, y_rnd, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1)$", data_label="Training set")
      plt.ylabel("$y$", fontsize=16, rotation=0)
      plt.title("Predizione Gradient Boosting", fontsize=16)

      plt.subplot(323)
      supp_plot_predictions([tree_reg2], X_rnd, y2, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", style="g-", data_style="k+", data_label="Residuals")
      plt.ylabel("$y - h_1(x_1)$", fontsize=16)

      plt.subplot(324)
      supp_plot_predictions([tree_reg1, tree_reg2], X_rnd, y_rnd, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
      plt.ylabel("$y$", fontsize=16, rotation=0)

      plt.subplot(325)
      supp_plot_predictions([tree_reg3], X_rnd, y3, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", style="g-", data_style="k+")
      plt.ylabel("$y - h_1(x_1) - h_2(x_1)$", fontsize=16)
      plt.xlabel("$x_1$", fontsize=16)

      plt.subplot(326)
      supp_plot_predictions([tree_reg1, tree_reg2, tree_reg3], X_rnd, y_rnd, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
      plt.xlabel("$x_1$", fontsize=16)
      plt.ylabel("$y$", fontsize=16, rotation=0)
      plt.show()

    X_rnd = np.random.rand(100, 1) - 0.5

    y_rnd = 3*X_rnd[:, 0]**2 + 0.05 * np.random.randn(100)
    tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg1.fit(X_rnd, y_rnd)

    y2 = y_rnd - tree_reg1.predict(X_rnd)
    tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg2.fit(X_rnd, y2)

    y3 = y2 - tree_reg2.predict(X_rnd)
    tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg3.fit(X_rnd, y3)

    plot_predicts(X_rnd, y_rnd, tree_reg1, tree_reg2, tree_reg3, y2, y3)






class c_clust:
  def __init__(self, X_train):
    self.X_train = X_train

    km_blob = KMeans(n_clusters = 2,  #3 clusters
                init= 'k-means++',  # random initialization ['random', 'k-means++']
                n_init= 10, # number of re-iterations
                max_iter= 300, # max number of iteration per run
                tol= 1e-4, # minimum increase 10^(-4)
                random_state=0)
    Y_train = km_blob.fit_predict(X_train)

    plt.scatter(X_train[Y_train == 0,0], X_train[Y_train == 0,1], s=50, c='lightgreen',
                marker='s', edgecolor='black', label='cluster 1')
    plt.scatter(X_train[Y_train == 1,0], X_train[Y_train == 1,1],
                s=50, c='orange', marker='o', edgecolor='black', label='cluster 2')
    plt.scatter(km_blob.cluster_centers_[:,0],km_blob.cluster_centers_[:,1],
                s=250, marker='*', c='red', edgecolor='black', label='centroids')
    plt.legend(scatterpoints=1)
    plt.show()


  def plot_distorsion(self):
    distortions = []
    for i in np.arange(1,11):
        km = KMeans(n_clusters= i, 
                    init='k-means++', 
                    n_init=10, 
                    max_iter=300, 
                    random_state=0)
        km.fit(self.X_train)
        distortions.append(km.inertia_)
      
    plt.plot(np.arange(1,11),distortions)
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()






class c_Neural_Network:
  # Artificial Neural Networks
  # MLP: ML Process
  def __init__(self, X_train, Y_train, X_test, Y_test):
    #(X_train_full, y_train_full),(X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    #print(X_train_full.shape, X_test.shape)
    
    from tensorflow import keras
    from sklearn.datasets import fetch_california_housing

    model = keras.models.Sequential([ # modello Sequential, i livelli sn messi in seq.
        #keras.layers.Flatten(input_shape=[28, 28]), # converte l'input originario (matrice NxM) in un vettore
        keras.layers.Dense(30, activation="relu", input_shape=[8]), # livello hidden 1
        keras.layers.Dense(10, activation="relu"), # livello hidden 2
        keras.layers.Dense(1) # perchè classificaz è binaria quindi Dense(1)        - Dense(10) se avessi 10 classi di output
    ])
    #print('model.summary', model.summary())


    ''' Compilazione del modello
      Dopo la creazione di un modello devo specificare la funzione di loss e l'algoritmo di ottimizzazione, 
      ed eventualmente le metriche di performance da utilizzare. 
      Il metodo **compile** viene invocato per questi scopi.
    '''
    model.compile(loss = 'binary_crossentropy', # 'sparse_categorical_crossentropy' SE MULTI-CLASSE,
              optimizer = keras.optimizers.SGD(learning_rate=1e-3), # 'sgd',
              metrics =  ['accuracy']
             )

    #X_valid_part, X_train_part = X_train[:5000] / 255., X_train[5000:] / 255.
    #y_valid_part, y_train_part = Y_train[:5000], Y_train[5000:]

    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)


    '''
      L'oggetto **History** restituito dal metodo fit contiene i parametri di training (**.params**), 
      la lista delle epoche (**.epoch**) e un dict con i valori della loss function e delle metriche di performance sul training e validation set per ogni epoca.
    '''

    
    history = model.fit(X_train, Y_train, 
                        epochs=30, 
                        validation_data=(X_valid, Y_valid)
                      ) # XXX Dà ERRORE QUA
    accuracy = model.evaluate(X_test, Y_test)
    print('accuracy', accuracy)
    X_new = X_test[:3]
    y_pred = model.predict(X_new)

    # SALVO MODELLO (Serializzare): rappresento un mio oggetto in modo che possa essere salvato (checkpoint dello stato)
    '''
    COME FUNZ APPRENDIM e MODIFICA pesi nella NN
      calcolo f.loss 
      Backpropagation: da output a input mi permette di aggiornare i pesi, portandosi dietro il gradiente (se grad = 0 --> nessun update)
        batch, per i k sottoinsiemi su cui ho diviso il dataset, faccio avanti-indietro
        e alla fine dell'epoca mi porta a avere riduzione importante della loss
          (è il cuore addestramento reti neurali: sia dense multi-layer    ||  convuluzionali  ||  ricorrenti)
    SALVATAGGIO BEST-MODEL: quello k ha ottenuto loss + bassa

    NNetw utilizzate oggi xk hanno trovato modo per far sì che gradiente non sia 0 per aggiornamento (problema vanishing gradinent) 2010
      - funzione attivazione
      - varianza layer

    Activation function: introduce la NON linearità
    '''
    return

    print(history.params, history.epoch); pd.DataFrame(history.history).plot(figsize=(12,4)); plt.grid(True); plt.show()

    # Osserviamo un po' di overfitting.
    # Nel caso venga invocato nuovamente il metodo fit, senza ricreare il modello, il processo di training riprende dall'ultimo stato dei parametri.
    print(model.evaluate(X_test, Y_test)) # Per valutare il modello appresso sul test set utilizziamo il metodo **evaluate**.

    # Mediante il metodo **predict** eseguiamo una predizione circa una nuova istanza.
    print(model.predict(X_test[:3]))

    # Se invece voglio ottenere la classe predetta, si utilizza il metodo **predict_classes**.      np.array(class_names)[model.predict_classes(X_test[:3])]

