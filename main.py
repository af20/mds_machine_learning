import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


# Data management
import pandas as pd

# Math and Stat modules
import numpy as np
from scipy.stats import sem
from random import choice

# Supervised Learning
from sklearn.model_selection import train_test_split

# Visualization
import matplotlib.pyplot as plt
import matplotlib as mpl


from ETL import do_ETL, lib_ETL_manage_imbalance
from classes import *
from datetime import date, datetime

T_START = datetime.now().replace(microsecond=0)
print('    T_START', T_START)



# .................... INPUT VALUES ....................
manage_imbalanced = None # [None, 'over', 'under']  
# To manage imbalanced classes (only Training set!)

data_percentage = 100 # Percentage of dataset to be used 
# (total rows: 20k. E.G. To use 2k rows, write data_percentage=10)
print('manage_imbalanced', manage_imbalanced, '     data_percentage', data_percentage)
# ..............................................

# ETL  |  Train-Test Split (gestione opzionale delle classi sbilanciate)
Matrix, v_target = do_ETL(data_percentage)
X_train, X_test, y_train, y_test = train_test_split(Matrix, v_target, test_size = 0.2, random_state = 42)
X_train, y_train = lib_ETL_manage_imbalance(manage_imbalanced, X_train, y_train)

# I Modelli di ML (Percettrone, Logistic Regression, SVM, Albero di Decisione, Random Forest)
# Per ciascun modello visualizziamo:
# - le statistiche (accuracy, precision, recall, F1 score)
# - le learning curves

'''
PERCEPTRON = c_Perceptron(X_train, X_test, y_train, y_test)
PERCEPTRON.plot_learning_curves()

LOG_REGR = c_Logistic_Regression(X_train, X_test, y_train, y_test, treshold=0.5)
LOG_REGR.plot_learning_curves()

LINEAR_SVM = c_linear_SVM(X_train, X_test, y_train, y_test)
LINEAR_SVM.plot_learning_curves()

NON_LINEAR_SVM = c_non_linear_SVM(X_train, X_test, y_train, y_test, 'poly', {'degree': 5, 'coef0': 10})
NON_LINEAR_SVM.plot_learning_curves_POLY()
NON_LINEAR_SVM = c_non_linear_SVM(X_train, X_test, y_train, y_test, 'rbf', {'gamma': 1, 'C': 0.01})
#NON_LINEAR_SVM.plot_learning_curves_RBF()

TREE = c_decision_tree(X_train, X_test, y_train, y_test)
#TREE.plot_learning_curve()

RF = c_random_forest(X_train, y_train)
RF.plot_learning_curve()

'''

T_END = datetime.now().replace(microsecond=0)
seconds = int(((T_END - T_START).total_seconds()))

print('    T_start:', T_START, '      |    T_end', T_END, '  |   done in', seconds, 'seconds')
