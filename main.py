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


from ETL import do_ETL
from classes import *
from datetime import date, datetime

T_START = datetime.now().replace(microsecond=0)
print('    T_START', T_START)

Matrix, v_target = do_ETL()
X_train, X_test, y_train, y_test = train_test_split(Matrix, v_target, test_size = 0.2, random_state = 42)

PERCEPTRON = c_Perceptron(X_train, X_test, y_train, y_test)
#PERCEPTRON.plot_learning_curves()

LOG_REGR = c_Logistic_Regression(X_train, X_test, y_train, y_test, treshold=0.5)
#LOG_REGR.plot_learning_curves()
LINEAR_SVM = c_linear_SVM(X_train, X_test, y_train, y_test)
#LINEAR_SVM.plot_learning_curves()

NON_LINEAR_SVM = c_non_linear_SVM(X_train, X_test, y_train, y_test, 'poly', {'degree': 5, 'coef0': 10})
NON_LINEAR_SVM = c_non_linear_SVM(X_train, X_test, y_train, y_test, 'rbf', {'gamma': 10, 'C': 1})
#NON_LINEAR_SVM.plot_learning_curves_POLY()

TREE = c_decision_tree(X_train, X_test, y_train, y_test)
#TREE.plot_learning_curve()

RF = c_random_forest(X_train, y_train)
#RF.plot_learning_curve()

T_END = datetime.now().replace(microsecond=0)
seconds = int(((T_END - T_START).total_seconds()))

print('    T_start:', T_START, '      |    T_end', T_END, '  |   done in', seconds, 'seconds')
