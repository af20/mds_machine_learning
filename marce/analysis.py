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


from marce.etl import do_ETL
from classes import *
from datetime import date, datetime

T_START = datetime.now().replace(microsecond=0)
print('    T_START', T_START)

Matrix, v_target = do_ETL()
X_train, X_test, y_train, y_test = train_test_split(Matrix, v_target, test_size = 0.2, random_state = 42)
print('UNQ:', np.unique(y_train))
L0 = [x for x in y_train if x == 0]
L1 = [x for x in y_train if x == 1]
print(len(L0), len(L1))

#PERCEPTRON = c_Perceptron(X_train, X_test, y_train, y_test)
#LOG_REGR = c_Logistic_Regression(X_train, X_test, y_train, y_test, treshold=1)

#LINEAR_SVM = c_linear_SVM(X_train, X_test, y_train, y_test)

#NON_LINEAR_SVM = c_non_linear_SVM(X_train, X_test, y_train, y_test, 'poly', {'degree': 5, 'coef0': 10})
#NON_LINEAR_SVM = c_non_linear_SVM(X_train, X_test, y_train, y_test, 'rbf', {'gamma': 1, 'C': 1})

#TREE = c_decision_tree(X_train, X_test, y_train, y_test)

RF = c_random_forest(X_train, y_train)


T_END = datetime.now().replace(microsecond=0)
seconds = int(((T_END - T_START).total_seconds()))

print('    T_start:', T_START, '      |    T_end', T_END, '  |   done in', seconds, 'seconds')


