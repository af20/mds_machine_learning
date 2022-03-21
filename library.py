import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd

def libg_get_if_distibution_is_normal(sample1):
  '''
    Test whether a sample differs from a normal distribution.
      This function tests the null hypothesis that a sample comes from a normal distribution. 
      It is based on D’Agostino and Pearson’s [1], [2] test that combines skew and kurtosis to produce an omnibus test of normality.
  '''
  from scipy import stats

  if len(sample1) < 10:
    return None

  stat,pvalue = stats.normaltest(sample1)
  pvalue_adj = max(0.000000001, pvalue) # 1 miliardo
  score = int(1/pvalue_adj)

  min_score_for_normality_of_distribution = 100
  if score >= min_score_for_normality_of_distribution:
    return False
  else:
    return True



def unknown_imputer(X):
  missing_value = 'nan'

  X = X.values
  X = np.array([[str(x[0])] for x in X])
  unique_values, count = np.unique(X,return_counts=True)
  num_nan = count[unique_values == missing_value]
  counting = count[unique_values != missing_value]
  values = unique_values[unique_values != missing_value]
  X_new = X.copy()
  freq = counting / np.sum(counting)
  X_new[X_new == missing_value] = np.random.choice(values,size=num_nan,p=freq)
  return X_new




def my_Map_function(X):
  v_unique_values, count = np.unique(X, return_counts=True)
  v_unique_values = list(v_unique_values)
  new_X = []
  for x in X:
    x = x[0]
    index = v_unique_values.index(x)
    new_X.append(index)
  final_X = np.array([[x] for x in new_X])
  return final_X
  

def my_int_transformer(X):
  new_X = []
  for x in X:
    x = x[0]
    if '<' in x:
      x = '0'
    elif '>' in x:
      x = '21'
    x = [int(x)]
    new_X.append(x)
  new_X = np.array(new_X)
  return new_X



def my_int_transformer_last_njob(X):
  # unique_values ['1' '>4' 'never' '4' '3' '2' nan]             LEN 7  
  new_X = []
  for x in X:
    x = x[0]
    if '>' in x:
      x = '5'
    elif x == 'never':
      x = '0'
    x = [int(x)]
    new_X.append(x)
  new_X = np.array(new_X)
  return new_X




def compute_model_stats(y_real, y_predicted, label):
  from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve

  accuracy = round(accuracy_score(y_real, y_predicted),2) # accuracy = round(np.sum(y_predicted == y_real)/len(y_real),2)
  precision = round(precision_score(y_real, y_predicted),2)
  recall = round(recall_score(y_real, y_predicted),2)
  F1_score = round(f1_score(y_real, y_predicted),2)
  print('  ' + label + '  |  accuracy:', accuracy, '     precision:', precision, '    recall:', recall, '    f1_score:', F1_score)
  return accuracy, precision, recall, F1_score




def plot_precision_recall_curve(y_train, v_target):
  from sklearn.metrics import precision_recall_curve
  import matplotlib.pyplot as plt

  prec, recall, soglia = precision_recall_curve(y_train, v_target)

  fig_prc = plt.figure(figsize=(16,9))
  ax = fig_prc.add_subplot()
  ax.plot(soglia, prec[:-1], 'r', label = 'precision')
  ax.plot(soglia, recall[:-1], 'b', label = 'recall')
  ax.legend(fontsize=20)
  plt.show()
