import os, sys, inspect

from ETL import do_ETL
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd

# Data preprocessing and trasformation (ETL)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, FunctionTransformer, Binarizer, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

def do_ETL():
  df = pd.read_csv('data/breast-cancer.csv')

  v_target = df['diagnosis'].map({'B': 0, 'M': 1}).values
  df.drop(columns=['id','diagnosis'], inplace = True)

  data_preprocessing = ColumnTransformer([
        ('scaler',StandardScaler(), ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'])
      ],
      remainder = 'passthrough'
  )

  feature_matrix = data_preprocessing.fit_transform(df)
  return feature_matrix, v_target
