import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Data management
import pandas as pd

# Data preprocessing and trasformation (ETL)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, FunctionTransformer, Binarizer, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt


# Math and Stat modules
import numpy as np

from library import *

'''
  DATASET 
    https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists
  COLUMNS
    enrollee_id             DROP                                                        0
    city                    [city_103, city_21, Other]  ==> ONE-HOT                     0   .. troppe colonne 123
    city_development_index  (già scalato, è tra 0.45 e 0.95)  ==> già OK                -1 
    gender                  [Male, Fem, Other, null] ==> uknImputer, ONE-HOT            1   (4)
    relevent_experience     [Has.., No..]   ==> MAP                                     2
    enrolled_university     [no_enr, full_t, Oth] ==> ONE-HOT                           1   (4)
    education_level         [Graduate, Mast, Oth] ==> uknImputer, ONE-HOT               1   (6)
    major_discipline        [STEM, Oth, null] ==> uknImputer, ONE-HOT                   1   (7)
    experience              [>20, 5, Oth]  ==> int+uknImp, scalare                      1   (23) ==> TODO
    company_size            [50-99, Oth, null] ==> uknImputer, ONE-HOT                  1   (9)
    company_type            [Pvt Ltd, Oth, null]  ==> uknImputer, ONE-HOT               1   (7)
    last_new_job            [1, >4, Oth] ==> uknImputer, ONE-HOT                        1   (7)
    training_hours          [int] ==> da Scalare                                        3
    target
'''


def do_ETL():
  N_DATA, SEED = 500, 0

  df = pd.read_csv('data/aug_train.csv')
  np.random.seed(SEED)
  rdm = np.random.randint(0, df.shape[0]-1, N_DATA)
  df = df.iloc[rdm]


  df['company_size'] = [str(x) for x in df['company_size'].tolist()]

  v_target = df['target'].tolist()
  df.drop(columns=['target', 'enrollee_id', 'city'], inplace=True)

  experience_PP = Pipeline([
    ('imputer', FunctionTransformer(unknown_imputer)),
    ('int_transformer', FunctionTransformer(my_int_transformer)),
    ('scaler', RobustScaler())
  ])

  last_njob_PP = Pipeline([
    ('imputer', FunctionTransformer(unknown_imputer)),
    ('int_transformer', FunctionTransformer(my_int_transformer_last_njob)),
    ('scaler', RobustScaler())
  ])

  one_hot_PP = Pipeline([
    ('imputer', FunctionTransformer(unknown_imputer)),
    ('one_hot', OneHotEncoder())
  ])


  scale_not_normal_PP = Pipeline([
    ('scaler', RobustScaler()) # is_normal = libg_get_if_distibution_is_normal(df['training_hours'].tolist()) ==> False
  ])

  rel_exp_PP = Pipeline([
    ('imputer', FunctionTransformer(unknown_imputer)),
    ('ordinal_enc', OrdinalEncoder(categories=[['Has relevent experience', 'No relevent experience']]))
  ])
  # OrdinalEncoder(categories=[['M','F']]), ['Gender']),       ['Has relevent experience' 'No relevent experience']


  data_preprocessing = ColumnTransformer(
    [
      ('relv_exp', rel_exp_PP, ['relevent_experience']), # TODO ORDINAL ENCODER !!! df['relevent_experience'] = df['relevent_experience'].map({'No relevent experience': 0, 'Has relevent experience': 1}).tolist()
      
      ('exp', experience_PP, ['experience']),
      ('disc', one_hot_PP, ['major_discipline']),
      ('csize', one_hot_PP, ['company_size']),
      ('ctype', one_hot_PP, ['company_type']),

      ('gend', one_hot_PP, ['gender']),
      ('enr_univ', one_hot_PP, ['enrolled_university']),
      ('edu_lvl', one_hot_PP, ['education_level']),

      ('last_nj', last_njob_PP, ['last_new_job']),

      ('thours', scale_not_normal_PP, ['training_hours'])
    ],
      remainder = 'passthrough'
  )

  # Calcolo la feature matrix
  Matrix = data_preprocessing.fit_transform(df)
  
  try:
    Matrix = Matrix.toarray()
  except:
    pass

  return Matrix, v_target
