import ETL

#df['company_size'].astype(str).tolist()
#print(df['company_size'].tolist())
#par = [str(x) for x in df['company_size'].tolist()]
  # print(df.describe(include=['object']))
# df['major_discipline'].value_counts()


# print('Len NULL:', len(df[df['experience'].isnull()]))
#unique_values = pd.unique(df['last_new_job'])
#print('unique_values', unique_values,'            LEN', len(unique_values))
#print(df['training_hours'].value_counts())


# rel_exp_PP = Pipeline([
#   ('imputer', FunctionTransformer(unknown_imputer))
#   #('map', FunctionTransformer(my_Map_function))
# ])


ETL.SimpleImputer