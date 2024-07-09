import re
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500) 

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier
from sklearn.impute import KNNImputer

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from classifier.utils import BERTEmbExtractor, metrics
import joblib
import torch

n_pca = 50
n_iter = 400
data_path = '/home/vadim/work/work_level_classifier/data/Таблица_данные_опроса_+_часть_из_обогащения_v_2.csv'


tqdm.pandas()

data = pd.read_csv(data_path)
data = data.drop(columns=['ИНН', 'Степень уверенности', 'Кол-во вариантов с таким названием компании'])


num_cols = ['Уставный капитал, сумма', 'Среднесписочная численность сотрудников', 'Сумма уплаченных налогов за 2020', 'age']

cat_cols = ['Вид экономической деятельности, ОКВЭД', 
            'Доп вид экономической деятельности_1',
            'Доп вид экономической деятельности_2', 
            'Доп вид экономической деятельности_3',
            'Уставный капитал, тип',
            'Тип по ОКОГУ',
            'Категория из реестра СМП'
            ]

tasks = ['Сфера деятельности по Классификатору ФОИР', 'Карьерная ступень по Классификатору ФОИР']

# extract age
data['Дата рождения'] = pd.to_datetime(data['Дата рождения'], format="%d.%m.%Y")
data['age'] = pd.Timestamp('now').year - data['Дата рождения'].dt.year

# work with target
LE = [LabelEncoder(), LabelEncoder()]
data['label_a'] = LE[0].fit_transform(data[tasks[0]])
data['label_b'] = LE[1].fit_transform(data[tasks[1]])

text_cols  = ['Место работы', 'Наименование текущей должности']
cat_cols += ['Пол', 'Регион', 'Страна проживания', 'Уровень образования', 'Федеральный округ']
data[text_cols] = data[text_cols].fillna('Пропущенное значение')

# split data
df = data.copy()

df_test = df[df['Ручная проверка карьерной ступени'].notna()][cat_cols + text_cols + num_cols ]
y_test = [df[df['Ручная проверка карьерной ступени'].notna()]['label_a'], 
          df[df['Ручная проверка карьерной ступени'].notna()]['label_b']]

df = df[df['Ручная проверка карьерной ступени'].isna()]
df_train, df_val = train_test_split(df, test_size=.20)

y_train = [df_train['label_a'], df_train['label_b']]
df_train = df_train[cat_cols + text_cols + num_cols].copy()

y_val = [df_val['label_a'], df_val['label_b']]
df_val = df_val[cat_cols + text_cols + num_cols].copy()

# define pipelines and transformer
data_transformer =  ColumnTransformer([
    ('cat_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
    ('text_encoder', BERTEmbExtractor(device=torch.device('cpu'), batch_size=32, n_features=n_pca), text_cols),    
    ('num_features', 'passthrough', num_cols)
    ], remainder='drop', verbose_feature_names_out=True)

data_prepare = Pipeline([
    ('preproc', data_transformer),
    ('imputer', KNNImputer()),
    ])

# prepare Xs
X_train = data_prepare.fit_transform(df_train)
X_val = data_prepare.transform(df_val)
X_test = data_prepare.transform(df_test)

# find best hyper params

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 110, num = 11)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 20, num = 10)] + [None]
criterion = ['gini', 'entropy', 'log_loss']
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [4, 10, 20, 30, 40]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 6, 8, 10, 20]
# Method of selecting samples for training each tree
max_features = ['sqrt', 'log2', None]
class_weight = ['balanced', 'balanced_subsample', None]
ccp_alpha = [0, 0.01, 0.001, 0.0001, 0.00001, 0.00000001]

# Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'max_features': max_features,
               'class_weight': class_weight,
               'ccp_alpha': ccp_alpha,
               'criterion': criterion
               }

pipes = []

for i, task in tqdm(enumerate(tasks)):
    print(task)
    print()
    
    if i <= 0:
        rsearch = RandomizedSearchCV(
            estimator=ExtraTreesClassifier(n_jobs=5),
            param_distributions=random_grid,
            n_iter=n_iter,
            scoring=['f1_micro', 'accuracy'],
            n_jobs=1,
            cv=3,
            # verbose=2,
            return_train_score=True,
            refit='f1_micro',
        )
        rsearch.fit(np.vstack((X_train, X_val)), np.hstack((y_train[i], y_val[i])))
        pipe = rsearch.best_estimator_
        print(f'Mean cross-validated score of the best_estimator - {rsearch.best_score_}')
        
    else:
        y_pred = pipes[0].predict(X_train)
        
        rsearch = RandomizedSearchCV(
            estimator=ExtraTreesClassifier(n_jobs=5),
            param_distributions=random_grid,
            n_iter=n_iter,
            scoring=['f1_micro', 'accuracy'],
            n_jobs=1,
            cv=3,
            # verbose=2,
            return_train_score=True,
            refit='f1_micro',
        )
        
        rsearch.fit(np.hstack([X_train, pipes[0].predict(X_train).reshape(-1, 1)]), y_train[i])
        print(f'Mean cross-validated score of the best_estimator - {rsearch.best_score_}')
        
        pipe = StackingClassifier(estimators=[('pred_extract', pipes[0])], 
                                  final_estimator=rsearch.best_estimator_, 
                                  cv='prefit',
                                  passthrough=True)
        
        pipe = pipe.fit(np.vstack((X_train, X_val)), np.hstack((y_train[i], y_val[i])))
    
    # print('точность :', cross_val_score(pipe, X_train.copy(), y_train[i], n_jobs=1))
    pipes.append(pipe)
    
# compute metrics
metrics(pipes[0], X_val, y_val[0], LE[0])
metrics(pipes[0], X_test, y_test[0], LE[0])
metrics(pipes[1], X_val, y_val[1], LE[1])
metrics(pipes[1], X_test, y_test[1], LE[1])

# save pipes and transformer
joblib.dump(pipes, '../models/pipelines.joblib')

# data_prepare.steps[0][1].transformers[1][1].to_device(torch.device('cpu'))
joblib.dump(data_prepare, '../models/data_prepare.joblib')
joblib.dump(LE, '../models/label_encoders.joblib')

print('FINISH')