import re
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500) 

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, cross_val_score

from utils import BERTEmbExtractor, find_best_ccp_aplpha, metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
import joblib
import torch

tqdm.pandas()

data = pd.read_csv('/home/admin01/vadim/classifier/Таблица_данные_опроса_+_часть_из_обогащения.csv')
data = data.drop(columns=['ИНН_x', 'ИНН_y', 'Степень уверенности', 'Кол-во вариантов с таким названием компании'])


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
    ('text_encoder', BERTEmbExtractor(batch_size=32), text_cols),
    ('num_features', 'passthrough', num_cols)
    ], remainder='drop', verbose_feature_names_out=True)

data_prepare = Pipeline([
    ('preproc', data_transformer),
    ('imputer', SimpleImputer()),
    ])

pipe = [Pipeline([
    ('model', ExtraTreesClassifier(class_weight='balanced_subsample', oob_score=True, bootstrap=True, n_jobs=5))
    ])
] * 2

# prepare Xs
X_train = data_prepare.fit_transform(df_train)
X_val = data_prepare.transform(df_val)
X_test = data_prepare.transform(df_test)

# find best hyper params
pipes = []

for i, task in enumerate(tasks):
    print(task)
    print()
    best_ccp = find_best_ccp_aplpha(X_train, y_train[i], X_val, y_val[i])
    
    pipe = Pipeline([
    ('model', ExtraTreesClassifier(class_weight='balanced_subsample', 
                                   oob_score=True, 
                                   bootstrap=True, 
                                   n_jobs=5, 
                                   ccp_alpha=best_ccp))
    ])
    
    print('точность :', cross_val_score(pipe, X_train, y_train[i], n_jobs=1))
    pipe.fit(X_train, y_train[i])
    pipes.append(pipe) 
    print()
    
# compute metrics
metrics(pipes[0], X_val, y_val[0], LE[0])
metrics(pipes[0], X_test, y_test[0], LE[0])
metrics(pipes[1], X_val, y_val[1], LE[1])
metrics(pipes[1], X_test, y_test[1], LE[1])

# save pipes and transformer
joblib.dump(pipes, 'pipelines.joblib')
joblib.dump(data_prepare, 'data_prepare.joblib')
joblib.dump(LE, 'label_encoders.joblib')

print('FINISH')