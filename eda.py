try:
    from fancyimpute import IterativeSVD
    from fancyimpute import KNN
    print("Library is already installed.")
except ImportError:
    print("Library is not installed. Proceed with installation.")
    !pip install fancyimpute
    from fancyimpute import IterativeSVD
    from fancyimpute import KNN

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import re

import warnings


base: str
if os.getcwd() == "/kaggle/working":
    base = "/kaggle"
else:
    base = os.path.join(os.getcwd())

def get_full_dir(sub_dir: str) -> str:
    return os.path.join(base, sub_dir)

df_sample_submission: pd.DataFrame = pd.read_csv(get_full_dir('input/playground-series-s3e15/sample_submission.csv'))
df_data: pd.DataFrame = pd.read_csv(get_full_dir('input/playground-series-s3e15/data.csv'), index_col='id')
df_train = df_data[~df_data['x_e_out [-]'].isna()]
df_test = df_data[df_data['x_e_out [-]'].isna()]
df_og: pd.DataFrame = pd.read_csv(get_full_dir('input/predicting-heat-flux/Data_CHF_Zhao_2020_ATE.csv'), index_col='id')

def remove_special_characters(column_name):
    return re.sub(r"[^a-zA-Z0-9_]+", "", column_name)

def remove_special_characters_from_dataframe(df):
    df.columns = [remove_special_characters(col) for col in df.columns]
    return df

df_data = remove_special_characters_from_dataframe(df_data)
df_train = remove_special_characters_from_dataframe(df_train)
df_test = remove_special_characters_from_dataframe(df_test)
df_og = remove_special_characters_from_dataframe(df_og)

train = pd.DataFrame(df_train.isna().sum() * 100 / df_train.count())
train['dataset'] = 'train'

test = pd.DataFrame(df_test.isna().sum() * 100 / df_test.count())
test['dataset'] = 'test'

og = pd.DataFrame(df_og.isna().sum() * 100 / df_og.count())
og['dataset'] = 'original'

df_tmp = pd.concat([train, test, og], axis=0)
df_tmp = df_tmp.replace(np.inf, 100)
df_tmp = df_tmp.rename(columns={0: 'NaN Percentage %', 'dataset': 'Data Set'})
df_tmp['Column'] = df_tmp.index

plt.figure(figsize=(20, 6))
ax = sns.barplot(data=df_tmp, y='NaN Percentage %', x="Column", hue="Data Set", orient='v')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f') + '%',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center',
                xytext = (0, 5),
                textcoords = 'offset points')
plt.xlabel('Column', fontsize=14)
plt.ylabel('NaN Percentage ', fontsize=14)
plt.title('Percentage of Missing Values per Column', fontsize=16)
plt.plot()
