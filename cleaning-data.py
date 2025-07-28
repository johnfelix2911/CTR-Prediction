# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
import numpy as np
import pandas as pd
# -

train_raw = pd.read_csv('/kaggle/input/raw-data/train_data_converted_to_.csv')
test_raw = pd.read_csv('/kaggle/input/raw-data/test_data_converted_to_.csv')

col_order = train_raw.columns.to_list()
col_order.remove('y')
test_raw = test_raw[col_order]

label = pd.read_csv('/kaggle/input/raw-data/mask_to_actual_labeling.csv')

zero_or_nan_cols = train_raw.columns[
    ((train_raw == 0) | train_raw.isna()).all()
].tolist()
train_raw.drop(columns=zero_or_nan_cols,inplace=True)
test_raw.drop(columns=zero_or_nan_cols,inplace=True)

columns_with_only_1_or_nan = train_raw.columns[
    ((train_raw == 1) | train_raw.isna()).all()
].tolist()
for col in columns_with_only_1_or_nan:
    train_raw[col] = train_raw[col].fillna(0)
    test_raw[col] = test_raw[col].fillna(0)

methods = pd.read_csv('/kaggle/input/raw-data/Imputation_methods.csv')

methods['imputed with'].unique()

for col in train_raw.columns:
    if col not in methods['masked_column'].to_list():
        print(col)

train_raw.drop(columns=['Unnamed: 0','f218'],inplace=True)
test_raw.drop(columns=['Unnamed: 0','f218'],inplace=True)

num_cols = label[label['Type']=='Numerical']['masked_column'].to_list()
copy = num_cols.copy()
for col in copy:
    if col not in train_raw.columns:
        num_cols.remove(col)
for col in num_cols:
    method = methods[methods['masked_column']==col]['imputed with'].to_list()[0]
    if method=='mode':
        train_raw[col] = train_raw[col].fillna(train_raw[col].mode()[0])
        test_raw[col] = test_raw[col].fillna(test_raw[col].mode()[0])
    elif method=='mean':
        train_raw[col] = train_raw[col].fillna(train_raw[col].mean())
        test_raw[col] = test_raw[col].fillna(test_raw[col].mean())
    elif method=='zero':
        train_raw[col] = train_raw[col].fillna(0)
        test_raw[col] = test_raw[col].fillna(0)
    else:
        print(col," : ",method)

cols = ['f18','f22','f33','f34','f36','f37']
for col in cols:
    print(train_raw[col].value_counts())

# +
train_raw['f18'] = train_raw['f18'].fillna(0)
train_raw['f22'] = train_raw['f22'].fillna(0)
train_raw['f33'] = train_raw['f33'].fillna(0)
train_raw['f34'] = train_raw['f34'].fillna(0)
train_raw['f36'] = train_raw['f36'].fillna(0)
train_raw['f37'] = train_raw['f37'].fillna(0)

test_raw['f18'] = test_raw['f18'].fillna(0)
test_raw['f22'] = test_raw['f22'].fillna(0)
test_raw['f33'] = test_raw['f33'].fillna(0)
test_raw['f34'] = test_raw['f34'].fillna(0)
test_raw['f36'] = test_raw['f36'].fillna(0)
test_raw['f37'] = test_raw['f37'].fillna(0)
# -

cols=[]
for i in range(187,198):
    col = 'f'+str(i)
    cols.append(col)
train_raw.drop(columns=cols,inplace=True)
test_raw.drop(columns=cols,inplace=True)

cols2=[]
for i in range(174,185):
    col='f'+str(i)
    cols2.append(col)

train_raw['dummy']=0
test_raw['dummy']=0
for col in cols2:
    train_raw['dummy']+=train_raw[col]
    test_raw['dummy']+=test_raw[col]
for col1,col2 in zip(cols,cols2):
    train_raw[col1] = train_raw[col2]/train_raw['dummy']
    test_raw[col1] = test_raw[col2]/test_raw['dummy']
train_raw.drop(columns=['dummy'],inplace=True)
test_raw.drop(columns=['dummy'],inplace=True)

train_raw = train_raw.copy()
test_raw = test_raw.copy()

for col in cols:
    train_raw[col] = train_raw[col].fillna(0)
    test_raw[col] = test_raw[col].fillna(0)

# +
train_raw['f347'] = train_raw['f347'].fillna(train_raw.groupby('id3')['f347'].transform('mean'))
train_raw['f348'] = train_raw['f348'].fillna(train_raw.groupby('id3')['f348'].transform('mean'))
train_raw['f347'] = train_raw['f347'].fillna(0)
train_raw['f348'] = train_raw['f348'].fillna(0)

test_raw['f347'] = test_raw['f347'].fillna(test_raw.groupby('id3')['f347'].transform('mean'))
test_raw['f348'] = test_raw['f348'].fillna(test_raw.groupby('id3')['f348'].transform('mean'))
test_raw['f347'] = test_raw['f347'].fillna(0)
test_raw['f348'] = test_raw['f348'].fillna(0)
# -

onehot_cols = label[label['Type']=='One hot encoded']['masked_column'].to_list()
copy = onehot_cols.copy()
for col in copy:
    if col not in train_raw.columns:
        onehot_cols.remove(col)
for col in onehot_cols:
    train_raw[col] = train_raw[col].fillna(0)
    test_raw[col] = test_raw[col].fillna(0)

import pandas.api.types as ptype
cat_cols = label[label['Type']=='Categorical']['masked_column'].to_list()
for col in cat_cols:
    if ptype.is_object_dtype(train_raw[col]):
        train_raw[col] = train_raw[col].fillna("__missing__")
        test_raw[col] = test_raw[col].fillna("__missing__")
    elif ptype.is_numeric_dtype(train_raw[col]):
        train_raw[col] = train_raw[col].fillna(-1)
        test_raw[col] = test_raw[col].fillna(-1)
    else:
        print(col)

train_raw.isnull().any().any()

test_raw.isnull().any().any()

# +
train_raw['id4'] = pd.to_datetime(train_raw['id4'])
test_raw['id4'] = pd.to_datetime(test_raw['id4'])

train_raw['year'] = train_raw['id4'].dt.year
train_raw['month'] = train_raw['id4'].dt.month
train_raw['day'] = train_raw['id4'].dt.day
train_raw['dayofweek'] = train_raw['id4'].dt.dayofweek # 0=Monday and 6=Sunday
train_raw['weekofyear'] = train_raw['id4'].dt.isocalendar().week
train_raw['quarter'] = train_raw['id4'].dt.quarter
train_raw['hour'] = train_raw['id4'].dt.hour
train_raw['minute'] = train_raw['id4'].dt.minute
train_raw['second'] = train_raw['id4'].dt.second
train_raw['is_month_start'] = train_raw['id4'].dt.is_month_start
train_raw['is_month_end'] = train_raw['id4'].dt.is_month_end
train_raw['is_weekend'] = train_raw['id4'].dt.dayofweek >= 5

test_raw['year'] = test_raw['id4'].dt.year
test_raw['month'] = test_raw['id4'].dt.month
test_raw['day'] = test_raw['id4'].dt.day
test_raw['dayofweek'] = test_raw['id4'].dt.dayofweek # 0=Monday and 6=Sunday
test_raw['weekofyear'] = test_raw['id4'].dt.isocalendar().week
test_raw['quarter'] = test_raw['id4'].dt.quarter
test_raw['hour'] = test_raw['id4'].dt.hour
test_raw['minute'] = test_raw['id4'].dt.minute
test_raw['second'] = test_raw['id4'].dt.second
test_raw['is_month_start'] = test_raw['id4'].dt.is_month_start
test_raw['is_month_end'] = test_raw['id4'].dt.is_month_end
test_raw['is_weekend'] = test_raw['id4'].dt.dayofweek >= 5

train_raw.drop(columns=['id4'],inplace=True)
test_raw.drop(columns=['id4'],inplace=True)
# -

train_raw.shape

test_raw.shape

# +
mask=list(train_raw['f29']>train_raw['f28'])
train_raw.loc[mask,'f29']=0
train_raw.loc[mask,'f28']=0

mask=list(test_raw['f29']>test_raw['f28'])
test_raw.loc[mask,'f29']=0
test_raw.loc[mask,'f28']=0

# +
mask=list(train_raw['f31']>train_raw['f30'])
train_raw.loc[mask,'f30']=0
train_raw.loc[mask,'f31']=0

mask=list(test_raw['f31']>test_raw['f30'])
test_raw.loc[mask,'f30']=0
test_raw.loc[mask,'f31']=0
# -

train_raw.to_csv('train_cleaned.csv',index=False)
test_raw.to_csv('test_cleaned.csv',index=False)


