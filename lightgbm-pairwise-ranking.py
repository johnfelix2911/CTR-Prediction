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
import pandas as pd
import numpy as np
# -

df = pd.read_csv('/kaggle/input/train-data-ready-frfr/train_actually_ready.csv')
label = pd.read_csv('/kaggle/input/train-data-ready-frfr/mask_to_actual_labeling.csv')

for col in df.columns:
    if col not in label['masked_column'].to_list():
        print(col)

label = label[label['masked_column'].isin(df.columns)]

masked_column=['interest','day','dayofweek','hour','is_month_start','f375','f378','id8','avg_trans_amt','offer_ctr']
Description=['interest score','day','dayofweek','hour','is_month_start','f375','f378','id8','avg_trans_amt','offer_ctr']
Type=['Numerical','Numerical','Categorical','Numerical','Categorical','Numerical','Categorical','Categorical','Numerical','Numerical']
to_add = pd.DataFrame({'masked_column':masked_column,'Description':Description,'Type':Type})
to_add['Type'].unique()

label = pd.concat([label,to_add],axis=0)

df.drop(columns=label[label['Type']=='One hot encoded']['masked_column'].to_list(),inplace=True)

from sklearn.preprocessing import LabelEncoder
cat_cols =  label[label['Type']=='Categorical']['masked_column'].to_list()
cat_cols.remove('id3')
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score

cust_ids = df['id2'].unique()
train_cust, test_cust = train_test_split(cust_ids, test_size=0.2, random_state=42)
train_mask = df['id2'].isin(train_cust)
df_train = df[train_mask]
df_test  = df[~train_mask]

FEATURES = [f for f in df.columns if f not in ['id2','id3','y','id1']]
X_train = df_train[FEATURES]
y_train = df_train['y']
X_test  = df_test[FEATURES]
y_test  = df_test['y']


def make_group(df_sub):
    return df_sub.groupby('id2').size().to_list()


group_train = make_group(df_train)
group_test  = make_group(df_test)

ranker = lgb.LGBMRanker(
    objective='lambdarank',
    metric='ndcg',
    ndcg_at=[7],
    learning_rate=0.05,
    num_leaves=31,
    n_estimators=1000
)

cat_cols =  label[label['Type']=='Categorical']['masked_column'].to_list()
cat_cols.remove('id3')
ranker.fit(
    X_train, y_train,
    group=group_train,
    eval_set=[(X_test, y_test)],
    eval_group=[group_test],
    categorical_feature=cat_cols,
    callbacks=[lgb.early_stopping(50),lgb.log_evaluation(50)]
)

test_df = pd.read_csv('/kaggle/input/test-ready-to-test/test_ready.csv')

test_df.drop(columns=label[label['Type']=='One hot encoded']['masked_column'].to_list(),inplace=True)

cat_cols =  label[label['Type']=='Categorical']['masked_column'].to_list()
cat_cols.remove('id3')
for col in cat_cols:
    le = LabelEncoder()
    test_df[col] = le.fit_transform(test_df[col])

temp_df = df.drop(columns=['y'])

temp_df.shape

test_df.shape

for i in range(len(temp_df.columns)):
    if temp_df[temp_df.columns[i]].dtypes!=test_df[test_df.columns[i]].dtypes:
        print("haaaaaaaaaaaaaaaa")

X_new = test_df[FEATURES]

ranker.predict(X_new)

# +
# 1. Get scores
test_df['score'] = ranker.predict(X_new)

# 2. Sort by customer and descending score
test_df = test_df.sort_values(['id2', 'score'], ascending=[True, False])
# -

raw = pd.read_csv('/kaggle/input/test-data-raw/test_data_converted_to_.csv')

test_df['id5'] = raw['id5'].copy()
test_df = test_df[['id1','id2','id3','id5','score']]

test_df.head()

test_df['pred'] = 1

test_df.head()

test_df.drop(columns=['score'],inplace=True)

test_df = test_df.reset_index()

test_df.head()

test_df.drop(columns=['index'],inplace=True)

test_df.head()

test_df.to_csv('submission2.csv',index=False)

df_test['score'] = ranker.predict(X_test)
df_test = df_test.sort_values(['id2', 'score'], ascending=[True, False])

df_test = df_test[['id1','id2','id3','score']]
df_test.head()

df_test['pred']=1
df_test.drop(columns=['score'],inplace=True)

df_test['actual'] = y_test


def apk(truth,pred,k):
    if len(pred)>k:
        pred = pred[:k]
        truth = truth[:k]
    score = 0.0
    num_hits = 0.0
    total_relevant = sum(truth)
    if total_relevant==0:
        return 0.0
    for i,(p,t) in enumerate(zip(pred,truth),start=1):
        if p==1 and t==1:
            num_hits += 1
            score += num_hits/i
    return score/total_relevant


def mapk(df,k=7):
    grouped = df.groupby('id2')
    ap_scores = []
    decrease_by=0
    for _,group in grouped:
        pred = group['pred'].to_list()
        actual = group['actual'].to_list()
        if sum(actual)!=0:
            ap = apk(actual, pred, k)
            ap_scores.append(ap)
    return sum(ap_scores)/len(ap_scores)


df_test.head()

map7_score = mapk(df_test,k=7)
print(map7_score)


