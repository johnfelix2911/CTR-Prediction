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

train = pd.read_csv('/kaggle/input/ready-for-feat-engg/train_to_feat_engg.csv')
test = pd.read_csv('/kaggle/input/ready-for-feat-engg/test_to_feat_engg.csv')

train_anj = pd.read_csv('/kaggle/input/cleaned-by-anjali/train_clean.csv')
test_anj = pd.read_csv('/kaggle/input/cleaned-by-anjali/test_clean.csv')

train.shape

train_anj.shape

train.sort_values(by='id1',ascending=True,inplace=True)
train_anj.sort_values(by='id1',ascending=True,inplace=True)
test.sort_values(by='id1',ascending=True,inplace=True)
test_anj.sort_values(by='id1',ascending=True,inplace=True)

lis1 = train['id1'].to_list()
lis2 = train_anj['id1'].to_list()
for i,j in zip(lis1,lis2):
    if i!=j:
        print("womp womp")

lis1 = test['id1'].to_list()
lis2 = test_anj['id1'].to_list()
for i,j in zip(lis1,lis2):
    if i!=j:
        print("womp womp")

cols=[]
for col in train_anj.columns:
    if col not in train.columns:
        cols.append(col)
print(cols)

cols=['industry_category', 'time_spent_ratio_180d', 'time_spent_|AMEX|ACQ|_ratio_180d', 'time_spent_|AMEX|LOY|_ratio_180d', 'time_spent_|ESTATEMENT|_ratio_180d', 'time_spent_|MR|_ratio_180d', 'time_spent_|OCE|_ratio_180d', 'time_spent_|TRAVEL|_ratio_180d', 'time_spent_|PREPAID|GIFTCARD|_ratio_180d', 'time_spent_|in none|_ratio_180d', 'decay_ctr_30d', 'decay_ctr_14d', 'time_of_day', 'reward_points', 'weighted_ctr_30d', 'weighted_clicks_30d', 'weighted_imp_30d', 'weighted_merch_ctr_30d', 'imp_ratio_30d', 'f12', 'f43_log', 'f44_log', 'f45_log', 'f46_log', 'f47_log', 'f49_log', 'f51_log', 'ctr_ratio_var_avg', 'Min_Spent', 'Max_Spent', 'Avg_Spent', 'Median_Spent', 'Most_Common_Time', 'Avg_Discount']
train[cols] = train_anj[cols].copy()
test[cols] = test_anj[cols].copy()

train.columns[train.isnull().any()]

test.columns[test.isnull().any()]

train[['Avg_Discount','Median_Spent','Avg_Spent','Max_Spent','Min_Spent']]=train[['Avg_Discount','Median_Spent','Avg_Spent','Max_Spent','Min_Spent']].fillna(0)
test[['Avg_Discount','Median_Spent','Avg_Spent','Max_Spent','Min_Spent']]=test[['Avg_Discount','Median_Spent','Avg_Spent','Max_Spent','Min_Spent']].fillna(0)

train['Most_Common_Time'] = train['Most_Common_Time'].fillna("__missing__")
test['Most_Common_Time'] = test['Most_Common_Time'].fillna("__missing__")

train.isnull().any().any()

test.isnull().any().any()

test.columns[test.isnull().any()]

test['offer_ctr'] = test['offer_ctr'].fillna(0)
test['industry_category'] = test['industry_category'].fillna('__missing__')
test['decay_ctr_30d'] = test['decay_ctr_30d'].fillna(0)
test['decay_ctr_14d'] = test['decay_ctr_14d'].fillna(0)

test.isnull().any().any()

label = pd.read_csv('/kaggle/input/ready-for-feat-engg/updated_labels2.csv')

cols=[]
for col in train.columns:
    if col not in label['masked_column'].to_list():
        print(col)
        cols.append(col)

desc=cols.copy()
types=['Categorical','Numerical','Numerical','Numerical','Numerical','Numerical','Numerical','Numerical','Numerical','Numerical',
       'Numerical','Numerical','Categorical','Numerical','Numerical','Numerical','Numerical','Numerical','Numerical','Numerical',
      'Numerical','Numerical','Numerical','Numerical','Numerical','Numerical','Numerical','Numerical','Numerical','Numerical',
       'Numerical','Numerical','Categorical','Numerical']

for col,ty in zip(cols,types):
    print(col," : ",ty)

to_add = pd.DataFrame({'masked_column':cols,'Description':desc,'Type':types})
label = pd.concat([label,to_add],axis=0)

for col in train.columns:
    if col not in label['masked_column'].to_list():
        print(col)

num_cols = label[label['Type']=='Numerical']['masked_column'].to_list()
num_cols.remove('id5')
corr_matrix = train[num_cols].corr()

# +
cols = num_cols.copy()
def find_col(col):
    for pos in range(len(lis)):
        if col in lis[pos]:
            return pos

check = {}
for col in cols:
    check[col]=0
lis=[]
for i in range(len(cols)):
    for j in range(len(cols)):
        if i!=j:
            if abs(corr_matrix[cols[i]][cols[j]])>=0.85:
                if check[cols[i]]==0 and check[cols[j]]==0:
                    lis.append([cols[i],cols[j]])
                    check[cols[i]]=1
                    check[cols[j]]=1
                elif check[cols[i]]==0:
                    pos = find_col(cols[j])
                    lis[pos].append(cols[i])
                    check[cols[i]]=1
                    check[cols[j]]=1
                elif check[cols[j]]==0:
                    pos = find_col(cols[i])
                    lis[pos].append(cols[j])
                    check[cols[i]]=1
                    check[cols[j]]=1
                else:
                    continue
                    # it will be handled in the next correlation matrix
# -

for l in lis:
    for col in l:
        print(label[label['masked_column']==col]['Description'].to_list()[0])
    print()

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
dropped=[]
for l in lis:
    cols=l.copy()
    if label[label['masked_column']==cols[0]]['Type'].to_list()[0]=='Numerical':
        X = train[cols].copy()
        y = train['y']
        mi_scores = mutual_info_classif(X,y,discrete_features=False,random_state=42)
        maxi=0
        copy = cols.copy()
        for col,score in zip(copy,mi_scores):
            if score>maxi:
                maxi=score
                keep=col
        cols.remove(keep)
        dropped.extend(cols)
    elif label[label['masked_column']==cols[0]]['Type'].to_list()[0]=='Categorical':
        X = train[cols].copy()
        y = train['y']
        for col in X.columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        mi_scores = mutual_info_classif(X,y,discrete_features=True,random_state=42)
        maxi=0
        copy = cols.copy()
        for col,score in zip(copy,mi_scores):
            if score>maxi:
                maxi=score
                keep=col
        cols.remove(keep)
        dropped.extend(cols)

train.drop(columns=dropped,inplace=True)
test.drop(columns=dropped,inplace=True)

train.shape

test.shape

train.to_csv('train_done.csv',index=False)
test.to_csv('test_done.csv',index=False)

label = label[label['masked_column'].isin(train.columns)]
label.to_csv('final_labels.csv',index=False)


