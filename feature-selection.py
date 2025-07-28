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

train = pd.read_csv('/kaggle/input/cleaned-from-raw/train_cleaned.csv')
test = pd.read_csv('/kaggle/input/cleaned-from-raw/test_cleaned.csv')

label = pd.read_csv('/kaggle/input/cleaned-from-raw/mask_to_actual_labeling.csv')
label = label[label['masked_column'].isin(train.columns)]

for col in train.columns:
    if col not in label['masked_column'].to_list():
        print(col)

train.drop(columns=['year','month','quarter','weekofyear'],inplace=True)
test.drop(columns=['year','month','quarter','weekofyear'],inplace=True)

cols = ['day','dayofweek','hour','minute','second','is_month_start','is_month_end','is_weekend']
desc = ['day','dayofweek','hour','minute','second','is_month_start','is_month_end','is_weekend']
types = ['Categorical','Categorical','Categorical','Categorical','Categorical','Categorical','Categorical','Categorical',]
to_add = pd.DataFrame({'masked_column':cols,'Description':desc,'Type':types})
label = pd.concat([label,to_add],axis=0)

train.shape

from sklearn.feature_selection import mutual_info_classif
num = label[label['Type']=='Numerical']['masked_column'].to_list()
num.remove('id5')
X = train[num]
y = train['y']
scores = []
dic = {}
for i in range(0,X.shape[1]+10,10):
    mi_scores = mutual_info_classif(X[X.columns[i:i+10]], y, discrete_features=False, random_state=0)
    scores.extend(mi_scores)
    for j in range(0,10,1):
        try:
            print(str(X.columns[i+j])+" , "+str(mi_scores[j]))
            dic[X.columns[i+j]] = mi_scores[j]
        except:
            print("out of range")

dropped = []
for col,score in dic.items():
    if score<=0.005:
        dropped.append(col)
train.drop(columns=dropped,inplace=True)
test.drop(columns=dropped,inplace=True)
label = label[label['masked_column'].isin(train.columns)]

num = label[label['Type']=='Numerical']['masked_column'].to_list()
num.remove('id5')
corr_matrix = train[num].corr()

# +
cols = num.copy()
def find_col(col):
    for pos in range(len(lis)):
        if col in lis[pos]:
            return pos

check = {}
for col in cols:
    check[col]=0
lis=[]
cnt=0
for i in range(len(cols)):
    for j in range(len(cols)):
        if i!=j:
            if abs(corr_matrix[cols[i]][cols[j]])>=0.85:
                cnt+=1
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
print(cnt)
# -

for l in lis:
    for col in l:
        print(label[label['masked_column']==col]['Description'].to_list()[0])
    print()

from sklearn.preprocessing import LabelEncoder
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

train.to_csv('train_cut_down.csv',index=False)
test.to_csv('test_cut_down.csv',index=False)

label.to_csv('updated_label.csv',index=False)


