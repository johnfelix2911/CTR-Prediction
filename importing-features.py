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

offer = pd.read_csv('/kaggle/input/addtional-data/offer_metadata_converted_to_.csv')

test = pd.read_csv('/kaggle/input/to-import-features/test_cut_down.csv')
train = pd.read_csv('/kaggle/input/to-import-features/train_cut_down.csv')

offer.drop(columns=['Unnamed: 0','id9','f377','f374','id11'],inplace=True)

offer['f376'] = offer['f376'].fillna(0)
offer['id8'] = offer['id8'].fillna(10000000)

offer.isnull().any().any()

train = train.merge(offer, on='id3', how='left')
test = test.merge(offer,on='id3',how='left')

train['id10'].value_counts()

# +
train['f375'] = train['f375'].fillna(train['f375'].mode()[0])
train['f376'] = train['f376'].fillna(train['f376'].mode()[0])
train['id10'] = train['id10'].fillna(-1.0)
train['f378'] = train['f378'].fillna('__missing__')
train['id8'] = train['id8'].fillna(10000000)

test['f375'] = test['f375'].fillna(test['f375'].mode()[0])
test['f376'] = test['f376'].fillna(test['f376'].mode()[0])
test['id10'] = test['id10'].fillna(-1.0)
test['f378'] = test['f378'].fillna('__missing__')
test['id8'] = test['id8'].fillna(10000000)
# -

train.drop(columns=['id12','id13'],inplace=True)
test.drop(columns=['id12','id13'],inplace=True)

train.isnull().any().any()

test.isnull().any().any()

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
cols=['f375', 'f376', 'id10', 'f378', 'id8']
mask=[False, False, True, True, True]
X = train[cols].copy()
y = train['y'].copy()
le = LabelEncoder()
X['id10'] = le.fit_transform(X['id10'].astype(str))
le = LabelEncoder()
X['f378'] = le.fit_transform(X['f378'].astype(str))
le = LabelEncoder()
X['id8'] = le.fit_transform(X['id8'].astype(str))
mi_scores = mutual_info_classif(X, y, discrete_features=mask, random_state=0)

mi_scores

train.drop(columns=['id10'],inplace=True)
test.drop(columns=['id10'],inplace=True)

trans = pd.read_csv('/kaggle/input/addtional-data/add_trans_converted_to_.csv')

trans['id8'] = trans['id8'].fillna(10000000)

industry_avg = trans.groupby('id8')['f367'].mean().reset_index()
industry_avg.rename(columns={'f367': 'avg_trans_amt'}, inplace=True)
train = train.merge(industry_avg, on='id8', how='left')
test = test.merge(industry_avg, on='id8', how='left')

event = pd.read_csv('/kaggle/input/addtional-data/add_event_converted_to_.csv')

event['clicked'] = event['id7'].isnull().astype(int)

ctr_per_offer = event.groupby('id3')['clicked'].mean().reset_index()
ctr_per_offer.columns = ['id3','offer_ctr']
train = train.merge(ctr_per_offer, on='id3', how='left')
test = test.merge(ctr_per_offer, on='id3', how='left')

train.to_csv('train_to_feat_engg.csv',index=False)
test.to_csv('test_to_feat_engg.csv',index=False)

label = pd.read_csv('/kaggle/input/to-import-features/updated_label.csv')

for col in train.columns:
    if col not in label['masked_column'].to_list():
        print(col)

cols=['f375','f376','f378','id8','avg_trans_amt','offer_ctr']
desc=['redemption freq','discout rate','offerings body','CM industry code','avg trans amt','offer ctr']
types=['Numerical','Numerical','Categorical','Categorical','Numerical','Numerical']
to_add = pd.DataFrame({'masked_column':cols,'Description':desc,'Type':types})
label = pd.concat([label,to_add],axis=0)

label.to_csv('updated_labels2.csv',index=False)


