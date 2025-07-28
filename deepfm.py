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

# !pip install scikit-learn==1.4.2 imbalanced-learn==0.12.2

# !pip install --force-reinstall --no-cache-dir scikit-learn==1.4.2 imbalanced-learn==0.12.2

# !pip install numpy==1.26.4 --force-reinstall --no-cache-dir

# + _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Reshape, Add, Concatenate, Dropout, Activation, Lambda
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import RandomOverSampler


# -

# !pip freeze > deepfm_requirements.txt

def DeepFM(num_numeric, categorical_feature_info ):
    # Define inputs: one for all numeric features, and one per categorical feature
    numeric_input = Input(shape=(num_numeric,), name='numeric_input')
    cat_inputs = []
    for name, vocab_size in categorical_feature_info.items():
        cat_inputs.append(Input(shape=(1,), name=f'{name}_input'))

    # 1) First-order linear terms
    linear_terms = []
    # Numeric linear term: Dense(1) on numeric inputs
    linear_terms.append(Dense(1, name='linear_numeric')(numeric_input))
    # Categorical linear term: embedding of size 1 per category
    for inp, (name, vocab_size) in zip(cat_inputs, categorical_feature_info.items()):
        lin_emb = Embedding(input_dim=vocab_size, output_dim=1, name=f'linear_emb_{name}')(inp)
        linear_terms.append(Flatten()(lin_emb))
    linear_logit = Add(name='linear_logit')(linear_terms)

    # 2) Second-order FM terms
    embed_dim = 10 # Tune this as required
    embeddings = []
    for inp, (name, vocab_size) in zip(cat_inputs, categorical_feature_info.items()):
        emb = Embedding(input_dim=vocab_size, output_dim=embed_dim, name=f'emb_{name}')(inp)
        embeddings.append(Reshape((1, embed_dim))(emb))

    # Stack to shape (None, num_cat, embed_dim)
    concat_embeds = Concatenate(axis=1)(embeddings)  # shape (batch_size, p, k)
    # Sum of embeddings: shape (batch_size, k)
    sum_of_embeds = Lambda(lambda x: tf.reduce_sum(x, axis=1))(concat_embeds)
    # Square of sum
    square_of_sum = Lambda(lambda x: tf.square(x))(sum_of_embeds)
    # Sum of squares
    sum_of_squares = Lambda(lambda x: tf.reduce_sum(tf.square(x), axis=1))(concat_embeds)
    # FM second-order vector (batch_size, k)
    fm_vec = Lambda(lambda x: 0.5 * (x[0] - x[1]))([square_of_sum, sum_of_squares])
    # Sum over latent dim to get scalar logit per sample
    fm_logit = Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims=True), name='fm_logit')(fm_vec)

    # 3) Deep part (DNN)
    # Flatten embeddings for DNN input
    flat_embeds = [Flatten()(emb) for emb in embeddings]  # each (batch_size, k)
    dnn_input = Concatenate()(flat_embeds + [numeric_input])  # (batch_size, p*k + num_numeric)
    # Pass through DNN layers
    x = Dropout(0.3)(dnn_input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    dnn_out = Dense(1, name='dnn_logit')(x)  # final logit from DNN

    # Combine FM and DNN parts
    final_logit = Add(name='final_logit')([linear_logit, fm_logit, dnn_out])
    output = Activation('sigmoid')(final_logit)
    
    # Build and compile model
    model = Model(inputs=[numeric_input] + cat_inputs, outputs=output)

    return model


def remove_constants():
    train = pd.read_csv('/kaggle/input/prepared-data-mlp-ffm/train_done.csv')
    dropped = []
    for col in train.columns:
        if train[col].nunique()<2:
            dropped.append(col)
    train.drop(columns=dropped,inplace=True)
    label = pd.read_csv('/kaggle/input/prepared-data-mlp-ffm/actual_final_label.csv')
    label = label[label['masked_column'].isin(train.columns)]
    return train, label


def prepare_train(train, label):
    # the model takes a list of [X_numeric, X_cat1, X_cat2, ...] as input
    y = train['y']
    X = []
    cat_cols = []
    encoder = {}
    cat_cols.extend(label[label['Type']=='Categorical']['masked_column'].to_list())
    cat_cols.extend(label[label['Type']=='One hot encoded']['masked_column'].to_list())
    for col in cat_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        encoder[col] = le
    num_cols = label[label['Type']=='Numerical']['masked_column'].to_list()
    scaler = StandardScaler()
    numerical_input = scaler.fit_transform(train[num_cols].values.astype(np.float32))
    X.append(np.array(numerical_input))
    for col in cat_cols:
        X.append(np.array(train[col], dtype=np.int32).reshape(-1,1))
    return X, y, encoder, scaler


def get_info(train,label):
    num_numeric = len(label[label['Type']=='Numerical']['masked_column'].to_list())
    cat_feat_info = {}
    cat_cols = []
    cat_cols.extend(label[label['Type']=='Categorical']['masked_column'].to_list())
    cat_cols.extend(label[label['Type']=='One hot encoded']['masked_column'].to_list())
    cat_cols.remove('id3')
    for col in cat_cols:
        cat_feat_info[col] = train[col].nunique()
    return num_numeric, cat_feat_info


def apply_ros(X,y,ratio):
    ros = RandomOverSampler(sampling_strategy=ratio,random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    X_resampled['y'] = y_resampled
    return X_resampled


train, label = remove_constants()
train.drop(columns=['id1','id2','id5','id3'],inplace=True) # not training the model on offer id as well
label = label[label['masked_column'].isin(train.columns)]
train = apply_ros(train.drop(columns=['y']).copy(), train['y'], 0.2)
train = train.sample(frac=1, random_state=42).reset_index(drop=True)
num_numeric, cat_feat_info = get_info(train,label)
X, y, encoder, scaler = prepare_train(train, label)
col_order = train.columns.to_list() # test data and train data should have the same feature order
col_order.remove('y')
del train
del label

model = DeepFM(num_numeric, cat_feat_info)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

early_stop = EarlyStopping(patience=2, restore_best_weights=True)
model.fit(X, y, batch_size=1024, epochs=5, validation_split=0.1, callbacks=[early_stop])


def prepare_test(test, encoder, scaler):
    label = pd.read_csv('/kaggle/input/prepared-data-mlp-ffm/actual_final_label.csv')
    label = label[label['masked_column'].isin(test.columns)]
    X = []
    cat_cols = []
    
    cat_cols.extend(label[label['Type']=='Categorical']['masked_column'].to_list())
    cat_cols.extend(label[label['Type']=='One hot encoded']['masked_column'].to_list())
    for col in cat_cols:
        le = encoder[col]
        known = set(le.classes_)
        test[col] = test[col].astype(str).apply(lambda x: x if x in known else 'unknown')
        le.classes_ = np.append(le.classes_,'unknown')
        test[col] = le.transform(test[col])
    num_cols = label[label['Type']=='Numerical']['masked_column'].to_list()
    numerical_input = scaler.transform(test[num_cols].values.astype(np.float32))
    X.append(np.array(numerical_input))
    for col in cat_cols:
        X.append(np.array(test[col],dtype=np.int32).reshape(-1,1))
    return X


test = pd.read_csv('/kaggle/input/prepared-data-mlp-ffm/test_done.csv')
extra = test[['id1','id2','id3','id5']]
test = test[col_order]
X_test = prepare_test(test, encoder, scaler)

y_pred = model.predict(X_test)

extra['pred_proba'] = y_pred
extra.sort_values(by=['id2','pred_proba'],ascending=[True,False],inplace=True)
extra.reset_index(drop=True,inplace=True)
extra.drop(columns=['pred_proba'],inplace=True)

extra['pred'] = 1

extra.head()

extra.to_csv('submission11.csv',index=False)


