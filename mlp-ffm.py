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
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.layers import Layer, Input, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

from imblearn.over_sampling import RandomOverSampler


# -

# !pip freeze > MLP_FFM_requirements.txt

# !pip install scikit-learn==1.4.2 imbalanced-learn==0.12.2

# !pip install --force-reinstall --no-cache-dir scikit-learn==1.4.2 imbalanced-learn==0.12.2

# !pip install numpy==1.26.4 --force-reinstall --no-cache-dir

class FFM_Layer(Layer):
    def __init__(self, sparse_feature_columns, k, w_reg=1e-6, v_reg=1e-6):
        super(FFM_Layer, self).__init__()
        self.sparse_feature_columns = sparse_feature_columns
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
        self.index_offset = []
        self.total_feat = 0
        for feat in self.sparse_feature_columns:
            self.index_offset.append(self.total_feat)
            self.total_feat += feat['feat_num']
        self.field_num = len(self.sparse_feature_columns)

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,), initializer='zeros', trainable=True)
        self.w = self.add_weight(name='w', shape=(self.total_feat, 1),
                                 initializer='random_normal',
                                 regularizer=regularizers.l2(self.w_reg), trainable=True)
        self.v = self.add_weight(name='v',
                                 shape=(self.total_feat, self.field_num, self.k),
                                 initializer='random_normal',
                                 regularizer=regularizers.l2(self.v_reg), trainable=True)

    def call(self, inputs):
        inputs = inputs + tf.constant(self.index_offset, dtype=tf.int32)
        first_order = self.w0 + tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1)
        second_order = 0
        latent = tf.nn.embedding_lookup(self.v, inputs)
        latent_sum = tf.reduce_sum(latent, axis=2)
        for i in range(self.field_num):
            for j in range(i + 1, self.field_num):
                vi = latent_sum[:, i, :]
                vj = latent_sum[:, j, :]
                second_order += tf.reduce_sum(vi * vj, axis=1, keepdims=True)
        return first_order + second_order


def build_ffm_with_mlp_numerical(sparse_cols, num_numerical, mlp_units=[64, 32]):
    k = 8
    sparse_input = Input(shape=(len(sparse_cols),), dtype=tf.int32, name='cat_input')
    ffm_output = FFM_Layer(sparse_cols, k, w_reg=1e-6, v_reg=1e-6)(sparse_input)

    num_input = Input(shape=(num_numerical,), dtype=tf.float32, name='num_input')
    x = num_input
    for idx, units in enumerate(mlp_units):
        x = Dense(units, activation='relu', name=f'num_dense_{idx}')(x)
    mlp_output = Dense(1, name='num_out')(x)

    logit = Add(name='logit')([ffm_output, mlp_output])
    proba = tf.keras.activations.sigmoid(logit)

    model = Model(inputs=[sparse_input, num_input], outputs=proba)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
    return model


def apply_ros(X,y,ratio):
    ros = RandomOverSampler(sampling_strategy=ratio,random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return X_resampled, y_resampled


def prepare_data_train(csv_path, label_col='y'):
    df = pd.read_csv(csv_path)

    X_resampled, y_resampled = apply_ros(df.drop(columns=[label_col]),df[label_col],0.1)

    extra = X_resampled[['id1','id2','id5','id3']]
    X = X_resampled.drop(columns=['id1','id2','id5','id3'])
    y = y_resampled.values
    
    label = pd.read_csv('/kaggle/input/prepared-data-mlp-ffm/actual_final_label.csv')
    
    cat_cols = label[label['Type']=='Categorical']['masked_column'].to_list()
    cat_cols.extend(label[label['Type']=='One hot encoded']['masked_column'].to_list())
    cat_cols.remove('id3')
    num_cols = label[label['Type']=='Numerical']['masked_column'].to_list()
    num_cols.remove('id5')

    X[cat_cols] = X[cat_cols].astype('category')

    # Encode categorical columns
    sparse_input_data = []
    sparse_cols_metadata = []

    encoder = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        num_classes = X[col].nunique()
        sparse_input_data.append(X[col].values)
        sparse_cols_metadata.append({'feat': col, 'feat_num': num_classes})
        encoder[col] = le

    sparse_input = np.stack(sparse_input_data, axis=1)
    
    scaler = StandardScaler()
    numerical_input = scaler.fit_transform(X[num_cols].values.astype(np.float32))

    return sparse_input, numerical_input, y, sparse_cols_metadata, len(num_cols), encoder, scaler, extra


def prepare_data_test(csv_path, encoder=None, scaler=None):
    df = pd.read_csv(csv_path)

    extra = df[['id1','id2','id3','id5']]
    df = df.drop(columns=['id1','id2','id5','id3'])

    # Separate label
    X = df.copy()
    
    label = pd.read_csv('/kaggle/input/prepared-data-mlp-ffm/actual_final_label.csv')
    
    cat_cols = label[label['Type']=='Categorical']['masked_column'].to_list()
    cat_cols.extend(label[label['Type']=='One hot encoded']['masked_column'].to_list())
    cat_cols.remove('id3')
    num_cols = label[label['Type']=='Numerical']['masked_column'].to_list()
    num_cols.remove('id5')

    X[cat_cols] = X[cat_cols].astype('category')

    # Encode categorical columns
    sparse_input_data = []
    sparse_cols_metadata = []

    for col in cat_cols:
        le = encoder[col]
        known_classes = set(le.classes_)
        X[col] = X[col].astype(str).apply(lambda x: x if x in known_classes else 'unknown')
        le.classes_ = np.append(le.classes_, 'unknown')
        X[col] = le.transform(X[col])
        num_classes = X[col].nunique()
        sparse_input_data.append(X[col].values)
        sparse_cols_metadata.append({'feat': col, 'feat_num': num_classes})

    sparse_input = np.stack(sparse_input_data, axis=1)
    
    numerical_input = scaler.transform(X[num_cols].values.astype(np.float32))

    return sparse_input, numerical_input, sparse_cols_metadata, len(num_cols), encoder, extra


def apk(actual, predicted, k=7):
    """
    actual: list of relevant ids (id3s with y==1)
    predicted: list of predicted ids in rank order
    """
    if not actual:
        return 0.0

    predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    
    return score / min(len(actual), k)


def mapk(df, k=7):
    ap_scores = []

    for _, group in df.groupby("id2"):
        actual = group[group["actual"] == 1]["id3"].tolist()
        predicted = group.sort_values("pred_proba", ascending=False)["id3"].tolist()
        

        if actual:
            ap = apk(actual, predicted, k)
            ap_scores.append(ap)

    return np.mean(ap_scores)

# +
csv_file_path = '/kaggle/input/prepared-data-mlp-ffm/train_done.csv'  # Replace with your actual path

X_cat, X_num, y, sparse_col_info, num_num, encoder, scaler, extra = prepare_data_train(csv_file_path)

row_indices = np.arange(len(y))

# Split
X_cat_train, X_cat_val, X_num_train, X_num_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X_cat, X_num, y, row_indices, test_size=0.2, random_state=42
)

# Build model
model = build_ffm_with_mlp_numerical(sparse_col_info, num_numerical=num_num, mlp_units=[128, 64])

early_stop = EarlyStopping(
    monitor='val_AUC',
    patience=2,
    restore_best_weights=True,
    mode='max'
)

# Train
model.fit([X_cat_train, X_num_train], y_train,
          validation_data=([X_cat_val, X_num_val], y_val),
          batch_size=1024, epochs=10,
         callbacks=[early_stop])

# +
y_pred = model.predict([X_cat_val,X_num_val])
extra = extra.iloc[idx_val].reset_index(drop=True).copy()

extra['pred_proba'] = y_pred.ravel()
extra['actual'] = y_val
# -

map7score = mapk(extra,k=7)
print(map7score)

test_file_path = '/kaggle/input/prepared-data-mlp-ffm/test_done.csv'
X_cat, X_num, sparse_col_info, num_num, encoder, extra = prepare_data_test(test_file_path,encoder=encoder,scaler=scaler)
y_pred = model.predict([X_cat,X_num])
extra['pred_proba'] = y_pred
extra.sort_values(by=['id2', 'pred_proba'], ascending=[True, False], inplace=True)

extra['pred']=1
extra.reset_index(drop=True,inplace=True)
extra.drop(columns=['pred_proba'],inplace=True)

extra.head()

extra.to_csv('submission3.csv',index=False)
