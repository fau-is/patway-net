#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:49:26 2021

@author: makraus
"""

import pandas as pd
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import matplotlib.pyplot as plt
import seaborn as sns
import epph.src.util as util

ds_path = '../data/Sepsis Cases - Event Log.csv'
n_hidden = 8
target_activity = 'Release B'

static_features = ['InfectionSuspected', 'DiagnosticBlood', 'DisfuncOrg',
                   'SIRSCritTachypnea', 'Hypotensie',
                   'SIRSCritHeartRate', 'Infusion', 'DiagnosticArtAstrup', 'Age',
                   'DiagnosticIC', 'DiagnosticSputum', 'DiagnosticLiquor',
                   'DiagnosticOther', 'SIRSCriteria2OrMore', 'DiagnosticXthorax',
                   'SIRSCritTemperature', 'DiagnosticUrinaryCulture', 'SIRSCritLeucos',
                   'Oligurie', 'DiagnosticLacticAcid', 'Diagnose', 'Hypoxie',
                   'DiagnosticUrinarySediment', 'DiagnosticECG']

seq_features = ['Leucocytes', 'CRP', 'LacticAcid', 'ER Triage', 'ER Sepsis Triage',
                'IV Liquid', 'IV Antibiotics', 'Admission NC', 'Admission IC',
                'Return ER', 'Release A', 'Release B', 'Release C', 'Release D',
                'Release E']


# pre-processing
df = pd.read_csv(ds_path)
df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])
diagnose_mapping = dict(zip(df['Diagnose'].unique(), np.arange(len(df['Diagnose'].unique()))))
print(diagnose_mapping)
df['Diagnose'] = df['Diagnose'].apply(lambda x: diagnose_mapping[x])
df['Age'] = df['Age'].fillna(-1.)


max_leucocytes = np.percentile(df['Leucocytes'].dropna(), 95)
max_lacticacid = np.percentile(df['LacticAcid'].dropna(), 95)


x_seqs = []
x_statics = []
y = []

for case in df['Case ID'].unique():
    after_registration_flag = False
    found_target_flag = False
    df_tmp = df[df['Case ID'] == case]
    df_tmp = df_tmp.sort_values(by='Complete Timestamp')
    for _, value in df_tmp.iterrows():
        if value['Activity'] == 'ER Registration':
            x_statics.append(value[static_features].values.astype(float))
            x_seqs.append([])
            after_registration_flag = True
            continue

        if 'Release' in value['Activity']:
            if value['Activity'] == target_activity:
                y.append(1)
                found_target_flag = True
            break

        """
        if x['Activity'] == target_activity:  # todo: cannot be reached
            y.append(1)
            found_target_flag = True
            break
        """

        if after_registration_flag:
            x_seqs[-1].append(util.get_custom_one_hot_of_activity(value, max_leucocytes, max_lacticacid))

    if not found_target_flag and after_registration_flag:
        y.append(0)

assert len(x_seqs) == len(x_statics) == len(y)  # check

max_len = int(np.percentile([len(x) for x in x_seqs], 95))  # we cut the extreme cases for runtime
print(f'Cutting everything after {max_len} activities')
x_seqs_final = np.zeros((len(x_seqs), max_len, len(x_seqs[0][0])), dtype=np.float32)
for idx, value in enumerate(x_seqs):
    x_seqs_final[idx, :min(len(value), max_len), :] = np.array(value[:max_len])
x_statics_final = np.array(x_statics)
y_final = np.array(y).astype(np.int32)


def train_lstm(x_train_seq, x_train_stat, y_train):
    max_case_len = x_train_seq.shape[1]
    num_features_seq = x_train_seq.shape[2]
    num_features_stat = x_train_stat.shape[1]

    # Bidirectional LSTM
    # Input layer
    input_layer_seq = tf.keras.layers.Input(shape=(max_case_len, num_features_seq), name='seq_input_layer')
    input_layer_static = tf.keras.layers.Input(shape=(num_features_stat), name='static_input_layer')

    # Hidden layer
    hidden_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        units=n_hidden,
        return_sequences=False))(input_layer_seq)

    concatenate_layer = tf.keras.layers.Concatenate(axis=1)([hidden_layer, input_layer_static])

    print(concatenate_layer)

    # Output layer
    output_layer = tf.keras.layers.Dense(1,
                                         activation='sigmoid',
                                         name='output_layer')(concatenate_layer)

    model = tf.keras.models.Model(inputs=[input_layer_seq, input_layer_static],
                                  outputs=[output_layer])

    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('../model/model.ckpt',
                                                          monitor='val_loss',
                                                          verbose=0,
                                                          save_best_only=True,
                                                          save_weights_only=False,
                                                          mode='auto')

    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                      factor=0.5,
                                                      patience=10,
                                                      verbose=0,
                                                      mode='auto',
                                                      min_delta=0.0001,
                                                      cooldown=0,
                                                      min_lr=0)

    model.summary()
    model.fit([x_train_seq, x_train_stat], y_train,
              validation_split=0.1,
              verbose=1,
              callbacks=[early_stopping, model_checkpoint, lr_reducer],
              batch_size=16,
              epochs=100)

    return model


model = train_lstm(x_seqs_final, x_statics_final, y_final)
output_weights = model.get_layer(name='output_layer').get_weights()[0].flatten()
output_names = [f'Hidden State {i}' for i in range(2 * n_hidden)] + static_features

fig, ax = plt.subplots(figsize=(10, 10))
y_pos = np.arange(len(output_names))
ax.barh(y_pos, output_weights)
ax.set_yticks(y_pos)
ax.set_yticklabels(output_names)
plt.show()

import shap

static_df = pd.DataFrame(data=x_statics_final, columns=static_features)
seqs_df = pd.DataFrame(data=x_seqs_final.reshape(-1, len(seq_features)),
                       columns=seq_features)

explainer = shap.DeepExplainer(model, [x_seqs_final, x_statics_final])
shap_values = explainer.shap_values([x_seqs_final, x_statics_final])

static_shaps = pd.DataFrame(data=shap_values[0][1], columns=[f'SHAP {x}' for x in static_features])
static_value_shaps = pd.concat([static_df, static_shaps], axis=1)

seq_shaps = pd.DataFrame(data=shap_values[0][0].reshape(-1, len(seq_features)),
                         columns=[f'SHAP {x}' for x in seq_features])
seq_value_shape = pd.concat([seqs_df, seq_shaps], axis=1)

sns.jointplot(data=static_value_shaps, x='SHAP Hypotensie', y='Hypotensie', marker='+', size=10, alpha=0.3)

sns.jointplot(data=seq_value_shape, x='SHAP Leucocytes', y='Leucocytes', marker='+', size=10, alpha=0.3)

