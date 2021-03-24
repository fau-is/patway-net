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
import matplotlib
import seaborn as sns
import epph.src.util as util
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import shap
import itertools

ds_path = '../data/Sepsis Cases - Event Log.csv'
n_hidden = 8
target_activity = 'Admission IC'
# Release A: Very good
# Release B: bad
# Release C-E: Few samples
# Admission IC: Good
# Admission NC: Bad

train_size = 0.8

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


def get_data(target_activity):
    x_seqs = []
    x_statics = []
    y = []

    for case in df['Case ID'].unique():
        after_registration_flag = False
        found_target_flag = False
        df_tmp = df[df['Case ID'] == case]
        df_tmp = df_tmp.sort_values(by='Complete Timestamp')
        for _, x in df_tmp.iterrows():
            if x['Activity'] == 'ER Registration':
                x_statics.append(x[static_features].values.astype(float))
                x_seqs.append([])
                after_registration_flag = True
                continue

            if 'Release' in x['Activity']:
                if x['Activity'] == target_activity:
                    y.append(1)
                    found_target_flag = True
                break

            if x['Activity'] == target_activity:
                y.append(1)
                found_target_flag = True
                break

            if after_registration_flag:
                x_seqs[-1].append(util.get_custom_one_hot_of_activity(x,
                                                                      max_leucocytes,
                                                                      max_lacticacid))

        if not found_target_flag and after_registration_flag:
            y.append(0)

    assert len(x_seqs) == len(x_statics) == len(y)

    max_len = int(np.percentile([len(x) for x in x_seqs], 95))  # we cut the extreme cases for runtime
    print(f'Cutting evrthing after {max_len} activities')
    x_seqs_final = np.zeros((len(x_seqs), max_len, len(x_seqs[0][0])), dtype=np.float32)
    for i, x in enumerate(x_seqs):
        if len(x) > 0:
            x_seqs_final[i, :min(len(x), max_len), :] = np.array(x[:max_len])
    x_statics_final = np.array(x_statics)
    y_final = np.array(y).astype(np.int32)

    return x_seqs_final, x_statics_final, y_final


def custom_data_split(x_seqs_final, x_statics_final, y_final, train_size):

    x_seqs_train = x_seqs_final[:int(train_size * x_seqs_final.shape[0])]
    x_seqs_test = x_seqs_final[int(train_size * x_seqs_final.shape[0]):]
    x_statics_train = x_statics_final[:int(train_size * x_statics_final.shape[0])]
    x_statics_test = x_statics_final[int(train_size * x_statics_final.shape[0]):]
    y_train = y_final[:int(train_size * len(y_final))]
    y_test = y_final[int(train_size * len(y_final)):]

    return x_seqs_train, x_seqs_test, \
           x_statics_train, x_statics_test, \
           y_train, y_test


def my_palplot(pal, size=1, ax=None):
    n = 5
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(n * size, size))
    ax.imshow(np.arange(n).reshape(n, 1),
              cmap=matplotlib.colors.ListedColormap(list(pal)),
              interpolation="nearest", aspect="auto")

    # )


def compute_shap_summary_plot(X_all):
    matplotlib.style.use('default')
    matplotlib.rcParams.update({'font.size': 16})

    shap_values = [
        'SHAP Leucocytes',
        'SHAP CRP',
        'SHAP LacticAcid']

    fig11 = plt.figure(figsize=(16, 8), constrained_layout=False)
    grid = fig11.add_gridspec(3, 3, width_ratios=[2, 20, 0.2], wspace=0.2, hspace=0.0)

    for i, c in enumerate(shap_values):
        ax = fig11.add_subplot(grid[i, 1])
        ax.set_xlim([-0.1, 0.1])
        col = c.replace('SHAP ', '')
        X_tmp = X_all[X_all[col] > 0.]
        bins = np.linspace(X_tmp[col].min(), X_tmp[col].max(), 5)
        digitized = np.digitize(X_tmp[col], bins)

        palette = itertools.cycle(sns.color_palette("viridis"))
        for b in np.unique(digitized):
            X_dat = X_tmp[digitized == b].sample(frac=0.2, replace=False, random_state=0)
            sns.swarmplot(data=X_dat, x=c, color=next(palette), alpha=1., size=4, ax=ax)
        [s.set_visible(False) for s in ax.spines.values()]
        if i != (len(shap_values) - 1):
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            # ax.arrow(0., 0., 1., 0.)
            # ax.arrow(0., 0., -2., 0.)
            ax.set_xlabel('SHAP value (effect on model output)')
            # ax.set_xticklabels(['-1\n(Euglycemia)', '-0.5', '-0.25', '0', '2', '4', '6', '8\n(Hypoglycemia)'])
        ax = fig11.add_subplot(grid[i, 0])
        ax.text(0, 0.3, col)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        [s.set_visible(False) for s in ax.spines.values()]
        # if i == (len(top_n_shap_values) - 1):
        #     ax.text(0.0, -1.0, 'Likely EU')

    ax = fig11.add_subplot(grid[1:-1, 2])
    my_palplot(sns.color_palette("viridis"), ax=ax)
    ax.text(-4.2, 6.9, '   Low\nfeature\n  value')
    ax.text(-4.2, -1.2, '  High\nfeature\n  value')
    # ax.text(0.0, 5.5, 'Likely Hypo')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig11.tight_layout()
    plt.savefig(f'plots/{target_activity}_shap.svg', bbox_inches="tight")


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

def test_lstm(x_test_seq, x_test_stat, y_test, model):

    results = model.evaluate([x_test_seq, x_test_stat], y_test, batch_size=16)

    print(f'Test loss: {results[0]}')
    print(f'Test accuracy: {results[1]}')


def evaluate_on_cut(x_seqs_final, x_statics_final, y_final):
    matplotlib.style.use('default')
    matplotlib.rcParams.update({'font.size': 16})

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cut_lengths = [1, 5, 10, 15, 20, 25]
    results = {}
    for cut_len in cut_lengths:
        results[cut_len] = {}
        # aucs = []
        accs = []
        for train_index, test_index in skf.split(np.zeros(len(y_final)), y_final):
            X_train_seq = x_seqs_final[train_index][:, :cut_len, :]
            X_train_stat = x_statics_final[train_index]
            y_train = y_final[train_index]

            X_test_seq = x_seqs_final[test_index][:, :cut_len, :]
            X_test_stat = x_statics_final[test_index]
            y_test = y_final[test_index]

            model = train_lstm(X_train_seq, X_train_stat, y_train.reshape(-1, 1))
            pred = model.predict([X_test_seq, X_test_stat])
            # auc = metrics.roc_auc_score(y_test, pred)
            acc = metrics.accuracy_score(y_test, pred > 0.5)

            # aucs.append(auc)
            accs.append(acc)

        # results[cut_len]['auc'] = aucs
        results[cut_len]['acc'] = accs

    # Make plots
    fig, ax = plt.subplots(figsize=(14, 10))
    mean_line = [np.mean(results[c]['acc']) for c in cut_lengths]
    min_line = [np.percentile(results[c]['acc'], 25) for c in cut_lengths]
    max_line = [np.percentile(results[c]['acc'], 75) for c in cut_lengths]

    ax.plot(cut_lengths, mean_line)
    ax.fill_between(cut_lengths, min_line, max_line, alpha=.2)
    ax.set_xlabel('Number of Steps Considered for Prediction')
    ax.set_ylabel('Accuracy')
    plt.savefig(f'plots/{target_activity}_acc.svg')


def run_coefficient(x_seqs_final, x_statics_final, y_final):
    model = train_lstm(x_seqs_final, x_statics_final, y_final.reshape(-1, 1))
    output_weights = model.get_layer(name='output_layer').get_weights()[0].flatten()[2 * n_hidden:]
    # output_names = [f'Hidden State {i}' for i in range(2 * n_hidden)] + static_features
    output_names = static_features

    fig, ax = plt.subplots(figsize=(10, 10))
    y_pos = np.arange(len(output_names))
    ax.barh(y_pos, output_weights)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(output_names)
    ax.set_xlabel('Values of Coefficient in Last Layer')
    fig.tight_layout()
    plt.savefig(f'plots/{target_activity}_coefs.svg')

    return model


x_seqs_final, x_statics_final, y_final = get_data(target_activity)



# Run CV on cuts to plot results --> Figure 1
evaluate_on_cut(x_seqs_final, x_statics_final, y_final)

# Train model and plot linear coeff --> Figure 2
model = run_coefficient(x_seqs_final, x_statics_final, y_final)

# Get Explanations for LSTM inputs --> Figure 3
explainer = shap.DeepExplainer(model, [x_seqs_final, x_statics_final])
shap_values = explainer.shap_values([x_seqs_final, x_statics_final])

seqs_df = pd.DataFrame(data=x_seqs_final.reshape(-1, len(seq_features)),
                        columns=seq_features)
seq_shaps = pd.DataFrame(data=shap_values[0][0].reshape(-1, len(seq_features)),
                          columns=[f'SHAP {x}' for x in seq_features])
seq_value_shape = pd.concat([seqs_df, seq_shaps], axis=1)

compute_shap_summary_plot(seq_value_shape)

