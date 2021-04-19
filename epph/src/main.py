#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:49:26 2021

@author: makraus, svwe
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
# target_activity_abbreviation = 'REA'

seed_val = 1377
seed = True
num_folds = 10

mode = "complete"  # complete; static; sequential; dt, lg

if seed:
    np.random.seed(1377)
    tf.random.set_seed(1377)

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
diagnose_mapping = dict(zip(df['Diagnose'].unique(), np.arange(len(df['Diagnose'].unique()))))  # ordinal encoding
df['Diagnose'] = df['Diagnose'].apply(lambda x: diagnose_mapping[x])
df['Diagnose'] = df['Diagnose'].apply(lambda x: x / max(df['Diagnose']))  # normalise ordinal encoding
df['Age'] = df['Age'].fillna(-1)
df['Age'] = df['Age'].apply(lambda x: x / max(df['Age']))

max_leucocytes = np.percentile(df['Leucocytes'].dropna(), 95)  # remove outliers
max_lacticacid = np.percentile(df['LacticAcid'].dropna(), 95)  # remove outliers

x_seqs = []
x_statics = []
y = []

max_len = 20  # we cut the extreme cases for runtime


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

    print(f'Cutting everything after {max_len} activities')
    x_seqs_final = np.zeros((len(x_seqs), max_len, len(x_seqs[0][0])), dtype=np.float32)
    for i, x in enumerate(x_seqs):
        if len(x) > 0:
            x_seqs_final[i, :min(len(x), max_len), :] = np.array(x[:max_len])
    x_statics_final = np.array(x_statics)
    y_final = np.array(y).astype(np.int32)

    return x_seqs_final, x_statics_final, y_final


def my_palplot(pal, size=1, ax=None):
    n = 5
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(n * size, size))
    ax.imshow(np.arange(n)[::-1].reshape(n, 1),
              cmap=matplotlib.colors.ListedColormap(list(pal)),
              interpolation="nearest", aspect="auto")


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
            if seed:
                X_dat = X_tmp[digitized == b].sample(frac=0.2, replace=False, random_state=seed_val)
            else:
                X_dat = X_tmp[digitized == b].sample(frac=0.2, replace=False, random_state=None)
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
    plt.savefig(f'../plots/{target_activity}_shap.svg', bbox_inches="tight")


def train_lstm(x_train_seq, x_train_stat, y_train, mode="complete"):
    max_case_len = x_train_seq.shape[1]
    num_features_seq = x_train_seq.shape[2]
    num_features_stat = x_train_stat.shape[1]

    if mode == "complete":
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

    if mode == "static":
        # Bidirectional LSTM
        # Input layer
        input_layer_static = tf.keras.layers.Input(shape=(num_features_stat), name='static_input_layer')

        # Output layer
        output_layer = tf.keras.layers.Dense(1,
                                             activation='sigmoid',
                                             name='output_layer')(input_layer_static)

        model = tf.keras.models.Model(inputs=[input_layer_static],
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
        model.fit([x_train_stat], y_train,
                  validation_split=0.1,
                  verbose=1,
                  callbacks=[early_stopping, model_checkpoint, lr_reducer],
                  batch_size=16,
                  epochs=100)

    if mode == "sequential":
        # Bidirectional LSTM
        # Input layer
        input_layer_seq = tf.keras.layers.Input(shape=(max_case_len, num_features_seq), name='seq_input_layer')

        # Hidden layer
        hidden_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=n_hidden,
            return_sequences=False))(input_layer_seq)

        # Output layer
        output_layer = tf.keras.layers.Dense(1,
                                             activation='sigmoid',
                                             name='output_layer')(hidden_layer)

        model = tf.keras.models.Model(inputs=[input_layer_seq],
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
        model.fit([x_train_seq], y_train,
                  validation_split=0.1,
                  verbose=1,
                  callbacks=[early_stopping, model_checkpoint, lr_reducer],
                  batch_size=16,
                  epochs=100)

    return model


def time_step_blow_up(X_seq, X_stat, y, ts_info=False):
    X_seq_ts = np.zeros((X_seq.shape[0] * X_seq.shape[1], X_seq.shape[1], X_seq.shape[2]))
    X_stat_ts = np.zeros((X_stat.shape[0] * X_seq.shape[1], X_stat.shape[1]))
    y_ts = np.zeros((len(y) * X_seq.shape[1]))

    idx = 0
    ts = []
    for idx_seq in range(0, X_seq.shape[0]):
        for idx_ts in range(1, X_seq.shape[1] + 1):
            X_seq_ts[idx, :idx_ts, :] = X_seq[idx_seq, :idx_ts, :]
            X_stat_ts[idx] = X_stat[idx_seq]
            y_ts[idx] = y[idx_seq]
            ts.append(idx_ts)
            idx += 1

    if ts_info:
        return X_seq_ts, X_stat_ts, y_ts, ts
    else:
        return X_seq_ts, X_stat_ts, y_ts


def evaluate_on_cut(x_seqs_final, x_statics_final, y_final, mode):
    matplotlib.style.use('default')
    matplotlib.rcParams.update({'font.size': 16})

    if seed:
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed_val)
    else:
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=None)

    # model training
    results = {}
    for train_index, test_index in skf.split(np.zeros(len(y_final)), y_final):

        X_train_seq, X_train_stat, y_train = time_step_blow_up(x_seqs_final[train_index],
                                                               x_statics_final[train_index],
                                                               y_final[train_index])

        X_test_seq, X_test_stat, y_test, ts = time_step_blow_up(x_seqs_final[test_index],
                                                                x_statics_final[test_index],
                                                                y_final[test_index], ts_info=True)

        if mode == "complete":
            model = train_lstm(X_train_seq, X_train_stat, y_train.reshape(-1, 1), mode)
            preds_proba = model.predict([X_test_seq, X_test_stat])

        elif mode == "static":
            model = train_lstm(X_train_seq, X_train_stat, y_train.reshape(-1, 1), mode)
            preds_proba = model.predict([X_test_stat])

        elif mode == "sequential":
            model = train_lstm(X_train_seq, X_train_stat, y_train.reshape(-1, 1), mode)
            preds_proba = model.predict([X_test_seq])

        results['preds'] = [int(round(pred[0])) for pred in preds_proba]
        results['preds_proba'] = [pred_proba[0] for pred_proba in preds_proba]
        results['gts'] = [int(y) for y in y_test]
        results['ts'] = ts

        results_temp = pd.DataFrame(
            list(zip(results['ts'],
                     results['preds'],
                     results['preds_proba'],
                     results['gts'])),
            columns=['ts', 'preds', 'preds_proba', 'gts'])

        cut_lengths = range(1, X_train_seq.shape[1] + 1)

        # init
        if cut_lengths[0] not in results:
            for cut_len in cut_lengths:
                results[cut_len] = {}
                results[cut_len]['acc'] = list()
                results[cut_len]['auc'] = list()
                results['all'] = {}
                results['all']['rep'] = list()
                results['all']['auc'] = list()

        # metrics per cut
        for cut_len in cut_lengths:
            results_temp_cut = results_temp[results_temp.ts == cut_len]

            if not results_temp_cut.empty:  # if cut length is longer than max trace length
                results[cut_len]['acc'].append(
                    metrics.accuracy_score(y_true=results_temp_cut['gts'], y_pred=results_temp_cut['preds']))
                results[cut_len]['auc'].append(
                    metrics.roc_auc_score(y_true=results_temp_cut['gts'], y_score=results_temp_cut['preds_proba']))

        # metrics across cuts
        results['all']['rep'].append(
            metrics.classification_report(y_true=results_temp['gts'], y_pred=results_temp['preds'], output_dict=True))
        results['all']['auc'].append(
            metrics.roc_auc_score(y_true=results_temp['gts'], y_score=results_temp['preds_proba']))

    # print accuracy plot
    fig, ax = plt.subplots(figsize=(14, 10))
    mean_line = [np.mean(results[c]['acc']) for c in cut_lengths]
    min_line = [np.percentile(results[c]['acc'], 25) for c in cut_lengths]
    max_line = [np.percentile(results[c]['acc'], 75) for c in cut_lengths]
    ax.plot(cut_lengths, mean_line)
    ax.fill_between(cut_lengths, min_line, max_line, alpha=.2)
    # ax.set_title(r'$M_{%s}$' % target_activity_abbreviation, fontsize=30)
    ax.set_xlabel('Size of Process Instance Prefix for Prediction', fontsize=20)
    ax.set_xticks(np.arange(1, max_len + 1, step=2))
    ax.set_ylabel('Accuracy', fontsize=20)
    ax.set_ylim(0.0, 1)
    plt.savefig(f'../plots/{target_activity}_acc.svg')

    # print auc roc plot
    fig, ax = plt.subplots(figsize=(14, 10))
    mean_line = [np.mean(results[c]['auc']) for c in cut_lengths]
    min_line = [np.percentile(results[c]['auc'], 25) for c in cut_lengths]
    max_line = [np.percentile(results[c]['auc'], 75) for c in cut_lengths]
    ax.plot(cut_lengths, mean_line)
    ax.fill_between(cut_lengths, min_line, max_line, alpha=.2)
    # ax.set_title(r'$M_{%s}$' % target_activity_abbreviation, fontsize=30)
    ax.set_xlabel('Size of Process Instance Prefix for Prediction', fontsize=20)
    ax.set_xticks(np.arange(1, max_len + 1, step=2))
    ax.set_ylabel(r'$AUC_{ROC}$', fontsize=20)
    ax.set_ylim(0.4, 0.9)
    plt.savefig(f'../plots/{target_activity}_auc.svg')

    # print metrics across cuts
    metrics_ = ["auc", "precision", "recall", "f1-score", "support", "accuracy"]
    labels = ["0", "1"]

    for metric_ in metrics_:
        vals = []
        if metric_ == "auc":
            for idx_fold in range(0, num_folds):
                vals.append(results['all']['auc'][idx_fold])
            print("Avg. value of metric %s: %s" % (metric_, sum(vals) / len(vals)))
        elif metric_ == "accuracy":
            for idx_fold in range(0, num_folds):
                vals.append(results['all']['rep'][idx_fold][metric_])
            print("Avg. value of metric %s: %s" % (metric_, sum(vals) / len(vals)))
        else:
            for label in labels:
                for idx_fold in range(0, num_folds):
                    vals.append(results['all']['rep'][idx_fold][label][metric_])
                print("Avg. value of metric %s for label %s: %s" % (metric_, label, sum(vals) / len(vals)))


def run_coefficient(x_seqs_final, x_statics_final, y_final):
    x_seqs_final, x_statics_final, y_final = time_step_blow_up(x_seqs_final,
                                                               x_statics_final,
                                                               y_final.reshape(-1, 1))

    model = train_lstm(x_seqs_final, x_statics_final, y_final)
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
    plt.savefig(f'../plots/{target_activity}_coefs.svg')

    return model


x_seqs_final, x_statics_final, y_final = get_data(target_activity)

# Run CV on cuts to plot results --> Figure 1
evaluate_on_cut(x_seqs_final, x_statics_final, y_final, mode)

if mode == "complete":
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
