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
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import shap
import itertools
import src.data as data

data_set = "sepsis"  # sepsis
n_hidden = 8
max_len = 20  # we cut the extreme cases for runtime
seed = False
num_repetitions = 10
mode = "dt"  # complete; static; sequential; dt, lr
train_size = 0.8


def my_palplot(pal, size=1, ax=None):
    n = 5
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(n * size, size))
    ax.imshow(np.arange(n)[::-1].reshape(n, 1),
              cmap=matplotlib.colors.ListedColormap(list(pal)),
              interpolation="nearest", aspect="auto")


def compute_shap_summary_plot(X_all, data_set):
    matplotlib.style.use('default')
    matplotlib.rcParams.update({'font.size': 16})

    if data_set == "sepsis":

        shap_values = [
            'SHAP Leucocytes',
            'SHAP CRP',
            'SHAP LacticAcid',
            'SHAP ER Triage',
            # 'SHAP ER Sepsis Triage',
            'SHAP IV Liquid',
            'SHAP IV Antibiotics'
            # 'SHAP Admission NC',
            # 'SHAP Admission IC',
            # 'SHAP Return ER',
            # 'SHAP Release A',
            # 'SHAP Release B',
            # 'SHAP Release C',
            # 'SHAP Release D',
            # 'SHAP Release E',
        ]

    elif data_set == "mimic":

        shap_values = [
            'Neurology',
            'Vascular',
            'Medicine'
        ]

    else:
        print("Data set not available!")

    fig11 = plt.figure(figsize=(16, 14), constrained_layout=False)  # 16, 8
    grid = fig11.add_gridspec(6, 3, width_ratios=[2, 20, 0.2], wspace=0.2, hspace=0.0)  # 3,3

    for i, c in enumerate(shap_values):
        ax = fig11.add_subplot(grid[i, 1])
        ax.set_xlim([-0.1, 0.1])  # -0.1, 0.1
        col = c.replace('SHAP ', '')
        X_tmp = X_all[X_all[col] > 0.]
        bins = np.linspace(X_tmp[col].min(), X_tmp[col].max(), 5)
        digitized = np.digitize(X_tmp[col], bins)

        palette = itertools.cycle(sns.color_palette("viridis"))
        for b in np.unique(digitized):
            # if seed:
            #    X_dat = X_tmp[digitized == b].sample(frac=0.2, replace=False, random_state=seed_val)
            # else:
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
            ax.set_xlabel('SHAP Value (Effect on Model Output)')
            # ax.set_xticklabels(
            # ['-1\n(Euglycemia)', '-0.5', '-0.25', '0', '2', '4', '6', '8\n(Hypoglycemia)'])
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
    ax.text(-4.2, 5.6, '   Low\nFeature\n  Value')  # 6.9
    ax.text(-4.2, -1.2, '  High\nFeature\n  Value')  # -1.2
    # ax.text(0.0, 5.5, 'Likely Hypo')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig11.tight_layout()
    plt.savefig(f'../plots/{target_activity}_shap.svg', bbox_inches="tight")


def concatenate_tensor_matrix(x_seq, x_stat):
    x_train_seq_ = x_seq.reshape(-1, x_seq.shape[1] * x_seq.shape[2])
    x_concat = np.concatenate((x_train_seq_, x_stat), axis=1)

    return x_concat


def train_dt(x_train_seq, x_train_stat, y_train):
    x_concat = concatenate_tensor_matrix(x_train_seq, x_train_stat)

    model = DecisionTreeClassifier()
    model.fit(x_concat, y_train)

    return model


def train_lr(x_train_seq, x_train_stat, y_train):
    x_concat = concatenate_tensor_matrix(x_train_seq, x_train_stat)

    model = LogisticRegression()
    model.fit(x_concat, np.ravel(y_train))

    return model


def train_lstm(x_train_seq, x_train_stat, y_train, mode="complete"):
    max_case_len = x_train_seq.shape[1]
    num_features_seq = x_train_seq.shape[2]
    num_features_stat = x_train_stat.shape[1]

    if mode == "complete":
        # Input layer
        input_layer_seq = tf.keras.layers.Input(shape=(max_case_len, num_features_seq), name='seq_input_layer')
        input_layer_static = tf.keras.layers.Input(shape=(num_features_stat), name='static_input_layer')

        hidden_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=n_hidden,
            return_sequences=False))(input_layer_seq)

        concatenate_layer = tf.keras.layers.Concatenate(axis=1)([hidden_layer, input_layer_static])

        # Output layer
        output_layer = tf.keras.layers.Dense(1,
                                             activation='sigmoid',
                                             name='output_layer')(concatenate_layer)

        model = tf.keras.models.Model(inputs=[input_layer_seq, input_layer_static],
                                      outputs=[output_layer])

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
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
                      optimizer='nadam',
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


def evaluate_on_cut(x_seqs_final, x_statics_final, y_final, mode, target_activity, data_set, x_time_vals_final=None):
    matplotlib.style.use('default')
    matplotlib.rcParams.update({'font.size': 16})

    data_index = list(range(0, len(y_final)))
    train_index = data_index[0: int(train_size * len(y_final))]
    test_index = data_index[int(train_size * len(y_final)):]

    print(0)

    # model training
    results = {}

    for repetition in range(0, num_repetitions):
        X_train_seq, X_train_stat, y_train = time_step_blow_up(x_seqs_final[train_index],
                                                               x_statics_final[train_index],
                                                               y_final[train_index])

        X_test_seq, X_test_stat, y_test, ts = time_step_blow_up(x_seqs_final[test_index],
                                                                x_statics_final[test_index],
                                                                y_final[test_index], ts_info=True)

        if mode == "complete":
            model = train_lstm(X_train_seq, X_train_stat, y_train.reshape(-1, 1), mode)
            preds_proba = model.predict([X_test_seq, X_test_stat])
            results['preds'] = [int(round(pred[0])) for pred in preds_proba]
            results['preds_proba'] = [pred_proba[0] for pred_proba in preds_proba]

        elif mode == "static":
            model = train_lstm(X_train_seq, X_train_stat, y_train.reshape(-1, 1), mode)
            preds_proba = model.predict([X_test_stat])
            results['preds'] = [int(round(pred[0])) for pred in preds_proba]
            results['preds_proba'] = [pred_proba[0] for pred_proba in preds_proba]

        elif mode == "sequential":
            model = train_lstm(X_train_seq, X_train_stat, y_train.reshape(-1, 1), mode)
            preds_proba = model.predict([X_test_seq])
            results['preds'] = [int(round(pred[0])) for pred in preds_proba]
            results['preds_proba'] = [pred_proba[0] for pred_proba in preds_proba]

        elif mode == "dt":
            model = train_dt(X_train_seq, X_train_stat, y_train.reshape(-1, 1))
            preds_proba = model.predict_proba(concatenate_tensor_matrix(X_test_seq, X_test_stat))
            results['preds'] = [np.argmax(pred_proba) for pred_proba in preds_proba]
            results['preds_proba'] = [pred_proba[1] for pred_proba in preds_proba]

        elif mode == "lr":
            model = train_lr(X_train_seq, X_train_stat, y_train.reshape(-1, 1))
            preds_proba = model.predict_proba(concatenate_tensor_matrix(X_test_seq, X_test_stat))
            results['preds'] = [np.argmax(pred_proba) for pred_proba in preds_proba]
            results['preds_proba'] = [pred_proba[1] for pred_proba in preds_proba]

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
            f = open(f'../output/{data_set}_{mode}_{target_activity}.txt', "a+")
            f.write(metric_ + '\n')
            print(metric_)
            for idx_ in range(0, num_repetitions):
                vals.append(results['all']['auc'][idx_])
                f.write(f'{idx_},{vals[-1]}\n')
                print(f'{idx_},{vals[-1]}')
            f.write(f'Avg,{sum(vals) / len(vals)}\n')
            print(f'Avg.: {sum(vals) / len(vals)}')
            f.close()

        elif metric_ == "accuracy":
            f = open(f'../output/{data_set}_{mode}_{target_activity}.txt', "a+")
            f.write(metric_ + '\n')
            print(metric_)
            for idx_ in range(0, num_repetitions):
                vals.append(results['all']['rep'][idx_][metric_])
                print(f'{idx_},{vals[-1]}')
                f.write(f'{idx_},{vals[-1]}')
            f.write(f'Avg,{sum(vals) / len(vals)}\n')
            print(f'Avg.: {sum(vals) / len(vals)}')
            f.close()
        else:
            for label in labels:
                f = open(f'../output/{data_set}_{mode}_{target_activity}.txt', "a+")
                f.write(metric_ + f' ({label})\n')
                print(metric_ + f' ({label})')
                vals = []
                for idx_ in range(0, num_repetitions):
                    vals.append(results['all']['rep'][idx_][label][metric_])
                    f.write(f'{idx_},{vals[-1]}\n')
                    print(f'{idx_},{vals[-1]}')
                f.write(f'Avg,{sum(vals) / len(vals)}\n')
                print(f'Avg.: {sum(vals) / len(vals)}')
                f.close()


def run_coefficient(x_seqs_final, x_statics_final, y_final, target_activity, static_features):
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
    ax.set_xlabel('Coefficient Value in Last Layer')
    ax.set_xlim(-3.0, 3.0)
    fig.tight_layout()
    plt.savefig(f'../plots/{target_activity}_coefs.svg')

    return model


if data_set == "sepsis":

    # Sepsis
    target_activity = 'Release A'
    # Release A: Very good
    # Release B: bad
    # Release C-E: Few samples
    # Admission IC: Good
    # Admission NC: Bad

    x_seqs_final, x_statics_final, y_final, x_time_vals_final, seq_features, static_features = data.get_sepsis_data(
        target_activity, max_len)

    # Run CV on cuts to plot results --> Figure 1
    evaluate_on_cut(x_seqs_final, x_statics_final, y_final, mode, target_activity, data_set, x_time_vals_final)

    if mode == "complete":
        # Train model and plot linear coeff --> Figure 2
        model = run_coefficient(x_seqs_final, x_statics_final, y_final, target_activity, static_features)

        # Get Explanations for LSTM inputs --> Figure 3
        explainer = shap.DeepExplainer(model, [x_seqs_final, x_statics_final])
        shap_values = explainer.shap_values([x_seqs_final, x_statics_final])

        seqs_df = pd.DataFrame(data=x_seqs_final.reshape(-1, len(seq_features)),
                               columns=seq_features)
        seq_shaps = pd.DataFrame(data=shap_values[0][0].reshape(-1, len(seq_features)),
                                 columns=[f'SHAP {x}' for x in seq_features])
        seq_value_shape = pd.concat([seqs_df, seq_shaps], axis=1)

        compute_shap_summary_plot(seq_value_shape, data_set)

elif data_set == "mimic":

    # MIMIC
    target_activity = 'Emergency Department Observation'
    # Emergency Department Observation: medium

    x_seqs_final, x_statics_final, y_final, seq_features, static_features = data.get_data_mimic(target_activity,
                                                                                                max_len)

    # Run CV on cuts to plot results --> Figure 1
    evaluate_on_cut(x_seqs_final, x_statics_final, y_final, mode, target_activity, data_set)

    if mode == "complete":
        # Train model and plot linear coeff --> Figure 2
        model = run_coefficient(x_seqs_final, x_statics_final, y_final, target_activity, static_features)

        # Get Explanations for LSTM inputs --> Figure 3
        explainer = shap.DeepExplainer(model, [x_seqs_final, x_statics_final])
        shap_values = explainer.shap_values([x_seqs_final, x_statics_final])

        seqs_df = pd.DataFrame(data=x_seqs_final.reshape(-1, len(seq_features)),
                               columns=seq_features)
        seq_shaps = pd.DataFrame(data=shap_values[0][0].reshape(-1, len(seq_features)),
                                 columns=[f'SHAP {x}' for x in seq_features])
        seq_value_shape = pd.concat([seqs_df, seq_shaps], axis=1)

        compute_shap_summary_plot(seq_value_shape, data_set)

else:
    print("Data set not available!")
