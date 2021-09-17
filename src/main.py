#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:49:26 2021

@author: makraus, svwe
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
tf.compat.v1.disable_v2_behavior()
import matplotlib
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import shap
import src.data as data

data_set = "sepsis"  # sepsis; mimic
n_hidden = 8
max_len = 20  # we cut the extreme cases for runtime
min_len = 3
seed = False
num_repetitions = 1
mode = "complete"  # complete; static; sequential; dt, lr
train_size = 0.8





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


def time_step_blow_up(X_seq, X_stat, y, max_len, ts_info=False, x_time=None, x_time_vals=None):

    # blow up
    X_seq_prefix, X_stat_prefix, y_prefix, x_time_vals_prefix, ts = [], [], [], [], []
    for idx_seq in range(0, len(X_seq)):
        for idx_ts in range(1, len(X_seq[idx_seq]) + 1):
            X_seq_prefix.append(X_seq[idx_seq][0:idx_ts])
            X_stat_prefix.append(X_stat[idx_seq])
            y_prefix.append(y[idx_seq])
            if x_time is not None:
                x_time_vals_prefix.append(x_time_vals[idx_seq][0:idx_ts])
            ts.append(idx_ts)

    # remove prefixes with future event from training set
    if x_time is not None:
        X_seq_prefix_temp, X_stat_prefix_temp, y_prefix_temp, ts_temp = [], [], [], []

        for idx_prefix in range(0, len(X_seq_prefix)):
            if x_time_vals_prefix[idx_prefix][-1].value <= x_time.value:
                X_seq_prefix_temp.append(X_seq_prefix[idx_prefix])
                X_stat_prefix_temp.append(X_stat_prefix[idx_prefix])
                y_prefix_temp.append(y_prefix[idx_prefix])
                ts_temp.append(ts[idx_prefix])

        X_seq_prefix, X_stat_prefix, y_prefix, ts = X_seq_prefix_temp, X_stat_prefix_temp, y_prefix_temp, ts_temp

    # vectorization
    X_seq_final = np.zeros((len(X_seq_prefix), max_len, len(X_seq_prefix[0][0])), dtype=np.float32)
    X_stat_final = np.zeros((len(X_seq_prefix), len(X_stat_prefix[0])))
    for i, x in enumerate(X_seq_prefix):
        X_seq_final[i, :len(x), :] = np.array(x)
        X_stat_final[i, :] = np.array(X_stat_prefix[i])
    y_final = np.array(y_prefix).astype(np.int32)

    if ts_info:
        return X_seq_final, X_stat_final, y_final, ts
    else:
        return X_seq_final, X_stat_final, y_final


def evaluate_on_cut(x_seqs, x_statics, y, mode, target_activity, data_set, x_time=None):
    matplotlib.style.use('default')
    matplotlib.rcParams.update({'font.size': 16})

    data_index = list(range(0, len(y)))
    test_index = data_index[int(train_size * len(y)):]

    if x_time is not None:
        time_start_test = x_time[test_index[0]][0]
        x_time = x_time[0: int(train_size * len(y))]

    # model training
    results = {}

    for repetition in range(0, num_repetitions):

        # timestamp exists
        if x_time is not None:
            X_train_seq, X_train_stat, y_train = time_step_blow_up(x_seqs[0: int(train_size * len(y))],
                                                                       x_statics[0: int(train_size * len(y))],
                                                                       y[0: int(train_size * len(y))],
                                                                       max_len,
                                                                       ts_info=False,
                                                                       x_time=time_start_test,
                                                                       x_time_vals=x_time)
        # no timestamp exists
        else:
            X_train_seq, X_train_stat, y_train = time_step_blow_up(x_seqs[0: int(train_size * len(y))],
                                                                       x_statics[0: int(train_size * len(y))],
                                                                       y[0: int(train_size * len(y))],
                                                                       max_len)

        X_test_seq, X_test_stat, y_test, ts = time_step_blow_up(x_seqs[int(train_size * len(y)):],
                                                                    x_statics[int(train_size * len(y)):],
                                                                    y[int(train_size * len(y)):],
                                                                    max_len,
                                                                    ts_info=True)

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

        cut_lengths = range(min_len, max_len + 1)  # range(1, X_train_seq.shape[1] + 1)

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
                try:
                    results[cut_len]['auc'].append(metrics.roc_auc_score(y_true=results_temp_cut['gts'], y_score=results_temp_cut['preds_proba']))
                except:
                    pass

        # metrics across cuts
        results['all']['rep'].append(
            metrics.classification_report(y_true=results_temp['gts'], y_pred=results_temp['preds'], output_dict=True))
        results['all']['auc'].append(
            metrics.roc_auc_score(y_true=results_temp['gts'], y_score=results_temp['preds_proba']))

    # save all results
    f = open(f'../output/{data_set}_{mode}_{target_activity}_summary.txt', 'w')
    results_ = results
    del results_['preds'], results_['preds_proba'], results_['gts'], results_['ts']
    f.write(str(results_))
    f.close()


    # print metrics
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
            f.write(f'Std,{np.std(vals, ddof=1)}\n')
            print(f'Avg,{sum(vals) / len(vals)}')
            print(f'Std,{np.std(vals, ddof=1)}\n')
            f.close()

        elif metric_ == "accuracy":
            f = open(f'../output/{data_set}_{mode}_{target_activity}.txt', "a+")
            f.write(metric_ + '\n')
            print(metric_)
            for idx_ in range(0, num_repetitions):
                vals.append(results['all']['rep'][idx_][metric_])
                f.write(f'{idx_},{vals[-1]}\n')
                print(f'{idx_},{vals[-1]}')
            f.write(f'Avg,{sum(vals) / len(vals)}\n')
            f.write(f'Std,{np.std(vals, ddof=1)}\n')
            print(f'Avg,{sum(vals) / len(vals)}')
            print(f'Std,{np.std(vals, ddof=1)}\n')
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
                f.write(f'Std,{np.std(vals, ddof=1)}\n')
                print(f'Avg,{sum(vals) / len(vals)}')
                print(f'Std,{np.std(vals, ddof=1)}')
                f.close()

    X_seq = np.concatenate((X_train_seq, X_test_seq), axis=0)
    X_stat = np.concatenate((X_train_stat, X_test_stat), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    return X_seq, X_stat, y


def run_coefficient(x_seqs_final, x_statics_final, y_final, target_activity, static_features):

    model = train_lstm(x_seqs_final, x_statics_final, y_final)
    output_weights = model.get_layer(name='output_layer').get_weights()[0].flatten()[2 * n_hidden:]
    output_names = static_features

    f = open(f'../output/{data_set}_{mode}_{target_activity}_coef.txt', 'w')
    f.write(",".join([str(x) for x in output_names]) + '\n')
    f.write(",".join([str(x) for x in output_weights]))
    f.close()

    return model


if data_set == "sepsis":

    for mode in ['complete', 'static', 'sequential', 'dt', 'lr']:

        # Sepsis
        target_activity = 'Release A'
        # Release A: Very good
        # Release B: bad
        # Release C-E: Few samples
        # Admission IC: Good
        # Admission NC: Bad

        x_seqs, x_statics, y, x_time_vals_final, seq_features, static_features = data.get_sepsis_data(
            target_activity, max_len, min_len)

        # Run CV on cuts to plot results --> Figure 1
        x_seqs_final, x_statics_final, y_final = evaluate_on_cut(x_seqs, x_statics, y, mode, target_activity,
                                                                 data_set, x_time_vals_final)

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

            with open(f'../output/{data_set}_{mode}_{target_activity}_shap.npy', 'wb') as f:
                pickle.dump(seq_value_shape, f)


elif data_set == "mimic":

    # MIMIC
    target_activity = 'Emergency Department Observation'
    # Emergency Department Observation:
    # Hematology/Oncology:
    # Medicine/Cardiology
    # Transplant
    # Med/Surg

    x_seqs_final, x_statics_final, y_final, seq_features, static_features = data.get_data_mimic(target_activity,
                                                                                                max_len, min_len)

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

        with open(f'../output/{data_set}_{mode}_{target_activity}_shap.npy', 'wb') as f:
            pickle.dump(seq_value_shape, f)

        # compute_shap_summary_plot(seq_value_shape, data_set)

else:
    print("Data set not available!")

