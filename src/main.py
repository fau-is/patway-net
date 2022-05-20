import pandas as pd
import numpy as np
import sklearn.tree
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()
import pickle
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import os
import shap
import src.data as data
import torch
from src.interpret_LSTM import Net, NaiveCustomLSTM

data_set = "sepsis"  # sepsis; mimic
n_hidden = 8
max_len = 50  # mimic=84; sepsis=100
min_len = 5
min_size_prefix = 1
seed = False
num_repetitions = 1
mode = "test"
val_size = 0.2
train_size = 0.8

hpo = True


def concatenate_tensor_matrix(x_seq, x_stat):
    x_train_seq_ = x_seq.reshape(-1, x_seq.shape[1] * x_seq.shape[2])
    x_concat = np.concatenate((x_train_seq_, x_stat), axis=1)

    return x_concat


def train_rf(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo):
    x_concat_train = concatenate_tensor_matrix(x_train_seq, x_train_stat)
    x_concat_val = concatenate_tensor_matrix(x_val_seq, x_val_stat)

    # Use only stat features
    x_concat_train = x_train_stat
    x_concat_val = x_val_stat

    if hpo:
        best_model = ""
        best_hpos = ""
        aucs = []

        for num_trees in hpos["rf"]["num_trees"]:
            for max_depth_trees in hpos["rf"]["max_depth_trees"]:
                for num_rand_vars in hpos["rf"]["num_rand_vars"]:

                    model = RandomForestClassifier(n_estimators=num_trees, max_depth=max_depth_trees,
                                                   max_features=num_rand_vars)
                    model.fit(x_concat_train, np.ravel(y_train))
                    preds_proba = model.predict_proba(x_concat_val)
                    preds_proba = [pred_proba[1] for pred_proba in preds_proba]
                    auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                    aucs.append(auc)

                    if auc >= max(aucs):
                        best_model = model
                        best_hpos = {"num_trees": num_trees, "max_depth_trees": max_depth_trees,
                                     "num_rand_vars": num_rand_vars}

        f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos.txt', 'a+')
        f.write(str(best_hpos))
        f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
        f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
        f.write(f'Std,{np.std(aucs, ddof=1)}\n')
        f.close()

        return best_model, best_hpos

    else:
        x_concat = np.concatenate((x_concat_train, x_concat_val), axis=0)
        y = np.concatenate((y_train, y_val), axis=0)

        model = RandomForestClassifier()
        model.fit(x_concat, np.ravel(y))

        return model


def train_lr(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo):
    x_concat_train = concatenate_tensor_matrix(x_train_seq, x_train_stat)
    x_concat_val = concatenate_tensor_matrix(x_val_seq, x_val_stat)

    # Use only stat features
    x_concat_train = x_train_stat
    x_concat_val = x_val_stat

    if hpo:
        best_model = ""
        best_hpos = ""
        aucs = []

        for c in hpos["lr"]["reg_strength"]:
            for solver in hpos["lr"]["solver"]:

                model = LogisticRegression(C=c, solver=solver)
                model.fit(x_concat_train, np.ravel(y_train))
                preds_proba = model.predict_proba(x_concat_val)
                preds_proba = [pred_proba[1] for pred_proba in preds_proba]
                auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                aucs.append(auc)

                if auc >= max(aucs):
                    best_model = model
                    best_hpos = {"c": c, "solver": solver}

        f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos.txt', 'a+')
        f.write(str(best_hpos))
        f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
        f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
        f.write(f'Std,{np.std(aucs, ddof=1)}\n')
        f.close()

        return best_model, best_hpos

    else:
        x_concat = np.concatenate((x_concat_train, x_concat_val), axis=0)
        y = np.concatenate((y_train, y_val), axis=0)

        model = LogisticRegression()
        model.fit(x_concat, np.ravel(y))

        return model


def train_svm(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo):
    x_concat_train = concatenate_tensor_matrix(x_train_seq, x_train_stat)
    x_concat_val = concatenate_tensor_matrix(x_val_seq, x_val_stat)

    # Use only stat features
    x_concat_train = x_train_stat
    x_concat_val = x_val_stat

    if hpo:
        best_model = ""
        best_hpos = ""
        aucs = []

        for kern_fkt in hpos["svm"]["kern_fkt"]:
            for cost in hpos["svm"]["cost"]:

                model = SVC(C=cost, kernel=kern_fkt, probability=True)
                model.fit(x_concat_train, np.ravel(y_train))
                preds_proba = model.predict_proba(x_concat_val)
                preds_proba = [pred_proba[1] for pred_proba in preds_proba]
                auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                aucs.append(auc)

                if auc >= max(aucs):
                    best_model = model
                    best_hpos = {"cost": cost, "kernel": kern_fkt}

        f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos.txt', 'a+')
        f.write(str(best_hpos))
        f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
        f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
        f.write(f'Std,{np.std(aucs, ddof=1)}\n')
        f.close()

        return best_model, best_hpos

    else:
        x_concat = np.concatenate((x_concat_train, x_concat_val), axis=0)
        y = np.concatenate((y_train, y_val), axis=0)

        model = SVC(probability=True)
        model.fit(x_concat, np.ravel(y))

        return model


def train_gb(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo):
    x_concat_train = concatenate_tensor_matrix(x_train_seq, x_train_stat)
    x_concat_val = concatenate_tensor_matrix(x_val_seq, x_val_stat)

    # Use only stat features
    x_concat_train = x_train_stat
    x_concat_val = x_val_stat

    if hpo:
        best_model = ""
        best_hpos = ""
        aucs = []

        for n_estimators in hpos["gb"]["n_estimators"]:
            for learning_rate in hpos["gb"]["learning_rate"]:

                model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
                model.fit(x_concat_train, np.ravel(y_train))
                preds_proba = model.predict_proba(x_concat_val)
                preds_proba = [pred_proba[1] for pred_proba in preds_proba]
                auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                aucs.append(auc)

                if auc >= max(aucs):
                    best_model = model
                    best_hpos = {"n_estimators": n_estimators, "learning_rate": learning_rate}

        f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos.txt', 'a+')
        f.write(str(best_hpos))
        f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
        f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
        f.write(f'Std,{np.std(aucs, ddof=1)}\n')
        f.close()

        return best_model, best_hpos

    else:
        x_concat = np.concatenate((x_concat_train, x_concat_val), axis=0)
        y = np.concatenate((y_train, y_val), axis=0)

        model = GradientBoostingClassifier()
        model.fit(x_concat, np.ravel(y))

        return model


def train_ada(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo):
    x_concat_train = concatenate_tensor_matrix(x_train_seq, x_train_stat)
    x_concat_val = concatenate_tensor_matrix(x_val_seq, x_val_stat)

    # Use only stat features
    x_concat_train = x_train_stat
    x_concat_val = x_val_stat

    if hpo:
        best_model = ""
        best_hpos = ""
        aucs = []

        for n_estimators in hpos["ada"]["n_estimators"]:
            for learning_rate in hpos["ada"]["learning_rate"]:

                model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
                model.fit(x_concat_train, np.ravel(y_train))
                preds_proba = model.predict_proba(x_concat_val)
                preds_proba = [pred_proba[1] for pred_proba in preds_proba]
                auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                aucs.append(auc)

                if auc >= max(aucs):
                    best_model = model
                    best_hpos = {"n_estimators": n_estimators, "learning_rate": learning_rate}

        f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos.txt', 'a+')
        f.write(str(best_hpos) + '\n')
        f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
        f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
        f.write(f'Std,{np.std(aucs, ddof=1)}\n')
        f.close()

        return best_model, best_hpos

    else:
        x_concat = np.concatenate((x_concat_train, x_concat_val), axis=0)
        y = np.concatenate((y_train, y_val), axis=0)

        model = AdaBoostClassifier()
        model.fit(x_concat, np.ravel(y))

        return model


def train_nb(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo):
    x_concat_train = concatenate_tensor_matrix(x_train_seq, x_train_stat)
    x_concat_val = concatenate_tensor_matrix(x_val_seq, x_val_stat)

    # Use only stat features
    x_concat_train = x_train_stat
    x_concat_val = x_val_stat

    if hpo:
        best_model = ""
        best_hpos = ""
        aucs = []

        for var_smoothing in hpos["nb"]["var_smoothing"]:

            model = GaussianNB(var_smoothing=var_smoothing)
            model.fit(x_concat_train, np.ravel(y_train))
            preds_proba = model.predict_proba(x_concat_val)
            preds_proba = [pred_proba[1] for pred_proba in preds_proba]
            auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
            aucs.append(auc)

            if auc >= max(aucs):
                best_model = model
                best_hpos = {"var_smoothing": var_smoothing}

        f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos.txt', 'a+')
        f.write(str(best_hpos) + '\n')
        f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
        f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
        f.write(f'Std,{np.std(aucs, ddof=1)}\n')
        f.close()

        return best_model, best_hpos

    else:
        x_concat = np.concatenate((x_concat_train, x_concat_val), axis=0)
        y = np.concatenate((y_train, y_val), axis=0)

        model = GaussianNB()
        model.fit(x_concat, np.ravel(y))

        return model


def train_dt(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo):
    x_concat_train = concatenate_tensor_matrix(x_train_seq, x_train_stat)
    x_concat_val = concatenate_tensor_matrix(x_val_seq, x_val_stat)

    # Use only stat features
    x_concat_train = x_train_stat
    x_concat_val = x_val_stat

    if hpo:
        best_model = ""
        best_hpos = ""
        aucs = []

        for max_depth in hpos["dt"]["max_depth"]:
            for min_samples_split in hpos["dt"]["min_samples_split"]:

                model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
                model.fit(x_concat_train, np.ravel(y_train))
                preds_proba = model.predict_proba(x_concat_val)
                preds_proba = [pred_proba[1] for pred_proba in preds_proba]
                auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                aucs.append(auc)

                if auc >= max(aucs):
                    best_model = model
                    best_hpos = {"max_depth": max_depth, "min_samples_split": min_samples_split}

        f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos.txt', 'a+')
        f.write(str(best_hpos) + '\n')
        f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
        f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
        f.write(f'Std,{np.std(aucs, ddof=1)}\n')
        f.close()

        return best_model, best_hpos

    else:
        x_concat = np.concatenate((x_concat_train, x_concat_val), axis=0)
        y = np.concatenate((y_train, y_val), axis=0)

        model = DecisionTreeClassifier()
        model.fit(x_concat, np.ravel(y))

        return model


def train_knn(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo):
    x_concat_train = concatenate_tensor_matrix(x_train_seq, x_train_stat)
    x_concat_val = concatenate_tensor_matrix(x_val_seq, x_val_stat)

    # Use only stat features
    x_concat_train = x_train_stat
    x_concat_val = x_val_stat

    if hpo:
        best_model = ""
        best_hpos = ""
        aucs = []

        for n_neighbors in hpos["knn"]["n_neighbors"]:

            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            model.fit(x_concat_train, np.ravel(y_train))
            preds_proba = model.predict_proba(x_concat_val)
            preds_proba = [pred_proba[1] for pred_proba in preds_proba]
            auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
            aucs.append(auc)

            if auc >= max(aucs):
                best_model = model
                best_hpos = {"n_eighbors": n_neighbors}

        f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos.txt', 'a+')
        f.write(str(best_hpos) + '\n')
        f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
        f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
        f.write(f'Std,{np.std(aucs, ddof=1)}\n')
        f.close()

        return best_model, best_hpos

    else:
        x_concat = np.concatenate((x_concat_train, x_concat_val), axis=0)
        y = np.concatenate((y_train, y_val), axis=0)

        model = KNeighborsClassifier()
        model.fit(x_concat, np.ravel(y))

        return model


def train_lstm(x_train_seq, x_train_stat, y_train, x_val_seq=False, x_val_stat=False, y_val=False, hpos=False,
               hpo=False, mode="complete"):
    max_case_len = x_train_seq.shape[1]
    num_features_seq = x_train_seq.shape[2]
    num_features_stat = x_train_stat.shape[1]

    if mode == "test":
        import torch
        import torch.nn as nn
        import torch.optim as optim

        if hpo:
            best_model = ""
            best_hpos = ""
            aucs = []

            x_train_seq = torch.from_numpy(x_train_seq)
            x_train_stat = torch.from_numpy(x_train_stat)
            y_train = torch.from_numpy(y_train)

            for learning_rate in hpos["test"]["learning_rate"]:
                for batch_size in hpos["test"]["batch_size"]:
                    for seq_feature_sz in hpos["test"]["seq_feature_sz"]:
                        for stat_feature_sz in hpos["test"]["stat_feature_sz"]:
                            for inter_seq_best in hpos["test"]["inter_seq_best"]:

                                model = Net(input_sz_seq=num_features_seq,
                                            hidden_per_seq_feat_sz=seq_feature_sz,
                                            interactions_seq=[],
                                            interactions_seq_itr=100,
                                            interactions_seq_best=inter_seq_best,
                                            interactions_seq_auto=True,
                                            input_sz_stat=num_features_stat,
                                            output_sz=1,
                                            only_static=False,
                                            masking=True,
                                            mlp_hidden_size=stat_feature_sz,
                                            x_seq=x_train_seq,
                                            x_stat=x_train_stat,
                                            y=y_train)

                                criterion = nn.BCEWithLogitsLoss()
                                # optimizer = optim.RMSprop(model_0.parameters(), lr=learning_rate)
                                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                                # optimizer = optim.NAdam(model.parameters(), lr=learning_rate)
                                idx = np.arange(len(x_train_seq))

                                import copy
                                best_val_loss = np.inf
                                patience = 10
                                epochs = 100
                                trigger_times = 0
                                model_best_es = copy.deepcopy(model)
                                flag_es = False

                                for epoch in range(epochs):
                                    print(f"Epoch: {epoch}")
                                    np.random.shuffle(idx)
                                    x_train_seq = x_train_seq[idx]
                                    x_train_stat = x_train_stat[idx]
                                    y_train = y_train[idx]
                                    number_batches = x_train_seq.shape[0] // batch_size

                                    for i in range(number_batches):

                                        optimizer.zero_grad()  # a clean up step for PyTorch
                                        out = model(x_train_seq[i * batch_size:(i + 1) * batch_size],
                                                    x_train_stat[i * batch_size:(i + 1) * batch_size])
                                        loss = criterion(out, y_train[i * batch_size:(i + 1) * batch_size].double())
                                        loss.backward()  # compute updates for each parameter
                                        optimizer.step()  # make the updates for each parameter

                                        # train loss per batch
                                        # print('[{}/{} {}/{}] train loss: {:.8}'.format(epoch, epochs, i+1, number_batches, loss.item()))

                                    # Early stopping
                                    def validation(model, x_val_seq, x_val_stat, y_val, loss_function):

                                        # if not torch.is_tensor(x_val_stat):
                                        x_val_stat = torch.from_numpy(x_val_stat)
                                        x_val_seq = torch.from_numpy(x_val_seq)
                                        y_val = torch.from_numpy(y_val)

                                        model.eval()
                                        loss_total = 0
                                        number_batches = x_val_seq.shape[0] // batch_size

                                        with torch.no_grad():
                                            for i in range(number_batches):
                                                out = model(x_val_seq[i * batch_size: (i + 1) * batch_size],
                                                            x_val_stat[i * batch_size: (i + 1) * batch_size])
                                                loss = loss_function(out, y_val[i * batch_size:(i + 1) * batch_size].double())
                                                loss_total += loss.item()
                                        return loss_total / number_batches

                                    current_val_loss = validation(model, x_val_seq, x_val_stat, y_val, criterion)
                                    print('Validation loss:', current_val_loss)

                                    if current_val_loss > best_val_loss:
                                        trigger_times += 1
                                        print('trigger times:', trigger_times)

                                        if trigger_times >= patience:
                                            print('Early stopping!\nStart to test process.')
                                            flag_es = True
                                            break
                                    else:
                                        print('trigger times: 0')
                                        trigger_times = 0
                                        model_best_es = copy.deepcopy(model)
                                        best_val_loss = current_val_loss

                                    if flag_es:
                                        break

                                # Select model_0 based on val auc
                                model.eval()
                                with torch.no_grad():

                                    x_val_stat_ = torch.from_numpy(x_val_stat)
                                    x_val_seq_ = torch.from_numpy(x_val_seq)

                                    preds_proba = torch.sigmoid(model_best_es(x_val_seq_, x_val_stat_))
                                    preds_proba = [pred_proba[0] for pred_proba in preds_proba]

                                    auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                                    aucs.append(auc)

                                    if auc >= max(aucs):
                                        best_model = model_best_es
                                        best_hpos = {"learning_rate": learning_rate, "batch_size": batch_size,
                                                     "seq_feature_sz": seq_feature_sz,
                                                     "stat_feature_sz": stat_feature_sz,
                                                     "inter_seq_best": inter_seq_best}

            f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos.txt', 'a+')
            f.write(str(best_hpos) + '\n')
            f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
            f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
            f.write(f'Std,{np.std(aucs, ddof=1)}\n')
            f.close()

            return best_model, best_hpos
        else:
            pass

    if mode == "complete":

        if hpo:
            best_model = ""
            best_hpos = ""
            aucs = []

            for size in hpos["complete"]["size"]:
                for learning_rate in hpos["complete"]["learning_rate"]:
                    for batch_size in hpos["complete"]["batch_size"]:

                        input_layer_seq = tf.keras.layers.Input(shape=(max_case_len, num_features_seq),
                                                                name='seq_input_layer')
                        input_layer_static = tf.keras.layers.Input(shape=(num_features_stat), name='static_input_layer')

                        hidden_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                            units=size,
                            return_sequences=False))(input_layer_seq)

                        concatenate_layer = tf.keras.layers.Concatenate(axis=1)([hidden_layer, input_layer_static])

                        output_layer = tf.keras.layers.Dense(1,
                                                             activation='sigmoid',
                                                             name='output_layer')(concatenate_layer)

                        model = tf.keras.models.Model(inputs=[input_layer_seq, input_layer_static],
                                                      outputs=[output_layer])

                        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

                        model.compile(loss='binary_crossentropy',
                                      optimizer=opt,
                                      metrics=['accuracy'])
                        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
                        model_checkpoint = tf.keras.callbacks.ModelCheckpoint('../model/model_0.ckpt',
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
                                  validation_data=([x_val_seq, x_val_stat], y_val),
                                  verbose=1,
                                  callbacks=[early_stopping, model_checkpoint, lr_reducer],
                                  batch_size=batch_size,
                                  epochs=100)

                        preds_proba = model.predict([x_val_seq, x_val_stat])
                        preds_proba = [pred_proba[0] for pred_proba in preds_proba]

                        auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                        aucs.append(auc)

                        if auc >= max(aucs):
                            best_model = model
                            best_hpos = {"size": size, "learning_rate": learning_rate, "batch_size": batch_size}

            f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos.txt', 'a+')
            f.write(str(best_hpos) + '\n')
            f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
            f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
            f.write(f'Std,{np.std(aucs, ddof=1)}\n')
            f.close()

            return best_model, best_hpos

        else:
            input_layer_seq = tf.keras.layers.Input(shape=(max_case_len, num_features_seq), name='seq_input_layer')
            input_layer_static = tf.keras.layers.Input(shape=(num_features_stat), name='static_input_layer')

            hidden_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                units=hpos['size'],
                return_sequences=False))(input_layer_seq)

            concatenate_layer = tf.keras.layers.Concatenate(axis=1)([hidden_layer, input_layer_static])

            output_layer = tf.keras.layers.Dense(1,
                                                 activation='sigmoid',
                                                 name='output_layer')(concatenate_layer)

            model = tf.keras.models.Model(inputs=[input_layer_seq, input_layer_static],
                                          outputs=[output_layer])

            opt = tf.keras.optimizers.Adam(learning_rate=hpos['learning_rate'])
            model.compile(loss='binary_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint('../model/model_0.ckpt',
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
                      validation_data=([x_val_seq, x_val_stat], y_val),
                      verbose=1,
                      callbacks=[early_stopping, model_checkpoint, lr_reducer],
                      batch_size=hpos['batch_size'],
                      epochs=100)

            return model

    if mode == "static":

        if hpo:
            best_model = ""
            best_hpos = ""
            aucs = []

            for learning_rate in hpos["complete"]["learning_rate"]:
                for batch_size in hpos["complete"]["batch_size"]:

                    input_layer_static = tf.keras.layers.Input(shape=(num_features_stat), name='static_input_layer')

                    output_layer = tf.keras.layers.Dense(1,
                                                         activation='sigmoid',
                                                         name='output_layer')(input_layer_static)

                    model = tf.keras.models.Model(inputs=[input_layer_static],
                                                  outputs=[output_layer])

                    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

                    model.compile(loss='binary_crossentropy',
                                  optimizer=opt,
                                  metrics=['accuracy'])
                    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
                    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('../model/model_0.ckpt',
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
                              validation_data=([x_val_stat], y_val),
                              verbose=1,
                              callbacks=[early_stopping, model_checkpoint, lr_reducer],
                              batch_size=batch_size,
                              epochs=100)

                    preds_proba = model.predict([x_val_stat])
                    preds_proba = [pred_proba[0] for pred_proba in preds_proba]

                    auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                    aucs.append(auc)

                    if auc >= max(aucs):
                        best_model = model
                        best_hpos = {"learning_rate": learning_rate, "batch_size": batch_size}

            f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos.txt', 'a+')
            f.write(str(best_hpos) + '\n')
            f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
            f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
            f.write(f'Std,{np.std(aucs, ddof=1)}\n')
            f.close()

            return best_model, best_hpos

        else:
            input_layer_static = tf.keras.layers.Input(shape=(num_features_stat), name='static_input_layer')

            output_layer = tf.keras.layers.Dense(1,
                                                 activation='sigmoid',
                                                 name='output_layer')(input_layer_static)

            model = tf.keras.models.Model(inputs=[input_layer_static],
                                          outputs=[output_layer])

            opt = tf.keras.optimizers.Adam(learning_rate=hpos['learning_rate'])
            model.compile(loss='binary_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint('../model/model_0.ckpt',
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
                      validation_data=([x_val_seq, x_val_stat], y_val),
                      verbose=1,
                      callbacks=[early_stopping, model_checkpoint, lr_reducer],
                      batch_size=hpos['batch_size'],
                      epochs=100)

            return model

    if mode == "sequential":

        if hpo:
            best_model = ""
            best_hpos = ""
            aucs = []

            for size in hpos["complete"]["size"]:
                for learning_rate in hpos["complete"]["learning_rate"]:
                    for batch_size in hpos["complete"]["batch_size"]:

                        input_layer_seq = tf.keras.layers.Input(shape=(max_case_len, num_features_seq),
                                                                name='seq_input_layer')

                        hidden_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                            units=size,
                            return_sequences=False))(input_layer_seq)

                        output_layer = tf.keras.layers.Dense(1,
                                                             activation='sigmoid',
                                                             name='output_layer')(hidden_layer)

                        model = tf.keras.models.Model(inputs=[input_layer_seq],
                                                      outputs=[output_layer])

                        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

                        model.compile(loss='binary_crossentropy',
                                      optimizer=opt,
                                      metrics=['accuracy'])

                        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
                        model_checkpoint = tf.keras.callbacks.ModelCheckpoint('../model/model_0.ckpt',
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
                                  validation_data=([x_val_seq, x_val_stat], y_val),
                                  verbose=1,
                                  callbacks=[early_stopping, model_checkpoint, lr_reducer],
                                  batch_size=batch_size,
                                  epochs=100)

                        preds_proba = model.predict([x_val_seq, x_val_stat])
                        preds_proba = [pred_proba[0] for pred_proba in preds_proba]

                        auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                        aucs.append(auc)

                        if auc >= max(aucs):
                            best_model = model
                            best_hpos = {"size": size, "learning_rate": learning_rate, "batch_size": batch_size}

            f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos.txt', 'a+')
            f.write(str(best_hpos) + '\n')
            f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
            f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
            f.write(f'Std,{np.std(aucs, ddof=1)}\n')
            f.close()

            return best_model, best_hpos

        else:

            input_layer_seq = tf.keras.layers.Input(shape=(max_case_len, num_features_seq), name='seq_input_layer')

            hidden_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                units=hpos['size'],
                return_sequences=False))(input_layer_seq)

            output_layer = tf.keras.layers.Dense(1,
                                                 activation='sigmoid',
                                                 name='output_layer')(hidden_layer)

            model = tf.keras.models.Model(inputs=[input_layer_seq],
                                          outputs=[output_layer])

            opt = tf.keras.optimizers.Adam(learning_rate=hpos['learning_rate'])
            model.compile(loss='binary_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint('../model/model_0.ckpt',
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
                      validation_data=([x_val_seq, x_val_stat], y_val),
                      verbose=1,
                      callbacks=[early_stopping, model_checkpoint, lr_reducer],
                      batch_size=hpos['batch_size'],
                      epochs=100)

            return model


def correct_static(seq, seqs_time, idx_sample, idx_time):
    features = seqs_time[0][0].index._values

    for idx, feature in enumerate(features):
        if feature == "age":
            seq[idx] = seqs_time[idx_sample][idx_time][feature]

        elif feature == "gender" or feature == "ethnicity":
            pass
        elif feature in ['admission_type', 'marital_status', 'language', 'religion', 'insurance']:
            seq[idx] = seqs_time[idx_sample][idx_time][feature]
        else:
            seq[idx] = seqs_time[idx_sample][idx_time][feature]
    return seq


def time_step_blow_up(X_seq, X_stat, y, max_len, ts_info=False, x_time=None, x_time_vals=None,
                      x_statics_vals_corr=None):

    X_seq_prefix, X_stat_prefix, y_prefix, x_time_vals_prefix, ts = [], [], [], [], []

    for idx_seq in range(0, len(X_seq)):
        for idx_ts in range(min_size_prefix, len(X_seq[idx_seq]) + 1):
            X_seq_prefix.append(X_seq[idx_seq][0:idx_ts])

            if data_set == "mimic":
                X_stat_prefix.append(correct_static(X_stat[idx_seq], x_statics_vals_corr, idx_seq, idx_ts - 1))
            else:
                X_stat_prefix.append(X_stat[idx_seq])

            y_prefix.append(y[idx_seq])
            if x_time is not None:
                x_time_vals_prefix.append(x_time_vals[idx_seq][0:idx_ts])
            ts.append(idx_ts)

    # remove temporal overlap
    if data_set is not "mimic":
        # Remove prefixes with future event from training set
        if x_time is not None:
            X_seq_prefix_temp, X_stat_prefix_temp, y_prefix_temp, ts_temp = [], [], [], []

            for idx_prefix in range(0, len(X_seq_prefix)):
                if x_time_vals_prefix[idx_prefix][-1].value <= x_time.value:
                    X_seq_prefix_temp.append(X_seq_prefix[idx_prefix])
                    X_stat_prefix_temp.append(X_stat_prefix[idx_prefix])
                    y_prefix_temp.append(y_prefix[idx_prefix])
                    ts_temp.append(ts[idx_prefix])

            X_seq_prefix, X_stat_prefix, y_prefix, ts = X_seq_prefix_temp, X_stat_prefix_temp, y_prefix_temp, ts_temp

    # Vectorization
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


def evaluate_on_cut(x_seqs, x_statics, y, mode, target_activity, data_set, hpos, hpo, static_features, x_time=None,
                    x_statics_vals_corr=None):

    data_index = list(range(0, len(y)))

    # import random
    # random.shuffle(data_index)

    train_index = data_index[0: int(train_size * (1 - val_size) * len(y))]
    val_index = data_index[int(train_size * (1 - val_size) * len(y)): int(train_size * len(y))]
    test_index = data_index[int(train_size * len(y)):]

    if x_time is not None:
        time_start_val = x_time[val_index[0]][0]
        time_start_test = x_time[test_index[0]][0]
        x_time_train = x_time[0: int(train_size * (1 - val_size) * len(y))]
        x_time_val = x_time[int(train_size * (1 - val_size) * len(y)): int(train_size * len(y))]

    results = {}
    best_hpos_repetitions = ""

    for repetition in range(0, num_repetitions):

        X_train_seq, X_train_stat, y_train = time_step_blow_up(
            [x_seqs[x] for x in train_index],
            [x_statics[x] for x in train_index],
            [y[x] for x in train_index],
            max_len,
            ts_info=False,
            x_time=time_start_val,
            x_time_vals=x_time_train,
            x_statics_vals_corr=None)

        X_val_seq, X_val_stat, y_val = time_step_blow_up(
            [x_seqs[x] for x in val_index],
            [x_statics[x] for x in val_index],
            [y[x] for x in val_index],
            max_len,
            ts_info=False,
            x_time=time_start_test,
            x_time_vals=x_time_val,
            x_statics_vals_corr=None)

        X_test_seq, X_test_stat, y_test, ts = time_step_blow_up([x_seqs[x] for x in test_index],
                                                                [x_statics[x] for x in test_index],
                                                                [y[x] for x in test_index],
                                                                max_len,
                                                                ts_info=True,
                                                                x_statics_vals_corr=None)


        if mode == "test":
            model, best_hpos = train_lstm(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                          y_val.reshape(-1, 1), hpos, hpo, mode)
            X_test_seq = torch.from_numpy(X_test_seq)
            X_test_stat = torch.from_numpy(X_test_stat)

            model.eval()
            with torch.no_grad():
                preds_proba = torch.sigmoid(model(X_test_seq, X_test_stat))

            def map_value(value):
                if value >= 0.5:
                    return 1
                else:
                    return 0

            results['preds'] = [map_value(pred[0]) for pred in preds_proba]
            results['preds_proba'] = [pred_proba[0] for pred_proba in preds_proba]

        if mode == "complete":
            model, best_hpos = train_lstm(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                          y_val.reshape(-1, 1), hpos, hpo, mode)
            preds_proba = model.predict([X_test_seq, X_test_stat])
            results['preds'] = [int(round(pred[0])) for pred in preds_proba]
            results['preds_proba'] = [pred_proba[0] for pred_proba in preds_proba]

        elif mode == "static":
            model, best_hpos = train_lstm(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                          y_val.reshape(-1, 1), hpos, hpo, mode)
            preds_proba = model.predict([X_test_stat])
            results['preds'] = [int(round(pred[0])) for pred in preds_proba]
            results['preds_proba'] = [pred_proba[0] for pred_proba in preds_proba]

        elif mode == "sequential":
            model, best_hpos = train_lstm(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                          y_val.reshape(-1, 1), hpos, hpo, mode)
            preds_proba = model.predict([X_test_seq])
            results['preds'] = [int(round(pred[0])) for pred in preds_proba]
            results['preds_proba'] = [pred_proba[0] for pred_proba in preds_proba]

        elif mode == "rf":
            model, best_hpos = train_rf(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                        y_val.reshape(-1, 1), hpos, hpo)
            # preds_proba = model_0.predict_proba(concatenate_tensor_matrix(X_test_seq, X_test_stat))
            preds_proba = model.predict_proba(X_test_stat)
            results['preds'] = [np.argmax(pred_proba) for pred_proba in preds_proba]
            results['preds_proba'] = [pred_proba[1] for pred_proba in preds_proba]

        elif mode == "lr":
            model, best_hpos = train_lr(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                        y_val.reshape(-1, 1), hpos, hpo)
            # preds_proba = model_0.predict_proba(concatenate_tensor_matrix(X_test_seq, X_test_stat))
            preds_proba = model.predict_proba(X_test_stat)
            results['preds'] = [np.argmax(pred_proba) for pred_proba in preds_proba]
            results['preds_proba'] = [pred_proba[1] for pred_proba in preds_proba]

        elif mode == "svm":
            model, best_hpos = train_svm(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                         y_val.reshape(-1, 1), hpos, hpo)
            # preds_proba = model_0.predict_proba(concatenate_tensor_matrix(X_test_seq, X_test_stat))
            preds_proba = model.predict_proba(X_test_stat)
            results['preds'] = [np.argmax(pred_proba) for pred_proba in preds_proba]
            results['preds_proba'] = [pred_proba[1] for pred_proba in preds_proba]

        elif mode == "gb":
            model, best_hpos = train_gb(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                        y_val.reshape(-1, 1), hpos, hpo)
            # preds_proba = model_0.predict_proba(concatenate_tensor_matrix(X_test_seq, X_test_stat))
            preds_proba = model.predict_proba(X_test_stat)
            results['preds'] = [np.argmax(pred_proba) for pred_proba in preds_proba]
            results['preds_proba'] = [pred_proba[1] for pred_proba in preds_proba]

        elif mode == "ada":
            model, best_hpos = train_ada(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                         y_val.reshape(-1, 1), hpos, hpo)
            # preds_proba = model_0.predict_proba(concatenate_tensor_matrix(X_test_seq, X_test_stat))
            preds_proba = model.predict_proba(X_test_stat)
            results['preds'] = [np.argmax(pred_proba) for pred_proba in preds_proba]
            results['preds_proba'] = [pred_proba[1] for pred_proba in preds_proba]

        elif mode == "nb":
            model, best_hpos = train_nb(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                        y_val.reshape(-1, 1), hpos, hpo)
            # preds_proba = model_0.predict_proba(concatenate_tensor_matrix(X_test_seq, X_test_stat))
            preds_proba = model.predict_proba(X_test_stat)
            results['preds'] = [np.argmax(pred_proba) for pred_proba in preds_proba]
            results['preds_proba'] = [pred_proba[1] for pred_proba in preds_proba]

        elif mode == "dt":
            model, best_hpos = train_dt(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                        y_val.reshape(-1, 1), hpos, hpo)

            from matplotlib import pyplot as plt
            fig = plt.figure(figsize=(100, 100))
            _ = sklearn.tree.plot_tree(model, feature_names=static_features, filled=True)
            fig.savefig("decision_tree_train.png")
            print("Taining")
            print(static_features)
            print(model.feature_importances_)

            # preds_proba = model_0.predict_proba(concatenate_tensor_matrix(X_test_seq, X_test_stat))
            preds_proba = model.predict_proba(X_test_stat)
            results['preds'] = [np.argmax(pred_proba) for pred_proba in preds_proba]
            results['preds_proba'] = [pred_proba[1] for pred_proba in preds_proba]

        elif mode == "knn":
            model, best_hpos = train_knn(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                         y_val.reshape(-1, 1), hpos, hpo)
            # preds_proba = model_0.predict_proba(concatenate_tensor_matrix(X_test_seq, X_test_stat))
            preds_proba = model.predict_proba(X_test_stat)
            results['preds'] = [np.argmax(pred_proba) for pred_proba in preds_proba]
            results['preds_proba'] = [pred_proba[1] for pred_proba in preds_proba]

        model = DecisionTreeClassifier(max_depth=8, min_samples_split=4)
        model.fit(X_test_stat, y_test)

        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=(100, 100))
        _ = sklearn.tree.plot_tree(model, feature_names=static_features, filled=True)
        fig.savefig("decision_tree_test.png")
        print("Test")
        print(static_features)
        print(model.feature_importances_)



        results['gts'] = [int(y) for y in y_test]
        results['ts'] = ts

        results_temp = pd.DataFrame(
            list(zip(results['ts'],
                     results['preds'],
                     results['preds_proba'],
                     results['gts'])),
            columns=['ts', 'preds', 'preds_proba', 'gts'])

        cut_lengths = range(0, max_len - 1)

        # Init
        if cut_lengths[0] not in results:
            for cut_len in cut_lengths:
                results[cut_len] = {}
                results[cut_len]['acc'] = list()
                results[cut_len]['auc'] = list()
                results['all'] = {}
                results['all']['rep'] = list()
                results['all']['auc'] = list()

        # Metrics per cut
        for cut_len in cut_lengths:
            results_temp_cut = results_temp[results_temp.ts == cut_len]

            if not results_temp_cut.empty:  # if cut length is longer than max trace length
                results[cut_len]['acc'].append(
                    metrics.accuracy_score(y_true=results_temp_cut['gts'], y_pred=results_temp_cut['preds']))
                try:
                    results[cut_len]['auc'].append(
                        metrics.roc_auc_score(y_true=results_temp_cut['gts'], y_score=results_temp_cut['preds_proba']))
                except:
                    pass

        # Metrics across cuts
        results['all']['rep'].append(
            metrics.classification_report(y_true=results_temp['gts'], y_pred=results_temp['preds'], output_dict=True))

        # Save model_0
        if mode == 'test':
            torch.save(model, os.path.join("../model", f"model_{repetition}"))

        try:
            auc = metrics.roc_auc_score(y_true=results_temp['gts'], y_score=results_temp['preds_proba'])
            results['all']['auc'].append(auc)

            if auc >= max(results['all']['auc']):
                best_hpos_repetitions = best_hpos

        except:
            pass

        # Save all results
        f = open(f'../output/{data_set}_{mode}_{target_activity}_summary.txt', 'w')
        results_ = results
        del results_['preds'], results_['preds_proba'], results_['gts'], results_['ts']
        f.write(str(results_))
        f.close()

        # print metrics
        metrics_ = ["auc", "precision", "recall", "f1-score", "support", "accuracy"]
        labels = ["0", "1"]


        def calculate_corrected_metric(metric, pos_class, support):
            fpr, tpr, thresholds = metrics.roc_curve(y_true=results_temp['gts'], y_score=results_temp['preds_proba'])
            J = tpr - fpr
            ix = np.argmax(J)

            def to_labels(pos_probs, threshold):
                return (pos_probs >= threshold).astype('int')

            best_thresh = thresholds[ix]
            # print("Best Threshold=%.3f" % (thresholds[ix]))

            if pos_class == "0":
                if metric == "precision":
                    return [metrics.precision_score(results_temp['gts'], to_labels(results_temp['preds_proba'], t), average="binary", pos_label=0) for t in thresholds][ix]
                elif metric == "recall":
                    return [metrics.recall_score(results_temp['gts'], to_labels(results_temp['preds_proba'], t), average="binary", pos_label=0) for t in thresholds][ix]
                elif metric == "f1-score":
                    return [metrics.f1_score(results_temp['gts'], to_labels(results_temp['preds_proba'], t), average="binary", pos_label=0) for t in thresholds][ix]
                elif metric == "support":
                    return support
            else:
                if metric == "precision":
                    return [metrics.precision_score(results_temp['gts'], to_labels(results_temp['preds_proba'], t), average="binary", pos_label=1) for t in thresholds][ix]
                elif metric == "recall":
                    return [metrics.recall_score(results_temp['gts'], to_labels(results_temp['preds_proba'], t), average="binary", pos_label=1) for t in thresholds][ix]
                elif metric == "f1-score":
                    return [metrics.f1_score(results_temp['gts'], to_labels(results_temp['preds_proba'], t), average="binary", pos_label=1) for t in thresholds][ix]
                elif metric == "support":
                    return support

        for metric_ in metrics_:
            vals = []
            if metric_ == "auc":
                try:
                    f = open(f'../output/{data_set}_{mode}_{target_activity}.txt', "a+")
                    f.write(metric_ + '\n')
                    print(metric_)
                    for idx_ in range(0, repetition + 1):
                        vals.append(results['all']['auc'][idx_])
                        f.write(f'{idx_},{vals[-1]}\n')
                        print(f'{idx_},{vals[-1]}')
                    f.write(f'Avg,{sum(vals) / len(vals)}\n')
                    f.write(f'Std,{np.std(vals, ddof=1)}\n')
                    print(f'Avg,{sum(vals) / len(vals)}')
                    print(f'Std,{np.std(vals, ddof=1)}\n')
                    f.close()

                except:
                    pass

            elif metric_ == "accuracy":
                f = open(f'../output/{data_set}_{mode}_{target_activity}.txt', "a+")
                f.write(metric_ + '\n')
                print(metric_)
                for idx_ in range(0, repetition + 1):
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
                    try:
                        f = open(f'../output/{data_set}_{mode}_{target_activity}.txt', "a+")
                        f.write(metric_ + f' ({label})\n')
                        print(metric_ + f' ({label})')
                        vals = []
                        for idx_ in range(0, repetition + 1):
                            # vals.append(results['all']['rep'][idx_][label][metric_])
                            vals.append(calculate_corrected_metric(metric_, label, results['all']['rep'][idx_][label][metric_]))
                            f.write(f'{idx_},{vals[-1]}\n')
                            print(f'{idx_},{vals[-1]}')
                        f.write(f'Avg,{sum(vals) / len(vals)}\n')
                        f.write(f'Std,{np.std(vals, ddof=1)}\n')
                        print(f'Avg,{sum(vals) / len(vals)}')
                        print(f'Std,{np.std(vals, ddof=1)}')
                        f.close()
                    except:
                        pass



    return X_train_seq, X_train_stat, y_train, X_val_seq, X_val_stat, y_val, best_hpos_repetitions


def run_coefficient(x_seqs_train, x_statics_train, y_train, x_seqs_val, x_statics_val, y_val, target_activity,
                    static_features, best_hpos_repetitions):
    model = train_lstm(x_seqs_train, x_statics_train, y_train, x_seqs_val, x_statics_val, y_val, best_hpos_repetitions,
                       False, mode="complete")
    output_weights = model.get_layer(name='output_layer').get_weights()[0].flatten()[2 * best_hpos_repetitions['size']:]
    output_names = static_features

    f = open(f'../output/{data_set}_{mode}_{target_activity}_coef.txt', 'w')
    f.write(",".join([str(x) for x in output_names]) + '\n')
    f.write(",".join([str(x) for x in output_weights]))
    f.close()

    return model


if __name__ == "__main__":

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #    tf.config.experimental.set_memory_growth(gpu, True)

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    hpos = {
        # "test": {"seq_feature_sz": [4,8,16], "stat_feature_sz": [4,8,16], "learning_rate": [0.001, 0.01], "batch_size": [32], "inter_seq_best": [4]},
        "test": {"seq_feature_sz": [4], "stat_feature_sz": [4], "learning_rate": [0.001], "batch_size": [128], "inter_seq_best": [1]},
        "complete": {"size": [4, 8, 32, 64], "learning_rate": [0.001, 0.01, 0.05], "batch_size": [32, 128]},
        "sequential": {"size": [4, 8, 32, 64], "learning_rate": [0.001, 0.01, 0.05], "batch_size": [32, 128]},
        "static": {"learning_rate": [0.001, 0.01, 0.05], "batch_size": [32, 128]},
        "lr": {"reg_strength": [pow(10, -3), pow(10, -2), pow(10, -1), pow(10, 0), pow(10, 1), pow(10, 2), pow(10, 3)],
               "solver": ["lbfgs"]},
        "rf": {"num_trees": [100, 200, 500], "max_depth_trees": [2, 5, 10], "num_rand_vars": [1, 3, 5, 10]},
        # "svm": {"kern_fkt": ["linear", "rbf"], "cost": [pow(10, -3), pow(10, -2), pow(10, -1), pow(10, 0), pow(10, 1), pow(10, 2), pow(10, 3)]},
        "gb": {"n_estimators": [100, 200, 500], "learning_rate": [0.01, 0.05, 0.1]},
        "ada": {"n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.5, 1.0]},
        "nb": {"var_smoothing": np.logspace(0, -9, num=10)},
        "dt": {"max_depth": [2, 4, 8, 16], "min_samples_split": [2, 4, 8, 16]},
        "knn": {"n_neighbors": [3, 5, 10]}
    }

    if data_set == "sepsis":

        for mode in ['dt']:  # 'complete', 'static', 'sequential', 'lr', 'rf', 'gb', 'ada', 'dt', 'knn', 'nb'
            for target_activity in ['Admission IC']:

                x_seqs, x_statics, y, x_time_vals_final, seq_features, static_features = data.get_sepsis_data(
                    target_activity, max_len, min_len)

                # Run eval on cuts to plot results --> Figure 1
                x_seqs_train, x_statics_train, y_train, x_seqs_val, x_statics_val, y_val, best_hpos_repetitions = evaluate_on_cut(
                    x_seqs, x_statics, y, mode, target_activity,
                    data_set, hpos, hpo, static_features, x_time=x_time_vals_final, x_statics_vals_corr=None)


                if mode == "complete":
                    # Train model_0 and plot linear coeff --> Figure 2
                    model = run_coefficient(x_seqs_train, x_statics_train, y_train, x_seqs_val, x_statics_val, y_val,
                                            target_activity, static_features, best_hpos_repetitions)

                    x_seqs_train = x_seqs_train[0:1000]
                    x_statics_train = x_statics_train[0:1000]

                    # Get Explanations for LSTM inputs --> Figure 3
                    explainer = shap.DeepExplainer(model, [x_seqs_train, x_statics_train])
                    shap_values = explainer.shap_values([x_seqs_train, x_statics_train])

                    seqs_df = pd.DataFrame(data=x_seqs_train.reshape(-1, len(seq_features)),
                                           columns=seq_features)
                    seq_shaps = pd.DataFrame(data=shap_values[0][0].reshape(-1, len(seq_features)),
                                             columns=[f'SHAP {x}' for x in seq_features])
                    seq_value_shape = pd.concat([seqs_df, seq_shaps], axis=1)

                    with open(f'../output/{data_set}_{mode}_{target_activity}_shap.npy', 'wb') as f:
                        pickle.dump(seq_value_shape, f)

    elif data_set == "mimic":

        for mode in ['complete']:  # 'complete', 'static', 'sequential', 'lr', 'rf', 'gb', 'ada', 'dt', 'knn', 'nb'
            for target_activity in ['LEFT AGAINST MEDICAL ADVI']:

                x_seqs, x_statics, y, x_time_vals_final, seq_features, static_features, static_vals_corr = data.get_mimic_data(
                    target_activity, max_len, min_len)

                # Run eval on cuts to plot results
                x_seqs_train, x_statics_train, y_train, x_seqs_val, x_statics_val, y_val, best_hpos_repetitions = evaluate_on_cut(
                    x_seqs, x_statics, y, mode, target_activity,
                    data_set, hpos, hpo, x_time=x_time_vals_final, x_statics_vals_corr=static_vals_corr)

                if mode == "complete":
                    # Train model_0 and plot linear coeff
                    model = run_coefficient(x_seqs_train, x_statics_train, y_train, x_seqs_val, x_statics_val, y_val,
                                            target_activity, static_features, best_hpos_repetitions)

                    x_seqs_train = x_seqs_train[0:1000]
                    x_statics_train = x_statics_train[0:1000]

                    # Get Explanations for LSTM inputs
                    explainer = shap.DeepExplainer(model, [x_seqs_train, x_statics_train])
                    shap_values = explainer.shap_values([x_seqs_train, x_statics_train])

                    seqs_df = pd.DataFrame(data=x_seqs_train.reshape(-1, len(seq_features)),
                                           columns=seq_features)
                    seq_shaps = pd.DataFrame(data=shap_values[0][0].reshape(-1, len(seq_features)),
                                             columns=[f'SHAP {x}' for x in seq_features])
                    seq_value_shape = pd.concat([seqs_df, seq_shaps], axis=1)

                    with open(f'../output/{data_set}_{mode}_{target_activity}_shap.npy', 'wb') as f:
                        pickle.dump(seq_value_shape, f)

    else:
        print("Data set not available!")
