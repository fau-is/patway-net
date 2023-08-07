import sys
sys.path.insert(0,"../")

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os
import src.data as data
import torch
from src.interpret_LSTM import Net
from sklearn.model_selection import StratifiedKFold
import pickle
import xgboost as xgb

max_len = 50
min_len = 3
min_size_prefix = 1
val_size = 0.2
train_size = 0.8
hpo = True
save_baseline_model = True


def concatenate_tensor_matrix(x_seq, x_stat):
    x_train_seq_ = x_seq.reshape(-1, x_seq.shape[1] * x_seq.shape[2])
    x_concat = np.concatenate((x_train_seq_, x_stat), axis=1)

    return x_concat


def train_rf(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo, data_set, target_activity=None):
    if hpo:
        best_model = ""
        best_hpos = ""
        aucs = []

        for max_depth in hpos["rf"]["max_depth"]:
            for n_estimators in hpos["rf"]["n_estimators"]:
                for max_leaf_nodes in hpos["rf"]["max_leaf_nodes"]:

                    model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators,
                                                   max_leaf_nodes=max_leaf_nodes)
                    model.fit(x_train_stat, np.ravel(y_train))

                    preds_proba = model.predict_proba(x_val_stat)
                    preds_proba = [pred_proba[1] for pred_proba in preds_proba]
                    try:
                        auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                        if np.isnan(auc):
                            auc = 0
                    except:
                        auc = 0
                    aucs.append(auc)

                    if auc >= max(aucs):
                        best_model = model
                        best_hpos = {"max_depth": max_depth, "n_estimators": n_estimators,
                                     "max_leaf_nodes": max_leaf_nodes}

        f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos_{seed}.txt', 'a+')
        f.write(str(best_hpos))
        f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
        f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
        f.write(f'Std,{np.std(aucs, ddof=1)}\n')
        f.close()

        return best_model, best_hpos

    else:
        model = RandomForestClassifier()
        model.fit(x_train_stat, np.ravel(y_train))

        return model

def train_xgb(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo, data_set, target_activity=None):
    if hpo:
        best_model = ""
        best_hpos = ""
        aucs = []

        for max_depth in hpos["xgb"]["max_depth"]:
            for learning_rate in hpos["xgb"]["learning_rate"]:

                model = xgb.XGBClassifier(max_depth=max_depth, learning_rate=learning_rate)
                model.fit(x_train_stat, np.ravel(y_train))

                preds_proba = model.predict_proba(x_val_stat)
                preds_proba = [pred_proba[1] for pred_proba in preds_proba]
                try:
                    auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                    if np.isnan(auc):
                        auc = 0
                except:
                    auc = 0
                aucs.append(auc)

                if auc >= max(aucs):
                    best_model = model
                    best_hpos = {"max_depth": max_depth, "learning_rate": learning_rate}

        f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos_{seed}.txt', 'a+')
        f.write(str(best_hpos))
        f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
        f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
        f.write(f'Std,{np.std(aucs, ddof=1)}\n')
        f.close()

        return best_model, best_hpos

    else:
        model = xgb.XGBClassifier()
        model.fit(x_train_stat, np.ravel(y_train))

        return model


def train_lr(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo, data_set, target_activity=None):
    if hpo:
        best_model = ""
        best_hpos = ""
        aucs = []

        for c in hpos["lr"]["reg_strength"]:
            for solver in hpos["lr"]["solver"]:

                model = LogisticRegression(C=c, solver=solver)
                model.fit(x_train_stat, np.ravel(y_train))
                preds_proba = model.predict_proba(x_val_stat)
                preds_proba = [pred_proba[1] for pred_proba in preds_proba]
                try:
                    auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                    if np.isnan(auc):
                        auc = 0
                except:
                    auc = 0
                aucs.append(auc)

                if auc >= max(aucs):
                    best_model = model
                    best_hpos = {"c": c, "solver": solver}

        f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos_{seed}.txt', 'a+')
        f.write(str(best_hpos))
        f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
        f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
        f.write(f'Std,{np.std(aucs, ddof=1)}\n')
        f.close()

        return best_model, best_hpos

    else:
        model = LogisticRegression()
        model.fit(x_train_stat, np.ravel(y_train))

        return model


def train_nb(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo, data_set, target_activity=None):
    if hpo:
        best_model = ""
        best_hpos = ""
        aucs = []

        for var_smoothing in hpos["nb"]["var_smoothing"]:

            model = GaussianNB(var_smoothing=var_smoothing)
            model.fit(x_train_stat, np.ravel(y_train))
            preds_proba = model.predict_proba(x_val_stat)
            preds_proba = [pred_proba[1] for pred_proba in preds_proba]
            try:
                auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                if np.isnan(auc):
                    auc = 0
            except:
                auc = 0
            aucs.append(auc)

            if auc >= max(aucs):
                best_model = model
                best_hpos = {"var_smoothing": var_smoothing}

        f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos_{seed}.txt', 'a+')
        f.write(str(best_hpos) + '\n')
        f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
        f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
        f.write(f'Std,{np.std(aucs, ddof=1)}\n')
        f.close()

        return best_model, best_hpos

    else:
        model = GaussianNB()
        model.fit(x_train_stat, np.ravel(y_train))

        return model


def train_dt(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo, data_set, target_activity=None):
    if hpo:
        best_model = ""
        best_hpos = ""
        aucs = []

        for max_depth in hpos["dt"]["max_depth"]:
            for min_samples_split in hpos["dt"]["min_samples_split"]:

                model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
                model.fit(x_train_stat, np.ravel(y_train))
                preds_proba = model.predict_proba(x_val_stat)
                preds_proba = [pred_proba[1] for pred_proba in preds_proba]
                try:
                    auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                    if np.isnan(auc):
                        auc = 0
                except:
                    auc = 0

                aucs.append(auc)

                if auc >= max(aucs):
                    best_model = model
                    best_hpos = {"max_depth": max_depth, "min_samples_split": min_samples_split}

        f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos_{seed}.txt', 'a+')
        f.write(str(best_hpos) + '\n')
        f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
        f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
        f.write(f'Std,{np.std(aucs, ddof=1)}\n')
        f.close()

        return best_model, best_hpos

    else:
        model = DecisionTreeClassifier()
        model.fit(x_train_stat, np.ravel(y_train))

        return model


def train_knn(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo, data_set, target_activity=None):
    if hpo:
        best_model = ""
        best_hpos = ""
        aucs = []

        for n_neighbors in hpos["knn"]["n_neighbors"]:

            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            model.fit(x_train_stat, np.ravel(y_train))
            preds_proba = model.predict_proba(x_val_stat)
            preds_proba = [pred_proba[1] for pred_proba in preds_proba]
            try:
                auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                if np.isnan(auc):
                    auc = 0
            except:
                auc = 0
            aucs.append(auc)

            if auc >= max(aucs):
                best_model = model
                best_hpos = {"n_eighbors": n_neighbors}

        f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos_{seed}.txt', 'a+')
        f.write(str(best_hpos) + '\n')
        f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
        f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
        f.write(f'Std,{np.std(aucs, ddof=1)}\n')
        f.close()

        return best_model, best_hpos

    else:
        model = KNeighborsClassifier()
        model.fit(x_train_stat, np.ravel(y_train))

        return model


def train_lstm(x_train_seq, x_train_stat, y_train, id, x_val_seq=False, x_val_stat=False, y_val=False, hpos=False,
               hpo=False, mode="pwn", data_set="sepsis", target_activity=None):
    max_case_len = x_train_seq.shape[1]
    num_features_seq = x_train_seq.shape[2]
    num_features_stat = x_train_stat.shape[1]

    interactions_seq_itr = 100
    patience = 10
    epochs = 100
    lstm_mode = ["pwn", "pwn_no_inter", "lstm", "pwn_all_feat_seq"][0]

    if lstm_mode == "pwn":
        masking = True
        interactions_seq_auto = True
        only_static = False
        all_feat_seq = False

    elif lstm_mode == "pwn_no_inter":
        masking = True
        interactions_seq_auto = False
        only_static = False
        all_feat_seq = False

    elif lstm_mode == "lstm":
        masking = False
        interactions_seq_auto = False
        only_static = False
        all_feat_seq = False

    elif lstm_mode == "pwn_all_feat_seq":
        masking = False
        interactions_seq_auto = False
        only_static = False
        all_feat_seq = True

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

        for learning_rate in hpos["pwn"]["learning_rate"]:
            for batch_size in hpos["pwn"]["batch_size"]:
                for seq_feature_sz in hpos["pwn"]["seq_feature_sz"]:
                    for stat_feature_sz in hpos["pwn"]["stat_feature_sz"]:
                        for inter_seq_best in hpos["pwn"]["inter_seq_best"]:

                            model = Net(input_sz_seq=num_features_seq,
                                        hidden_per_seq_feat_sz=seq_feature_sz,
                                        interactions_seq=[],
                                        interactions_seq_itr=interactions_seq_itr,
                                        interactions_seq_best=inter_seq_best,
                                        interactions_seq_auto=interactions_seq_auto,
                                        input_sz_stat=num_features_stat,
                                        output_sz=1,
                                        only_static=only_static,
                                        all_feat_seq=all_feat_seq,
                                        masking=masking,
                                        mlp_hidden_size=stat_feature_sz,
                                        x_seq=x_train_seq,
                                        x_stat=x_train_stat,
                                        y=y_train)

                            criterion = nn.BCEWithLogitsLoss()
                            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                            # optimizer = optim.NAdam(model.parameters(), lr=learning_rate)
                            idx = np.arange(len(x_train_seq))

                            import copy
                            best_val_loss = np.inf
                            trigger_times = 0
                            model_best_es = copy.deepcopy(model)
                            flag_es = False

                            for epoch in range(epochs):
                                print(f"Epoch: {epoch + 1}")
                                np.random.shuffle(idx)
                                x_train_seq = x_train_seq[idx]
                                x_train_stat = x_train_stat[idx]
                                y_train = y_train[idx]
                                number_batches = x_train_seq.shape[0] // batch_size

                                for i in range(number_batches):
                                    optimizer.zero_grad()  # clean up step for PyTorch
                                    out = model(x_train_seq[i * batch_size:(i + 1) * batch_size],
                                                x_train_stat[i * batch_size:(i + 1) * batch_size])
                                    loss = criterion(out, y_train[i * batch_size:(i + 1) * batch_size].double())
                                    loss.backward()  # compute updates for each parameter
                                    optimizer.step()  # make the updates for each parameter

                                # Early stopping
                                def validation(model, x_val_seq, x_val_stat, y_val, loss_function):

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
                                            loss = loss_function(out, y_val[
                                                                      i * batch_size:(i + 1) * batch_size].double())
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

                            # Select model based on val auc
                            model_best_es.eval()
                            with torch.no_grad():

                                x_val_stat_ = torch.from_numpy(x_val_stat)
                                x_val_seq_ = torch.from_numpy(x_val_seq)

                                preds_proba = torch.sigmoid(model_best_es(x_val_seq_, x_val_stat_))
                                preds_proba = [pred_proba[0] for pred_proba in preds_proba]
                                try:
                                    auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                                    if np.isnan(auc):
                                        auc = 0
                                except:
                                    auc = 0
                                aucs.append(auc)

                                if auc >= max(aucs):
                                    best_model = copy.deepcopy(model_best_es)
                                    best_hpos = {"learning_rate": learning_rate, "batch_size": batch_size,
                                                 "seq_feature_sz": seq_feature_sz,
                                                 "stat_feature_sz": stat_feature_sz,
                                                 "inter_seq_best": inter_seq_best}

            f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos_{seed}.txt', 'a+')
            f.write(str(best_hpos) + '\n')
            f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
            f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
            f.write(f'Std,{np.std(aucs, ddof=1)}\n')
            f.close()

            torch.save(model, os.path.join("../model", f"model_{lstm_mode}_{id}_{seed}"))

            return best_model, best_hpos
        else:
            pass


def time_step_blow_up(X_seq, X_stat, y, max_len):
    X_seq_prefix, X_stat_prefix, y_prefix, x_time_vals_prefix, ts = [], [], [], [], []

    for idx_seq in range(0, len(X_seq)):
        for idx_ts in range(min_size_prefix, len(X_seq[idx_seq]) + 1):
            X_seq_prefix.append(X_seq[idx_seq][0:idx_ts])
            X_stat_prefix.append(X_stat[idx_seq])
            y_prefix.append(y[idx_seq])

    X_seq_final = np.zeros((len(X_seq_prefix), max_len, len(X_seq_prefix[0][0])), dtype=np.float32)
    X_stat_final = np.zeros((len(X_seq_prefix), len(X_stat_prefix[0])))

    for i, x in enumerate(X_seq_prefix):
        X_seq_final[i, :len(x), :] = np.array(x)
        X_stat_final[i, :] = np.array(X_stat_prefix[i])
    y_final = np.array(y_prefix).astype(np.int32)

    return X_seq_final, X_stat_final, y_final


def evaluate(x_seqs, x_statics, y, mode, target_activity, data_set, hpos, hpo, static_features, seed):
    k = 5
    results = {}
    id = -1

    skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for train_index_, test_index in skfold.split(X=x_statics, y=y):

        id += 1

        train_index = train_index_[0: int(len(train_index_) * (1 - val_size))]
        val_index = train_index_[int(len(train_index_) * (1 - val_size)):]

        X_train_seq, X_train_stat, y_train = time_step_blow_up(
            [x_seqs[x] for x in train_index],
            [x_statics[x] for x in train_index],
            [y[x] for x in train_index], max_len)

        X_val_seq, X_val_stat, y_val = time_step_blow_up(
            [x_seqs[x] for x in val_index],
            [x_statics[x] for x in val_index],
            [y[x] for x in val_index], max_len)

        X_test_seq, X_test_stat, y_test = time_step_blow_up(
            [x_seqs[x] for x in test_index],
            [x_statics[x] for x in test_index],
            [y[x] for x in test_index], max_len)

        if save_baseline_model:
            with open(f"../data_prediction_plot/test_data_{seed}", "ab") as output:
                data_dictionary = {"fold": id, "x_test_seq": X_test_seq,
                                   "x_test_stat": X_test_stat, "label": y_test, "seed": seed}
                pickle.dump(data_dictionary, output)
                print("Test data from fold " + str(id) + " saved to " + str(output))

        if mode == "pwn":
            model, best_hpos = train_lstm(X_train_seq, X_train_stat, y_train.reshape(-1, 1), id, X_val_seq, X_val_stat,
                                          y_val.reshape(-1, 1), hpos, hpo, mode, data_set, target_activity=target_activity)

            X_train_seq = torch.from_numpy(X_train_seq)
            X_train_stat = torch.from_numpy(X_train_stat)

            X_val_seq = torch.from_numpy(X_val_seq)
            X_val_stat = torch.from_numpy(X_val_stat)

            X_test_seq = torch.from_numpy(X_test_seq)
            X_test_stat = torch.from_numpy(X_test_stat)

            model.eval()
            with torch.no_grad():
                preds_proba_train = torch.sigmoid(model(X_train_seq, X_train_stat))
                preds_proba_val = torch.sigmoid(model(X_val_seq, X_val_stat))
                preds_proba_test = torch.sigmoid(model(X_test_seq, X_test_stat))

            def map_value(value):
                if value >= 0.5:
                    return 1
                else:
                    return 0

            results['preds_train'] = [map_value(pred[0]) for pred in preds_proba_train]
            results['preds_proba_train'] = [pred_proba[0] for pred_proba in preds_proba_train]
            results['preds_val'] = [map_value(pred[0]) for pred in preds_proba_val]
            results['preds_proba_val'] = [pred_proba[0] for pred_proba in preds_proba_val]
            results['preds_test'] = [map_value(pred[0]) for pred in preds_proba_test]
            results['preds_proba_test'] = [pred_proba[0] for pred_proba in preds_proba_test]

        elif mode == "rf":
            model, best_hpos = train_rf(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                        y_val.reshape(-1, 1), hpos, hpo, data_set, target_activity=target_activity)

        elif mode == "xgb":
            model, best_hpos = train_xgb(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                        y_val.reshape(-1, 1), hpos, hpo, data_set, target_activity=target_activity)

        elif mode == "lr":
            model, best_hpos = train_lr(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                         y_val.reshape(-1, 1), hpos, hpo, data_set, target_activity=target_activity)

        elif mode == "nb":
            model, best_hpos = train_nb(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                        y_val.reshape(-1, 1), hpos, hpo, data_set, target_activity=target_activity)

        elif mode == "dt":
            model, best_hpos = train_dt(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                        y_val.reshape(-1, 1), hpos, hpo, data_set, target_activity=target_activity)

            """
            from matplotlib import pyplot as plt
            import sklearn.tree
            fig = plt.figure(figsize=(100, 100))
            _ = sklearn.tree.plot_tree(model, feature_names=static_features, filled=True)
            fig.savefig("../plots/decision_tree_train")
            print("Training")
            print(static_features)
            print(model.feature_importances_)
            """

        elif mode == "knn":
            model, best_hpos = train_knn(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                         y_val.reshape(-1, 1), hpos, hpo, data_set, target_activity=target_activity)

        if mode in ["rf", "xgb", "dt", "knn", "nb", "lr"]:
            preds_proba_train = model.predict_proba(X_train_stat)
            results['preds_train'] = [np.argmax(pred_proba) for pred_proba in preds_proba_train]
            results['preds_proba_train'] = [pred_proba[1] for pred_proba in preds_proba_train]
            preds_proba_val = model.predict_proba(X_val_stat)
            results['preds_val'] = [np.argmax(pred_proba) for pred_proba in preds_proba_val]
            results['preds_proba_val'] = [pred_proba[1] for pred_proba in preds_proba_val]
            preds_proba_test = model.predict_proba(X_test_stat)
            results['preds_test'] = [np.argmax(pred_proba) for pred_proba in preds_proba_test]
            results['preds_proba_test'] = [pred_proba[1] for pred_proba in preds_proba_test]

            if save_baseline_model:
                torch.save(model, os.path.join("../model", f"model_{mode}_{id}_{seed}"))

        results['gts_train'] = [int(y) for y in y_train]
        results['gts_val'] = [int(y) for y in y_val]
        results['gts_test'] = [int(y) for y in y_test]

        results_temp_train = pd.DataFrame(
            list(zip(results['preds_train'], results['preds_proba_train'], results['gts_train'])),
            columns=['preds', 'preds_proba', 'gts'])
        results_temp_val = pd.DataFrame(list(zip(results['preds_val'], results['preds_proba_val'], results['gts_val'])),
                                        columns=['preds', 'preds_proba', 'gts'])
        results_temp_test = pd.DataFrame(
            list(zip(results['preds_test'], results['preds_proba_test'], results['gts_test'])),
            columns=['preds', 'preds_proba', 'gts'])

        if id == 0:
            results['pr_auc_train'] = list()
            results['pr_auc_val'] = list()
            results['pr_auc_test'] = list()
            results['roc_auc_train'] = list()
            results['roc_auc_val'] = list()
            results['roc_auc_test'] = list()
            results['mcc_train'] = list()
            results['mcc_val'] = list()
            results['mcc_test'] = list()
            results['f1_pos_train_corr'] = list()
            results['f1_pos_val_corr'] = list()
            results['f1_pos_test_corr'] = list()
            results['f1_neg_train_corr'] = list()
            results['f1_neg_val_corr'] = list()
            results['f1_neg_test_corr'] = list()
            results['f1_pos_train'] = list()
            results['f1_pos_val'] = list()
            results['f1_pos_test'] = list()
            results['f1_neg_train'] = list()
            results['f1_neg_val'] = list()
            results['f1_neg_test'] = list()
            results['support_train'] = list()
            results['support_val'] = list()
            results['support_test'] = list()

        def calc_roc_auc(gts, probs):
            try:
                auc = metrics.roc_auc_score(gts, probs)
                if np.isnan(auc):
                    auc = 0
                return auc
            except:
                return 0

        def calc_pr_auc(gts, probs):
            try:
                precision, recall, thresholds = metrics.precision_recall_curve(gts, probs)
                auc = metrics.auc(recall, precision)
                if np.isnan(auc):
                    auc = 0
                return auc
            except:
                return 0

        def calc_mcc(gts, preds):
            try:
                return metrics.matthews_corrcoef(gts, preds)
            except:
                return 0

        def calc_f1(gts, preds, proba, label_id, corr=False):
            try:
                if corr:
                    fpr, tpr, thresholds = metrics.roc_curve(y_true=gts, y_score=proba)
                    J = tpr - fpr
                    ix = np.argmax(J)

                    def to_labels(pos_probs, threshold):
                        return (pos_probs >= threshold).astype('int')

                    return \
                        [metrics.precision_score(gts, to_labels(proba, t), average="binary", pos_label=label_id) for t
                         in
                         thresholds][ix]
                else:
                    return metrics.f1_score(gts, preds, average="binary", pos_label=label_id)
            except:
                return 0

        results['pr_auc_train'].append(
            calc_pr_auc(gts=results_temp_train['gts'], probs=results_temp_train['preds_proba']))
        results['pr_auc_val'].append(calc_pr_auc(gts=results_temp_val['gts'], probs=results_temp_val['preds_proba']))
        results['pr_auc_test'].append(calc_pr_auc(gts=results_temp_test['gts'], probs=results_temp_test['preds_proba']))

        results['roc_auc_train'].append(
            calc_roc_auc(gts=results_temp_train['gts'], probs=results_temp_train['preds_proba']))
        results['roc_auc_val'].append(calc_roc_auc(gts=results_temp_val['gts'], probs=results_temp_val['preds_proba']))
        results['roc_auc_test'].append(
            calc_roc_auc(gts=results_temp_test['gts'], probs=results_temp_test['preds_proba']))

        results['mcc_train'].append(calc_mcc(gts=results_temp_train['gts'], preds=results_temp_train['preds']))
        results['mcc_val'].append(calc_roc_auc(gts=results_temp_val['gts'], probs=results_temp_val['preds']))
        results['mcc_test'].append(calc_roc_auc(gts=results_temp_test['gts'], probs=results_temp_test['preds']))

        results['f1_pos_train_corr'].append(calc_f1(gts=results_temp_train['gts'], preds=results_temp_train['preds'],
                                                    proba=results_temp_train['preds_proba'], label_id=1, corr=True))
        results['f1_pos_val_corr'].append(
            calc_f1(gts=results_temp_val['gts'], preds=results_temp_val['preds'], proba=results_temp_val['preds_proba'],
                    label_id=1, corr=True))
        results['f1_pos_test_corr'].append(calc_f1(gts=results_temp_test['gts'], preds=results_temp_test['preds'],
                                                   proba=results_temp_test['preds_proba'], label_id=1, corr=True))

        results['f1_neg_train_corr'].append(calc_f1(gts=results_temp_train['gts'], preds=results_temp_train['preds'],
                                                    proba=results_temp_train['preds_proba'], label_id=0, corr=True))
        results['f1_neg_val_corr'].append(
            calc_f1(gts=results_temp_val['gts'], preds=results_temp_val['preds'], proba=results_temp_val['preds_proba'],
                    label_id=0, corr=True))
        results['f1_neg_test_corr'].append(calc_f1(gts=results_temp_test['gts'], preds=results_temp_test['preds'],
                                                   proba=results_temp_test['preds_proba'], label_id=0, corr=True))

        results['f1_pos_train'].append(calc_f1(gts=results_temp_train['gts'], preds=results_temp_train['preds'],
                                               proba=results_temp_train['preds_proba'], label_id=1))
        results['f1_pos_val'].append(
            calc_f1(gts=results_temp_val['gts'], preds=results_temp_val['preds'], proba=results_temp_val['preds_proba'],
                    label_id=1))
        results['f1_pos_test'].append(calc_f1(gts=results_temp_test['gts'], preds=results_temp_test['preds'],
                                              proba=results_temp_test['preds_proba'], label_id=1))

        results['f1_neg_train'].append(calc_f1(gts=results_temp_train['gts'], preds=results_temp_train['preds'],
                                               proba=results_temp_train['preds_proba'], label_id=0))
        results['f1_neg_val'].append(
            calc_f1(gts=results_temp_val['gts'], preds=results_temp_val['preds'], proba=results_temp_val['preds_proba'],
                    label_id=0))
        results['f1_neg_test'].append(calc_f1(gts=results_temp_test['gts'], preds=results_temp_test['preds'],
                                              proba=results_temp_test['preds_proba'], label_id=0))

        results['support_train'].append(f'{y_train.tolist().count(1)}--{y_train.tolist().count(0)}')
        results['support_val'].append(f'{y_val.tolist().count(1)}--{y_val.tolist().count(0)}')
        results['support_test'].append(f'{y_test.tolist().count(1)}--{y_test.tolist().count(0)}')

        metrics_ = ["pr_auc_train", "pr_auc_val", "pr_auc_test",
                    "roc_auc_train", "roc_auc_val", "roc_auc_test",
                    "mcc_train", "mcc_val", "mcc_test",
                    "f1_pos_train_corr", "f1_pos_val_corr", "f1_pos_test_corr",
                    "f1_neg_train_corr", "f1_neg_val_corr", "f1_neg_test_corr",
                    "f1_pos_train", "f1_pos_val", "f1_pos_test",
                    "f1_neg_train", "f1_neg_val", "f1_neg_test",
                    "support_train", "support_val", "support_test"]

        for metric_ in metrics_:
            vals = []
            try:
                f = open(f'../output/{data_set}_{mode}_{target_activity}_{seed}.txt', "a+")
                f.write(metric_ + '\n')
                print(metric_)
                for idx_ in range(0, id + 1):
                    vals.append(results[metric_][idx_])
                    f.write(f'{idx_},{vals[-1]}\n')
                    print(f'{idx_},{vals[-1]}')
                if "support" not in metric_:
                    f.write(f'Avg,{sum(vals) / len(vals)}\n')
                    f.write(f'Std,{np.std(vals, ddof=1)}\n')
                    print(f'Avg,{sum(vals) / len(vals)}')
                    print(f'Std,{np.std(vals, ddof=1)}\n')
                f.close()
            except:
                pass

    return X_train_seq, X_train_stat, y_train, X_val_seq, X_val_stat, y_val


if __name__ == "__main__":

    data_set = "sepsis"  # bpi2012, hospital

    hpos = {
        "pwn": {"seq_feature_sz": [8], "stat_feature_sz": [8], "learning_rate": [0.01], "batch_size": [32], "inter_seq_best": [1]},
        # "pwn": {"seq_feature_sz": [4, 8], "stat_feature_sz": [4, 8], "learning_rate": [0.001, 0.01], "batch_size": [32, 128], "inter_seq_best": [1]},
        "lr": {"reg_strength": [pow(10, -3), pow(10, -2), pow(10, -1), pow(10, 0), pow(10, 1), pow(10, 2), pow(10, 3)],
               "solver": ["lbfgs"]},
        "nb": {"var_smoothing": np.logspace(0, -9, num=10)},
        "dt": {"max_depth": [2, 3, 4], "min_samples_split": [2]},
        "knn": {"n_neighbors": [3, 5, 10]},
        "xgb": {"max_depth": [2, 6, 12], "learning_rate": [0.3, 0.1, 0.03]},
        "rf": {"max_depth": [2, 6, 12], "n_estimators": [100, 200, 400], "max_leaf_nodes": [2, 6, 12]}
    }

    if data_set == "sepsis":
        for seed in [15, 37, 98, 137, 245]:  # [15, 37, 98, 137, 245]:
            for mode in ['xgb']:  # 'pwn', 'lr', 'dt', 'knn', 'nb', 'xgb', 'rf'
                procedure = mode
                for target_activity in ['Admission IC']:

                    np.random.seed(seed=seed)
                    torch.manual_seed(seed=seed)

                    x_seqs, x_statics, y, x_time_vals_final, seq_features, static_features = data.get_sepsis_data(
                        target_activity, max_len, min_len)

                    x_seqs_train, x_statics_train, y_train, x_seqs_val, x_statics_val, y_val = \
                        evaluate(x_seqs, x_statics, y, mode, target_activity, data_set, hpos, hpo, static_features, seed)

    elif data_set == "bpi2012":
        for seed in [15]:  # 15, 37, 98, 137, 245]:
            for mode in ['rf']:  # 'pwn', 'lr', 'dt', 'knn', 'nb', 'xgb', 'rf'
                procedure = mode

                np.random.seed(seed=seed)
                torch.manual_seed(seed=seed)

                x_seqs, x_statics, y, x_time_vals_final, seq_features, static_features = data.get_bpi_data(max_len, min_len)

                x_seqs_train, x_statics_train, y_train, x_seqs_val, x_statics_val, y_val = \
                    evaluate(x_seqs, x_statics, y, mode, "deviant", data_set, hpos, hpo, static_features, seed)

    elif data_set == "hospital":
        for seed in [15]:  # [15, 37, 98, 137, 245]:
            for mode in ['rf']:  # 'pwn', 'lr', 'dt', 'knn', 'nb', 'xgb', 'rf'
                procedure = mode

                np.random.seed(seed=seed)
                torch.manual_seed(seed=seed)

                x_seqs, x_statics, y, x_time_vals_final, seq_features, static_features = data.get_hospital_data(max_len, min_len)

                x_seqs_train, x_statics_train, y_train, x_seqs_val, x_statics_val, y_val = \
                    evaluate(x_seqs, x_statics, y, mode, "deviant", data_set, hpos, hpo, static_features, seed)

    else:
        print("Data set not available!")
