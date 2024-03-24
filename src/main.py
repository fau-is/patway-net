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
from src.nns import MLP, SLP, LSTM
from sklearn.model_selection import StratifiedKFold
import pickle
import xgboost as xgb
import time
import copy

max_len = 50
min_len = 3
min_size_prefix = 1
val_size = 0.2
train_size = 0.8
hpo = True
save_baseline_model = False


def concatenate_tensor_matrix(x_seq, x_stat):
    """
    This function reshapes a 3D tensor 'x_seq' into a 2D matrix and then concatenates it with another 2D matrix 'x_stat'.
    The reshaping of 'x_seq' is done such that the second and third dimensions are flattened into a single dimension.
    The concatenation is performed along the second axis (columns).

    Parameters:
        x_seq (array_like): The 3D input tensor.
        x_stat (array_like): The 2D matrix to be concatenated with 'x_seq'.

    Returns:
        x_concat(array_like): The resulting 2D matrix after reshaping and concatenation.
        
    """
    x_train_seq_ = x_seq.reshape(-1, x_seq.shape[1] * x_seq.shape[2])
    x_concat = np.concatenate((x_train_seq_, x_stat), axis=1)

    return x_concat


def train_rf(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo, data_set, target_activity=None):
    """
    This function trains a Random Forest Classifier with or without hyperparameter optimization.
    If hyperparameter, optimization is enabled(hpo=true), the function will train the model with all possible combinations of the
    hyperparameters such as depth of the trees(max_depth), number of trees(n_estimators) or maximum number of leaf nodes(max_leaf_nodes) and select the best model based on the validation AUC.
    If hyperparameter optimization is disabled(hpo=false), the function will train the model with the default hyperparameters.

    Parameters:
        x_train_seq (?): The training sequences.
        x_train_stat (array_like): The training input static features.
        y_train (array_like): The training target labels.
        x_val_seq (?): The validation sequences.
        x_val_stat (array_like): The validation input static features.
        y_val (array_like): The validation target labels.
        hpo (bool): Whether to perform hyperparameter optimization.
                    Defaults to False.
        hpos (dict): The hyperparameter optimization space.
                     Defaults to None.
        data_set (str): The name of the dataset.
        arget_activity (str): The target activity.
                                Defaults to None.
    Returns:
    If hpo is True:
        best_model (RandomForestClassifier): The best trained model.
        best_hpos (dict): The best hyperparameters.

    If hpo is False:
        model (RandomForestClassifier): The trained model, without hyperparameter optimization.

    """
    if hpo:
        best_model = ""
        best_hpos = ""
        aucs = []
        #loop through all possible combinations of max_depth, n_estimators, max_leaf_nodes hyperparameters
        for max_depth in hpos["rf"]["max_depth"]:
            for n_estimators in hpos["rf"]["n_estimators"]:
                for max_leaf_nodes in hpos["rf"]["max_leaf_nodes"]:

                    model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators,
                                                   max_leaf_nodes=max_leaf_nodes)
                    model.fit(x_train_stat, np.ravel(y_train))

                    preds_proba = model.predict_proba(x_val_stat)
                    preds_proba = [pred_proba[1] for pred_proba in preds_proba]
                    #calculate the AUC
                    try:
                        auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                        if np.isnan(auc):
                            auc = 0
                    except:
                        auc = 0
                    aucs.append(auc)
                    #select the best model based on the AUC
                    if auc >= max(aucs):
                        best_model = model
                        best_hpos = {"max_depth": max_depth, "n_estimators": n_estimators,
                                     "max_leaf_nodes": max_leaf_nodes}
        #write the best hyperparameters and the AUCs to a file
        f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos_{seed}.txt', 'a+')
        f.write(str(best_hpos))
        f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
        f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
        f.write(f'Std,{np.std(aucs, ddof=1)}\n')
        f.close()

        return best_model, best_hpos
    #default learning wthout hyperparameter optimization
    else:
        model = RandomForestClassifier()
        model.fit(x_train_stat, np.ravel(y_train))

        return model

def train_xgb(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo, data_set, target_activity=None):
    
    """
    This function trains an XGBoost Classifier with or without hyperparameter optimization.
    If hyperparameter optimization is enabled(hpo=true), the function will train the model with all possible combinations of the
    hyperparameters such as max_depth and learning_rate and select the best model based on the validation AUC.
    If hyperparameter optimization is disabled(hpo=false), the function will train the model with the default hyperparameters.

   Parameters:
        x_train_seq (?): The training sequences.
        x_train_stat (array_like): The training input static features.
        y_train (array_like): The training target labels.
        x_val_seq (?): The validation sequences.
        x_val_stat (array_like): The validation input static features.
        y_val (array_like): The validation target labels.
        hpo (bool): Whether to perform hyperparameter optimization.
                Defaults to False.
        hpos (dict): The hyperparameter optimization space.
                     Defaults to None.
        data_set (str): The name of the dataset.
        target_activity (str): The target activity.
                                Defaults to None.
    Returns:
    If hpo is True:
        best_model (XGBClassifier): The best trained model.
        best_hpos (dict): The best hyperparameters.

    If hpo is False:
        model (XGBClassifier): The trained model, without hyperparameter optimization.
    """

    if hpo:
        best_model = ""
        best_hpos = ""
        aucs = []
        #loop through all possible combinations of hyperparameters
        for max_depth in hpos["xgb"]["max_depth"]:
            for learning_rate in hpos["xgb"]["learning_rate"]:

                model = xgb.XGBClassifier(max_depth=max_depth, learning_rate=learning_rate)
                model.fit(x_train_stat, np.ravel(y_train))

                preds_proba = model.predict_proba(x_val_stat)
                preds_proba = [pred_proba[1] for pred_proba in preds_proba]
                #calculate the AUC
                try:
                    auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                    if np.isnan(auc):
                        auc = 0
                except:
                    auc = 0
                aucs.append(auc)
                #select the best model based on the AUC
                if auc >= max(aucs):
                    best_model = model
                    best_hpos = {"max_depth": max_depth, "learning_rate": learning_rate}
        #write the best hyperparameters and the AUCs to a file
        f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos_{seed}.txt', 'a+')
        f.write(str(best_hpos))
        f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
        f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
        f.write(f'Std,{np.std(aucs, ddof=1)}\n')
        f.close()

        return best_model, best_hpos
    #default learning wthout hyperparameter optimization
    else:
        model = xgb.XGBClassifier()
        model.fit(x_train_stat, np.ravel(y_train))

        return model


def train_lr(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo, data_set, target_activity=None):
    """
    This function trains a Logistic Regression Classifier with or without hyperparameter optimization.
    If hyperparameter optimization is enabled(hpo=true), the function will train the model with all possible combinations of the
    hyperparameters and select the best model based on the validation AUC.
    If hyperparameter optimization is disabled(hpo=false), the function will train the model with the default hyperparameters.

    Parameters:
        x_train_seq (?): The training sequences.
        x_train_stat (array_like): The training input static features.
        y_train (array_like): The training target labels.
        x_val_seq (?): The validation sequences.
        x_val_stat (array_like): The validation input static features.
        y_val (array_like): The validation target labels.
        hpo (bool): Whether to perform hyperparameter optimization.
                    Defaults to False.
        hpos (dict): The hyperparameter optimization space.
                    Defaults to None.
        data_set (str): The name of the dataset.
        target_activity (str): The target activity.
                                Defaults to None.
    Returns:
    If hpo is True:
        best_model (LogisticRegression): The best trained model.
        best_hpos (dict): The best hyperparameters.

    If hpo is False:
        model (LogisticRegression): The trained model, without hyperparameter optimization.


    """

    if hpo:
        best_model = ""
        best_hpos = ""
        aucs = []
        #loop through all possible combinations of hyperparameters
        for c in hpos["lr"]["reg_strength"]:
            for solver in hpos["lr"]["solver"]:

                model = LogisticRegression(C=c, solver=solver)
                model.fit(x_train_stat, np.ravel(y_train))
                preds_proba = model.predict_proba(x_val_stat)
                preds_proba = [pred_proba[1] for pred_proba in preds_proba]
                #calculate the AUC
                try:
                    auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                    if np.isnan(auc):
                        auc = 0
                except:
                    auc = 0
                aucs.append(auc)
                #select the best model based on the AUC
                if auc >= max(aucs):
                    best_model = model
                    best_hpos = {"c": c, "solver": solver}
        #write the best hyperparameters and the AUCs to a file
        f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos_{seed}.txt', 'a+')
        f.write(str(best_hpos))
        f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
        f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
        f.write(f'Std,{np.std(aucs, ddof=1)}\n')
        f.close()

        return best_model, best_hpos
    #default learning wthout hyperparameter optimization
    else:
        model = LogisticRegression()
        model.fit(x_train_stat, np.ravel(y_train))

        return model


def train_nb(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo, data_set, target_activity=None):
    """
    This function trains a Naive Bayes Classifier with or without hyperparameter optimization.
    If hyperparameter optimization is enabled(hpo=true), the function will train the model with all possible combinations of the
    hyperparameters and select the best model based on the validation AUC.
    If hyperparameter optimization is disabled(hpo=false), the function will train the model with the default hyperparameters.

    Parameters:
        x_train_seq (?): The training sequences.
        x_train_stat (array_like): The training input static features.
        y_train (array_like): The training target labels.
        x_val_seq (?): The validation sequences.
        x_val_stat (array_like): The validation input static features.
        y_val (array_like): The validation target labels.
        hpo (bool): Whether to perform hyperparameter optimization.
                    Defaults to False.
        hpos (dict): The hyperparameter optimization space.
                     Defaults to None.
        data_set (str): The name of the dataset.
        target_activity (str): The target activity.
                            Defaults to None.
    Returns:
    If hpo is True:
        best_model (GaussianNB): The best trained model.
        best_hpos (dict): The best hyperparameters.

    If hpo is False:
        model (GaussianNB): The trained model, without hyperparameter optimization.
    """
    if hpo:
        best_model = ""
        best_hpos = ""
        aucs = []
        #loop through all possible var_smoothing values
        for var_smoothing in hpos["nb"]["var_smoothing"]:

            model = GaussianNB(var_smoothing=var_smoothing)
            model.fit(x_train_stat, np.ravel(y_train))
            preds_proba = model.predict_proba(x_val_stat)
            preds_proba = [pred_proba[1] for pred_proba in preds_proba]
            #calculate the AUC
            try:
                auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                if np.isnan(auc):
                    auc = 0
            except:
                auc = 0
            aucs.append(auc)
            #select the best model based on the AUC
            if auc >= max(aucs):
                best_model = model
                best_hpos = {"var_smoothing": var_smoothing}
        #write the best hyperparameters and the AUCs to a file
        f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos_{seed}.txt', 'a+')
        f.write(str(best_hpos) + '\n')
        f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
        f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
        f.write(f'Std,{np.std(aucs, ddof=1)}\n')
        f.close()

        return best_model, best_hpos
    #default learning wthout hyperparameter optimization
    else:
        model = GaussianNB()
        model.fit(x_train_stat, np.ravel(y_train))

        return model


def train_dt(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo, data_set, target_activity=None):
    """
    This function trains a Decision Tree Classifier with or without hyperparameter optimization.
    If hyperparameter, optimization is enabled(hpo=true), the function will train the model with all possible combinations of the
    hyperparameters such as depth of the trees(max_depth), minimum number of samples required to split an internal node(min_samples_split) 
    and select the best model based on the validation AUC.
    If hyperparameter optimization is disabled(hpo=false), the function will train the model with the default hyperparameters.

    Parameters:
        x_train_seq (?): The training sequences.
        x_train_stat (array_like): The training input static features.
        y_train (array_like): The training target labels.
        x_val_seq (?): The validation sequences.
        x_val_stat (array_like): The validation input static features.
        y_val (array_like): The validation target labels.
        hpo (bool): Whether to perform hyperparameter optimization.
                    Defaults to False.
        hpos (dict): The hyperparameter optimization space.
                    Defaults to None.
        data_set (str): The name of the dataset.
        target_activity (str): The target activity.
                                Defaults to None.

    Returns:
    If hpo is True:
        best_model (DecisionTreeClassifier): The best trained model.
        best_hpos (dict): The best hyperparameters.

    If hpo is False:
        model (DecisionTreeClassifier): The trained model, without hyperparameter optimization.

    """
    if hpo:
        best_model = ""
        best_hpos = ""
        aucs = []
        #loop through all possible combinations of max_depth and min_samples_split hyperparameters
        for max_depth in hpos["dt"]["max_depth"]:
            for min_samples_split in hpos["dt"]["min_samples_split"]:

                model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
                model.fit(x_train_stat, np.ravel(y_train))
                preds_proba = model.predict_proba(x_val_stat)
                preds_proba = [pred_proba[1] for pred_proba in preds_proba]
                #calculate the AUC
                try:
                    auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                    if np.isnan(auc):
                        auc = 0
                except:
                    auc = 0

                aucs.append(auc)
                #select the best model based on the AUC
                if auc >= max(aucs):
                    best_model = model
                    best_hpos = {"max_depth": max_depth, "min_samples_split": min_samples_split}
        #write the best hyperparameters and the AUCs to a text file
        f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos_{seed}.txt', 'a+')
        f.write(str(best_hpos) + '\n')
        f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
        f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
        f.write(f'Std,{np.std(aucs, ddof=1)}\n')
        f.close()

        return best_model, best_hpos
    #default learning wthout hyperparameter optimization
    else:
        model = DecisionTreeClassifier()
        model.fit(x_train_stat, np.ravel(y_train))

        return model


def train_knn(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo, data_set, target_activity=None):
    """
    This function trains a K-Nearest Neighbors Classifier with or without hyperparameter optimization.
    If hyperparameter optimization is enabled(hpo=true), the function will train the model with all possible combinations of the
    hyperparameters such as number of neighbors(n_neighbors) and select the best model based on the validation AUC.
    If hyperparameter optimization is disabled(hpo=false), the function will train the model with the default hyperparameters.

    Parameters:
        x_train_seq (?): The training sequences.
        x_train_stat (array_like): The training input static features.
        y_train (array_like): The training target labels.
        x_val_seq (?): The validation sequences.
        x_val_stat (array_like): The validation input static features.
        y_val (array_like): The validation target labels.
        hpo (bool): Whether to perform hyperparameter optimization.
                Defaults to False.
        hpos (dict): The hyperparameter optimization space.
                    Defaults to None.
        data_set (str): The name of the dataset.
        target_activity (str): The target activity.
                            Defaults to None.

    Returns:
    If hpo is True:
        best_model (KNeighborsClassifier): The best trained model.
        best_hpos (dict): The best hyperparameters.

    If hpo is False:
        model (KNeighborsClassifier): The trained model, without hyperparameter optimization.
    """
    if hpo:
        best_model = ""
        best_hpos = ""
        aucs = []
        #loop through all possible combinations of hyperparameters
        for n_neighbors in hpos["knn"]["n_neighbors"]:

            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            model.fit(x_train_stat, np.ravel(y_train))
            preds_proba = model.predict_proba(x_val_stat)
            preds_proba = [pred_proba[1] for pred_proba in preds_proba]
            #calculate the AUC
            try:
                auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                if np.isnan(auc):
                    auc = 0
            except:
                auc = 0
            aucs.append(auc)
            #select the best model based on the AUC
            if auc >= max(aucs):
                best_model = model
                best_hpos = {"n_eighbors": n_neighbors}
        #write the best hyperparameters and the AUCs to a text file
        f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos_{seed}.txt', 'a+')
        f.write(str(best_hpos) + '\n')
        f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
        f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
        f.write(f'Std,{np.std(aucs, ddof=1)}\n')
        f.close()

        return best_model, best_hpos
    #default learning wthout hyperparameter optimization
    else:
        model = KNeighborsClassifier()
        model.fit(x_train_stat, np.ravel(y_train))

        return model


def train_lstm(x_train_seq, x_train_stat, y_train, id, x_val_seq=False, x_val_stat=False, y_val=False, hpos=False,
    hpo=False, mode="pwn", data_set="sepsis", target_activity=None):
    """
    This function trains a Long Short-Term Memory (LSTM) model with or without hyperparameter optimization.
    If hyperparameter optimization is enabled (hpo=true), the function will train the model with all possible combinations of the
    hyperparameters such as learning_rate, batch_size, seq_feature_sz, stat_feature_sz and select the best model based on the validation AUC.

    Parameters:
        x_train_seq (array_like): The training sequences.
        x_train_stat (array_like): The training input static features.
        y_train (array_like): The training target labels.
        id (int): The id of the model.
        x_val_seq (array_like): The validation sequences.
        x_val_stat (array_like): The validation input static features.
        y_val (array_like): The validation target labels.
        hpo (bool): Whether to perform hyperparameter optimization.
                    Defaults to False.
        hpos (dict): The hyperparameter optimization space.
                        Defaults to None.
        mode (str): The mode of the LSTM model.
                    Defaults to "pwn".
        data_set (str): The name of the dataset.
                        Defaults to "sepsis".
        target_activity (str): The target activity.
                                Defaults to None.

    Returns:
    If hpo is True:
        best_model (Net): The best trained model.
        best_hpos (dict): The best hyperparameters.

    """
    max_case_len = x_train_seq.shape[1]
    num_features_seq = x_train_seq.shape[2]
    num_features_stat = x_train_stat.shape[1]

    import torch
    import copy
    x_train_stat_ = torch.reshape(x_train_stat, (-1, 1, x_train_stat.shape[1]))
    x_train_stats = copy.copy(x_train_stat_)
    T = x_train_seq.shape[1]
    for t in range(1, T):
        x_train_stats = torch.concat((x_train_stats, x_train_stat_), 1)
    x_train_seq = torch.concat((x_train_seq, x_train_stats), 2).float()

    if x_val_seq is not False:
        x_val_stat_ = torch.reshape(x_val_stat, (-1, 1, x_val_stat.shape[1]))
        x_val_stats = copy.copy(x_val_stat_)
        T = x_val_seq.shape[1]
        for t in range(1, T):
            x_val_stats = torch.concat((x_val_stats, x_val_stat_), 1)
        x_val_seq = torch.concat((x_val_seq, x_val_stats), 2).float()

    patience = 10
    epochs = 100

    import torch
    import torch.nn as nn
    import torch.optim as optim

    if hpo:
        best_model = ""
        best_hpos = ""
        aucs = []

        # x_train_seq = torch.from_numpy(x_train_seq)
        y_train = torch.from_numpy(y_train)

        for learning_rate in hpos["lstm"]["learning_rate"]:
            for batch_size in hpos["lstm"]["batch_size"]:
                for hidden_sz in hpos["lstm"]["hidden_sz"]:

                    model = LSTM(input_size=num_features_seq+num_features_stat, hidden_size=hidden_sz)
                    criterion = nn.BCEWithLogitsLoss()
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.000)
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
                        y_train = y_train[idx]
                        number_batches = x_train_seq.shape[0] // batch_size

                        for i in range(number_batches):
                            optimizer.zero_grad()  # clean up step for PyTorch
                            out = model(x_train_seq[i * batch_size:(i + 1) * batch_size])
                            loss = criterion(out, y_train[i * batch_size:(i + 1) * batch_size].double())
                            loss.backward()  # compute updates for each parameter
                            optimizer.step()  # make the updates for each parameter

                        # Early stopping
                        def validation(model, x_val_seq, y_val, loss_function):

                            # x_val_seq = torch.from_numpy(x_val_seq)
                            y_val = torch.from_numpy(y_val)

                            model.eval()
                            loss_total = 0
                            number_batches = x_val_seq.shape[0] // batch_size

                            with torch.no_grad():
                                for i in range(number_batches):
                                    out = model(x_val_seq[i * batch_size: (i + 1) * batch_size])
                                    loss = loss_function(out, y_val[i * batch_size:(i + 1) * batch_size].double())
                                    loss_total += loss.item()
                            return loss_total / number_batches

                        current_val_loss = validation(model, x_val_seq, y_val, criterion)
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

                        # x_val_seq_ = torch.from_numpy(x_val_seq)
                        preds_proba = torch.sigmoid(model_best_es(x_val_seq))
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
                                         "hidden_sz": hidden_sz}

            f = open(f'../output/{data_set}_{mode}_{target_activity}_hpos_{seed}.txt', 'a+')
            f.write(str(best_hpos) + '\n')
            f.write("Validation aucs," + ",".join([str(x) for x in aucs]) + '\n')
            f.write(f'Avg,{sum(aucs) / len(aucs)}\n')
            f.write(f'Std,{np.std(aucs, ddof=1)}\n')
            f.close()

            torch.save(model, os.path.join("../model", f"model_{mode}_{id}_{seed}"))

            return best_model, best_hpos
        else:
            pass

def train_pwn(x_train_seq, x_train_stat, y_train, id, x_val_seq=False, x_val_stat=False, y_val=False, hpos=False,
              hpo=False, mode="pwn", data_set="sepsis", target_activity=None):
    """
    This function trains PatWay-Net with or without hyperparameter optimization.
    If hyperparameter optimization is enabled (hpo=true), the function will train the model with all possible combinations of the
    hyperparameters such as learning_rate, batch_size, seq_feature_sz, stat_feature_sz and select the best model based on the validation AUC.

    Parameters:
        x_train_seq (array_like): The training sequences.
        x_train_stat (array_like): The training input static features.
        y_train (array_like): The training target labels.
        id (int): The id of the model.
        x_val_seq (array_like): The validation sequences.
        x_val_stat (array_like): The validation input static features.
        y_val (array_like): The validation target labels.
        hpo (bool): Whether to perform hyperparameter optimization.
                    Defaults to False.
        hpos (dict): The hyperparameter optimization space.
                        Defaults to None.
        mode (str): The mode of the LSTM model.
                    Defaults to "pwn".
        data_set (str): The name of the dataset.
                        Defaults to "sepsis".
        target_activity (str): The target activity.
                                Defaults to None.

    Returns:
    If hpo is True:
        best_model (Net): The best trained model.
        best_hpos (dict): The best hyperparameters.

    """
    max_case_len = x_train_seq.shape[1]
    num_features_seq = x_train_seq.shape[2]
    num_features_stat = x_train_stat.shape[1]

    interactions_seq_itr = 100
    patience = 10
    epochs = 100
    lstm_mode = ["pwn", "pwn_no_inter", "pwn_only_feat_static", "pwn_all_feat_seq", "lstm"][2]

    # Check the mode of the LSTM model and set the corresponding settings
    if lstm_mode == "pwn":
        masking = True
        interactions_seq_auto = True
        only_feat_static = False
        all_feat_seq = False

    elif lstm_mode == "pwn_no_inter":
        masking = True
        interactions_seq_auto = False
        only_feat_static = False
        all_feat_seq = False

    elif lstm_mode == "pwn_only_feat_static":
        masking = False
        interactions_seq_auto = False
        only_feat_static = True
        all_feat_seq = False

    elif lstm_mode == "lstm":
        masking = False
        interactions_seq_auto = False
        only_feat_static = False
        all_feat_seq = False

    elif lstm_mode == "pwn_all_feat_seq":
        masking = False
        interactions_seq_auto = False
        only_feat_static = False
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
        
        #loop through all possible combinations of hyperparameters
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
                                        only_static=only_feat_static,
                                        all_feat_seq=all_feat_seq,
                                        masking=masking,
                                        mlp_hidden_size=stat_feature_sz,
                                        x_seq=x_train_seq,
                                        x_stat=x_train_stat,
                                        y=y_train)

                            criterion = nn.BCEWithLogitsLoss()
                            lamb_mlp = 0.0
                            lamb_lstm = 0.0
                            # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
                            # optimizer = optim.NAdam(model.parameters(), lr=learning_rate)
                            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.000)  # weight_decay = 0.002, 0.001
                            idx = np.arange(len(x_train_seq))
                            
                            import copy
                            best_val_loss = np.inf
                            trigger_times = 0
                            model_best_es = copy.deepcopy(model)
                            flag_es = False
                            
                            # Shuffle the training data for each epoch
                            for epoch in range(epochs):
                                print(f"Epoch: {epoch + 1}")
                                np.random.shuffle(idx)
                                x_train_seq = x_train_seq[idx]
                                x_train_stat = x_train_stat[idx]
                                y_train = y_train[idx]
                                number_batches = x_train_seq.shape[0] // batch_size

                                for i in range(number_batches):
                                    optimizer.zero_grad()  # clean up step for PyTorch
                                    """
                                    out, out_mlp, out_lstm = model(x_train_seq[i * batch_size:(i + 1) * batch_size],
                                                x_train_stat[i * batch_size:(i + 1) * batch_size], out_=True)
                                    loss = criterion(out, y_train[i * batch_size:(i + 1) * batch_size].double()) + \
                                           lamb_mlp * torch.mean(out_mlp ** 2) + lamb_lstm * torch.mean(out_lstm ** 2)
                                    """
                                    # Apply the model to the current batch of training data to get the prediction 'out'
                                    out = model(x_train_seq[i * batch_size:(i + 1) * batch_size], x_train_stat[i * batch_size:(i + 1) * batch_size])
                                    # Calculate the loss by comparing the model's prediction 'out' with the corresponding target values in 'y_train'
                                    loss = criterion(out, y_train[i * batch_size:(i + 1) * batch_size].double())
                                    loss.backward()  # compute updates for each parameter
                                    optimizer.step()  # make the updates for each parameter

                                # Early stopping
                                def validation(model, x_val_seq, x_val_stat, y_val, loss_function, lamb, lamb2):
                                    """
                                    This function is used for early stopping during the training process.
                                    It calculates the validation loss over the validation data. If the validation loss does not improve over a certain 
                                    number of epochs, the training process will be stopped early. This helps to prevent overfitting 
                                    by stopping the training when the model starts to perform worse on the validation data.

                                    Parameters:
                                        model: The model being trained.
                                        x_val_seq, x_val_stat: The sequential and static validation data, respectively.
                                        y_val: The target values for the validation data.
                                        oss_function: The function used to calculate the loss.
                                        amb, lamb2: Regularization parameters, used in the commented-out alternative loss calculation.

                                    Returns:
                                        loss_total / number_batches: The average validation loss per batch
                                    """    
                                
                                    x_val_stat = torch.from_numpy(x_val_stat)
                                    x_val_seq = torch.from_numpy(x_val_seq)
                                    y_val = torch.from_numpy(y_val)
                                    
                                    model.eval()# Set the model to evaluation mode
                                    loss_total = 0
                                    number_batches = x_val_seq.shape[0] // batch_size

                                    # Calculate the validation loss for each batch
                                    with torch.no_grad():
                                        for i in range(number_batches):
                                            """
                                            out, out_mlp, out_lstm = model(x_val_seq[i * batch_size: (i + 1) * batch_size],
                                                        x_val_stat[i * batch_size: (i + 1) * batch_size], out_=True)
                                            loss = loss_function(out, y_val[i * batch_size:(i + 1) * batch_size].double()) + \
                                                   lamb * torch.mean(out_mlp ** 2) + lamb2 * torch.mean(out_lstm ** 2)
                                            """
                                            out = model(x_val_seq[i * batch_size: (i + 1) * batch_size], x_val_stat[i * batch_size: (i + 1) * batch_size])
                                            loss = loss_function(out, y_val[i * batch_size:(i + 1) * batch_size].double())
                                            loss_total += loss.item()
                                    return loss_total / number_batches
                                
                                # Calculate the validation loss
                                current_val_loss = validation(model, x_val_seq, x_val_stat, y_val, criterion, lamb_mlp, lamb_lstm)
                                print('Validation loss:', current_val_loss)

                                # Early stopping mechanism
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
                                    model_best_es = copy.deepcopy(model) # Save the current model as the best model so far
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
                                #calculate the AUC
                                try:
                                    auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                                    if np.isnan(auc):
                                        auc = 0
                                except:
                                    auc = 0
                                aucs.append(auc)
                                #saving the best model and hyperparameters based on the AUC
                                if auc >= max(aucs):
                                    best_model = copy.deepcopy(model_best_es)
                                    best_hpos = {"learning_rate": learning_rate, "batch_size": batch_size,
                                                 "seq_feature_sz": seq_feature_sz,
                                                 "stat_feature_sz": stat_feature_sz,
                                                 "inter_seq_best": inter_seq_best}
                                    
            #write the best hyperparameters and the AUCs to a text file
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


def train_mlps_sln(x_train_seq, x_train_stat, y_train, id, x_val_seq=False, x_val_stat=False, y_val=False, hpos=False):
    """
    This function first trains per static feature a Multi-Layer Perceptron (MLP) model and 
    trains based on the outputs of the MLPs a Single-Layer Perceptron (SLP).      

    Parameters:
        x_train_seq (array_like): The training sequences.
        x_train_stat (array_like): The training input static features.
        y_train (array_like): The training target labels.  
        id (int): The id of the model.
        x_val_seq (array_like): The validation sequences.
                            Defaults to False.
        x_val_stat (array_like): The validation input static features.
                             Defaults to False.
        y_val (array_like): The validation target labels.
                        Defaults to False.
        hpos (dict): The hyperparameter optimization space.
                    Defaults to None.
    Returns:
        models (dict): The trained models.
    """

    num_features_stat = x_train_stat.shape[1]

    models = {"mlps": [], "slp": ""} # Initialize dictionary to store models

    patience = 10
    epochs = 100

    import torch
    import torch.nn as nn
    import torch.optim as optim

    x_train_stat = torch.from_numpy(x_train_stat)
    y_train = torch.from_numpy(y_train)

    # Iterate through each static feature
    for j in range(0, num_features_stat):

        best_model = ""
        aucs = []
        #loop through all possible combinations of hyperparameters
        for learning_rate in hpos["mlps_sln"]["learning_rate"]:
            for batch_size in hpos["mlps_sln"]["batch_size"]:
                for stat_feature_sz in hpos["mlps_sln"]["stat_feature_sz"]:
                    
                    # Define the MLP model
                    model = MLP(input_size=1, hidden_size=stat_feature_sz)
                    criterion = nn.BCEWithLogitsLoss()
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.000)
                    idx = np.arange(len(x_train_seq))

                    # Initialize variables for early stopping
                    import copy
                    best_val_loss = np.inf
                    trigger_times = 0
                    model_best_es = copy.deepcopy(model)
                    flag_es = False

                    # Shuffle the training data for each epoch
                    for epoch in range(epochs):
                        print(f"Epoch: {epoch + 1} --- MLP: {j + 1}")
                        np.random.shuffle(idx)
                        x_train_stat = x_train_stat[idx]
                        y_train = y_train[idx]
                        number_batches = x_train_stat.shape[0] // batch_size

                        # Train the model for each batch
                        for i in range(number_batches):
                            optimizer.zero_grad()  # clean up step for PyTorch
                            out = model(x_train_stat[i * batch_size:(i + 1) * batch_size, j].reshape(-1,1).float())
                            loss = criterion(out, y_train[i * batch_size:(i + 1) * batch_size].double())
                            loss.backward()  # compute updates for each parameter
                            optimizer.step()  # make the updates for each parameter

                        # Early stopping
                        def validation(model, x_val_stat, y_val, loss_function):
                            """
                            This function is used for early stopping during the training process.
                            It calculates the validation loss over the validation data. If the validation loss does not improve over a certain 
                            number of epochs, the training process will be stopped early. This helps to prevent overfitting 
                            by stopping the training when the model starts to perform worse on the validation data.

                            Parameters:
                                model: The model being trained.
                                x_val_seq, x_val_stat: The sequential and static validation data, respectively.
                                y_val: The target values for the validation data.
                                loss_function: The function used to calculate the loss.
                                lamb, lamb2: Regularization parameters, used in the commented-out alternative loss calculation.

                            Returns:
                                loss_total / number_batches: The average validation loss per batch
                            """ 

                            x_val_stat = torch.from_numpy(x_val_stat)
                            y_val = torch.from_numpy(y_val)

                            model.eval()# Set the model to evaluation mode
                            loss_total = 0
                            number_batches = x_val_stat.shape[0] // batch_size

                            # Calculate the validation loss for each batch
                            with torch.no_grad():
                                for i in range(number_batches):
                                    out = model(x_train_stat[i * batch_size:(i + 1) * batch_size, j].reshape(-1,1).float())
                                    loss = loss_function(out, y_val[i * batch_size:(i + 1) * batch_size].double())
                                    loss_total += loss.item()
                            return loss_total / number_batches
                        
                        # Calculate the validation loss
                        current_val_loss = validation(model, x_val_stat, y_val, criterion)
                        print('Validation loss:', current_val_loss)

                        # Early stopping mechanism
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
                            model_best_es = copy.deepcopy(model)# Save the current model as the best model so far
                            best_val_loss = current_val_loss

                        if flag_es:
                            break

                    # Select model based on val auc
                    model_best_es.eval()
                    with torch.no_grad():

                        x_val_stat_update_ = torch.from_numpy(x_val_stat)
                        preds_proba = torch.sigmoid(model_best_es(x_val_stat_update_.reshape(-1,1).float()))
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

        models["mlps"].append(best_model)

    # transform data 
    x_train_stat_update = x_train_stat
    x_val_stat_update = torch.from_numpy(x_val_stat)
    for c in range(0, num_features_stat):
        x_train_stat_update[:, c] = torch.sigmoid(models["mlps"][c](x_train_stat_update[:, c].reshape(-1,1).float())).reshape(-1)
        x_val_stat_update[:, c] = torch.sigmoid(models["mlps"][c](x_val_stat_update[:, c].reshape(-1, 1).float())).reshape(-1)

    # Fit MLP based on models["mlps"]
    best_model = ""
    aucs = []
    
    #loop through all possible combinations of hyperparameters
    for learning_rate in hpos["mlps_sln"]["learning_rate"]:
        for batch_size in hpos["mlps_sln"]["batch_size"]:

            model = SLP(input_size=num_features_stat)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.000)
            idx = np.arange(len(x_train_seq))

            import copy
            best_val_loss = np.inf
            trigger_times = 0
            model_best_es = copy.deepcopy(model)
            flag_es = False

            # Shuffle the training data for each epoch
            for epoch in range(epochs):
                print(f"Epoch: {epoch + 1} --- SLP")
                np.random.shuffle(idx)
                x_train_stat_update = x_train_stat_update[idx]
                y_train = y_train[idx]
                number_batches = x_train_stat_update.shape[0] // batch_size

                for i in range(number_batches):
                    optimizer.zero_grad()  # clean up step for PyTorch
                    out = model(x_train_stat_update[i * batch_size:(i + 1) * batch_size].float())
                    loss = criterion(out, y_train[i * batch_size:(i + 1) * batch_size].double())
                    loss.backward(retain_graph=True)  # compute updates for each parameter
                    optimizer.step()  # make the updates for each parameter

                # Early stopping mechanism
                def validation(model, x_val_stat, y_val, loss_function):

                    y_val = torch.from_numpy(y_val)

                    model.eval()
                    loss_total = 0
                    number_batches = x_val_stat.shape[0] // batch_size

                    with torch.no_grad():
                        for i in range(number_batches):
                            out = model(x_train_stat[i * batch_size:(i + 1) * batch_size].float())
                            loss = loss_function(out, y_val[i * batch_size:(i + 1) * batch_size].double())
                            loss_total += loss.item()
                    return loss_total / number_batches

                 # Calculate validation loss
                current_val_loss = validation(model, x_val_stat_update, y_val, criterion)
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

                x_val_stat_update_ = x_val_stat_update
                preds_proba = torch.sigmoid(model_best_es(x_val_stat_update_.float()))
                preds_proba = [pred_proba[0] for pred_proba in preds_proba]
                #calculate the AUC
                try:
                    auc = metrics.roc_auc_score(y_true=y_val, y_score=preds_proba)
                    if np.isnan(auc):
                        auc = 0
                except:
                    auc = 0
                aucs.append(auc)
                #saving the best model based on the AUC
                if auc >= max(aucs):
                    best_model = copy.deepcopy(model_best_es)

    models["slp"] = best_model

    return models

def time_step_blow_up(X_seq, X_stat, y, max_len):
    """
    Creates prefix sequences from the given input sequences and statistics.
    
    Parameters:
         X_seq (list): List of input sequences.
         X_stat (list): List of input statistics.
         y (list): List of output labels.
         max_len (int): Maximum length of the prefix sequences.
        
    Returns:
         tuple: A tuple containing the final prefix sequences, input statistics, and output labels.
    """
    X_seq_prefix, X_stat_prefix, y_prefix, x_time_vals_prefix, ts = [], [], [], [], []

    # prefix
    for idx_seq in range(0, len(X_seq)):
        for idx_ts in range(min_size_prefix, len(X_seq[idx_seq]) + 1):
            X_seq_prefix.append(X_seq[idx_seq][0:idx_ts])
            X_stat_prefix.append(X_stat[idx_seq])
            y_prefix.append(y[idx_seq])

    """
    # no prefix
    for idx_seq in range(0, len(X_seq)):
        X_seq_prefix.append(X_seq[idx_seq])
        X_stat_prefix.append(X_stat[idx_seq])
        y_prefix.append(y[idx_seq])
    """

    X_seq_final = np.zeros((len(X_seq_prefix), max_len, len(X_seq_prefix[0][0])), dtype=np.float32) 
    X_stat_final = np.zeros((len(X_seq_prefix), len(X_stat_prefix[0]))) 
    
    for i, x in enumerate(X_seq_prefix):
        X_seq_final[i, :len(x), :] = np.array(x)
        X_stat_final[i, :] = np.array(X_stat_prefix[i])
    y_final = np.array(y_prefix).astype(np.int32)

    return X_seq_final, X_stat_final, y_final


def evaluate(x_seqs, x_statics, y, mode, target_activity, data_set, hpos, hpo, static_features, seed):
    """
    Evaluate the performance of different machine learning models.

    Parameters:
         x_seqs (list): List of input sequences.
         x_statics (list): List of static features.
         y (list): List of target labels.
         mode (str): Mode of evaluation (e.g., "pwn", "mlps_sln", "rf", "xgb", "lr", "nb", "dt", "knn").
         target_activity (str): Target activity for evaluation.
         data_set (str): Name of the dataset.
         hpos (dict): Hyperparameters for the model.
         hpo (bool): Flag indicating whether to perform hyperparameter optimization.
         static_features (list): List of static feature names.
         seed (int): Random seed for reproducibility.

    Returns:
         dict: Dictionary containing evaluation results.
         x_test_seq (list): List of test sequences.
         x_test_stat (list): List of test static features.
         y_test (list): List of test target labels.
         x_val_seq (list): List of validation sequences.
         X_val_stat (list): List of validation static features.
         y_val (list): List of validation target labels.
    """
    k = 5
    results = {}
    id = -1
    # Split the data into k folds
    skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    # Iterate through each fold
    for train_index_, test_index in skfold.split(X=x_statics, y=y):

        id += 1

        if id == 0:
            results['training_time'] = list()
            results['inference_time'] = list()
        # Split the data into training and validation sets
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
                
        if mode == "lstm":
            training_start_time = time.time()

            X_train_seq = torch.from_numpy(X_train_seq)
            X_train_stat = torch.from_numpy(X_train_stat)

            X_val_seq = torch.from_numpy(X_val_seq)
            X_val_stat = torch.from_numpy(X_val_stat)

            X_test_seq = torch.from_numpy(X_test_seq)
            X_test_stat = torch.from_numpy(X_test_stat)

            model, best_hpos = train_lstm(X_train_seq, X_train_stat, y_train.reshape(-1, 1), id, X_val_seq, X_val_stat,
                                         y_val.reshape(-1, 1), hpos, hpo, mode, data_set, target_activity=target_activity)

            results['training_time'].append(time.time() - training_start_time)

            X_train_stat_ = torch.reshape(X_train_stat, (-1, 1, X_train_stat.shape[1]))
            X_train_stats = copy.copy(X_train_stat_)
            T = X_train_seq.shape[1]
            for t in range(1, T):
                X_train_stats = torch.concat((X_train_stats, X_train_stat_), 1)
            X_train_seq = torch.concat((X_train_seq, X_train_stats), 2).float()

            X_val_stat_ = torch.reshape(X_val_stat, (-1, 1, X_val_stat.shape[1]))
            X_val_stats = copy.copy(X_val_stat_)
            T = X_val_seq.shape[1]
            for t in range(1, T):
                X_val_stats = torch.concat((X_val_stats, X_val_stat_), 1)
            X_val_seq = torch.concat((X_val_seq, X_val_stats), 2).float()

            X_test_stat_ = torch.reshape(X_test_stat, (-1, 1, X_test_stat.shape[1]))
            X_test_stats = copy.copy(X_test_stat_)
            T = X_test_seq.shape[1]
            for t in range(1, T):
                X_test_stats = torch.concat((X_test_stats, X_test_stat_), 1)
            X_test_seq = torch.concat((X_test_seq, X_test_stats), 2).float()

            model.eval()
            with torch.no_grad():
                preds_proba_train = torch.sigmoid(model(X_train_seq))
                preds_proba_val = torch.sigmoid(model(X_val_seq))
                inference_start_time = time.time()
                preds_proba_test = torch.sigmoid(model(X_test_seq))
                results['inference_time'].append(time.time() - inference_start_time)

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


        elif mode == "pwn":
            training_start_time = time.time()

            model, best_hpos = train_pwn(X_train_seq, X_train_stat, y_train.reshape(-1, 1), id, X_val_seq, X_val_stat,
                                         y_val.reshape(-1, 1), hpos, hpo, mode, data_set, target_activity=target_activity)

            results['training_time'].append(time.time() - training_start_time)

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
                inference_start_time = time.time()
                preds_proba_test = torch.sigmoid(model(X_test_seq, X_test_stat))
                results['inference_time'].append(time.time() - inference_start_time)

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

        elif mode == "mlps_sln":
            training_start_time = time.time()

            models = train_mlps_sln(X_train_seq, X_train_stat, y_train.reshape(-1, 1), id, X_val_seq, X_val_stat, y_val.reshape(-1, 1), hpos)

            results['training_time'].append(time.time() - training_start_time)

            X_train_stat = torch.from_numpy(X_train_stat)
            X_val_stat = torch.from_numpy(X_val_stat)
            X_test_stat = torch.from_numpy(X_test_stat)

            models["slp"].eval()
            with torch.no_grad():
                inference_start_time = time.time()
                num_features_stat = X_train_stat.shape[1]
                for c in range(0, num_features_stat):
                    X_train_stat[:, c] = torch.sigmoid(models["mlps"][c](X_train_stat[:, c].reshape(-1, 1).float())).reshape(-1)
                    X_val_stat[:, c] = torch.sigmoid(models["mlps"][c](X_val_stat[:, c].reshape(-1, 1).float())).reshape(-1)
                    X_test_stat[:, c] = torch.sigmoid(models["mlps"][c](X_test_stat[:, c].reshape(-1, 1).float())).reshape(-1)

                preds_proba_train = torch.sigmoid(models["slp"](X_train_stat.float()))
                preds_proba_val = torch.sigmoid(models["slp"](X_val_stat.float()))
                preds_proba_test = torch.sigmoid(models["slp"](X_test_stat.float()))
                results['inference_time'].append(time.time() - inference_start_time)

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
            training_start_time = time.time()
            model, best_hpos = train_lr(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                         y_val.reshape(-1, 1), hpos, hpo, data_set, target_activity=target_activity)
            results['training_time'].append(time.time() - training_start_time)

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
            inference_start_time = time.time()
            preds_proba_train = model.predict_proba(X_train_stat)
            results['inference_time'].append(time.time() - inference_start_time)
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
            """
            This function calculates the ROC AUC score.

            Parameters:
                gts (array-like): Ground truth (correct) target values.
                probs (array-like): Target scores.

            Returns:
                float: The ROC AUC score as a float in the range [0, 1].
            """
            try:
                auc = metrics.roc_auc_score(gts, probs)
                if np.isnan(auc):
                    auc = 0
                return auc
            except:
                return 0

        def calc_pr_auc(gts, probs):
            """
            This function calculates the PR AUC (Precision-Recall Area Under the Curve).

            Parameters:
                gts (array-like): Ground truth (correct) target values.
                probs (array-like): Target scores.

            Returns:
                float: The PR AUC score as a float in the range [0, 1].
            """
            try:
                precision, recall, thresholds = metrics.precision_recall_curve(gts, probs)
                auc = metrics.auc(recall, precision)
                if np.isnan(auc):
                    auc = 0
                return auc
            except:
                return 0

        def calc_mcc(gts, preds):
            """
            This function calculates the Matthews correlation coefficient (MCC).

            Parameters:
                gts (array-like): Ground truth (correct) target values.
                preds (array-like): Predictions, expected to be a binary vector corresponding to the predicted class (0 or 1) for each sample.

            Returns:
                float: The MCC score as a float in the range [-1, 1]. A higher absolute value indicates a better prediction.
            """
            try:
                return metrics.matthews_corrcoef(gts, preds)
            except:
                return 0

        def calc_f1(gts, preds, proba, label_id, corr=False):
            """
            This function calculates the F1 score, which is the harmonic mean of precision and recall,
            and optionally adjusts the threshold for prediction based on maximizing the difference
            between the True Positive Rate (TPR) and False Positive Rate (FPR) if correction is applied.

            Parameters:
                gts (array-like): Ground truth (correct) target values.
                preds (array-like): Predictions, expected to be a binary vector corresponding to the predicted class (0 or 1) for each sample.
                proba (array-like): Probability estimates of the positive class. This parameter is used
                                    only if `corr` is True to adjust the prediction threshold.
                label_id (int or string): The label of the positive class which `f1_score` and
                                          `precision_score` will consider as the positive label.
                corr (bool, optional): If True, the function will adjust the prediction threshold to maximize
                                       the difference between TPR and FPR before calculating the F1 score.
                                       Default is False.

            Returns:
                float: The calculated F1 score as a float. If `corr` is True, the score is calculated using
                       the adjusted threshold that maximizes the difference between TPR and FPR.
            """
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

        if mode in ["pwn", "lr", "mlps_sln"]:
            metrics_ = metrics_ + ["training_time", "inference_time"]
        # Save the results to a file
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
        # "mlps_sln": {"stat_feature_sz": [4], "learning_rate": [0.01], "batch_size": [32]},
        # "pwn": {"seq_feature_sz": [4], "stat_feature_sz": [4], "learning_rate": [0.01], "batch_size": [32], "inter_seq_best": [1]},
        "lstm": {"hidden_sz": [4, 32, 128], "learning_rate": [0.001, 0.01], "batch_size": [32, 128]},
        # "lstm": {"hidden_sz": [4], "learning_rate": [0.001], "batch_size": [32]},
        "pwn": {"seq_feature_sz": [4, 8], "stat_feature_sz": [4, 8], "learning_rate": [0.001, 0.01], "batch_size": [32, 128], "inter_seq_best": [1]},
        "lr": {"reg_strength": [pow(10, -3), pow(10, -2), pow(10, -1), pow(10, 0), pow(10, 1), pow(10, 2), pow(10, 3)],
               "solver": ["lbfgs"]},
        "nb": {"var_smoothing": np.logspace(0, -9, num=10)},
        "dt": {"max_depth": [2, 3, 4], "min_samples_split": [2]},
        "knn": {"n_neighbors": [3, 5, 10]},
        "xgb": {"max_depth": [2, 6, 12], "learning_rate": [0.3, 0.1, 0.03]},
        "rf": {"max_depth": [2, 6, 12], "n_estimators": [100, 200, 400], "max_leaf_nodes": [2, 6, 12]}
    }

    if data_set == "sepsis":
        for seed in [98]:  # [15, 37, 98, 137, 245]:
            for mode in ['lstm']:  # 'pwn', 'lr', 'dt', 'knn', 'nb', 'xgb', 'rf', 'mlps_sln', 'lstm'
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
            for mode in ['rf']:  # 'pwn', 'lr', 'dt', 'knn', 'nb', 'xgb', 'rf', 'mlps_sln', 'lstm'
                procedure = mode

                np.random.seed(seed=seed)
                torch.manual_seed(seed=seed)

                x_seqs, x_statics, y, x_time_vals_final, seq_features, static_features = data.get_bpi_data(max_len, min_len)

                x_seqs_train, x_statics_train, y_train, x_seqs_val, x_statics_val, y_val = \
                    evaluate(x_seqs, x_statics, y, mode, "deviant", data_set, hpos, hpo, static_features, seed)

    elif data_set == "hospital":
        for seed in [15]:  # [15, 37, 98, 137, 245]:
            for mode in ['rf']:  # 'pwn', 'lr', 'dt', 'knn', 'nb', 'xgb', 'rf', 'mlps_sln', 'lstm'
                procedure = mode

                np.random.seed(seed=seed)
                torch.manual_seed(seed=seed)

                x_seqs, x_statics, y, x_time_vals_final, seq_features, static_features = data.get_hospital_data(max_len, min_len)

                x_seqs_train, x_statics_train, y_train, x_seqs_val, x_statics_val, y_val = \
                    evaluate(x_seqs, x_statics, y, mode, "deviant", data_set, hpos, hpo, static_features, seed)

    else:
        print("Data set not available!")
