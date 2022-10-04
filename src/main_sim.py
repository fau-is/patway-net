import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import os
from src.data import get_sim_data
import torch
from src.interpret_LSTM import Net
from sklearn.model_selection import KFold

max_len = 12
val_size = 0.2
train_size = 0.8
hpo = True


def concatenate_tensor_matrix(x_seq, x_stat):
    x_train_seq_ = x_seq.reshape(-1, x_seq.shape[1] * x_seq.shape[2])
    x_concat = np.concatenate((x_train_seq_, x_stat), axis=1)

    return x_concat


def train_lasso(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo, data_set):
    if hpo:
        best_model = ""
        best_hpos = ""
        mses = []

        for alpha in hpos["lasso"]["alpha"]:

            model = Lasso(alpha=alpha)
            model.fit(x_train_stat, np.ravel(y_train))
            preds = model.predict(x_val_stat)

            try:
                mse = metrics.mean_squared_error(y_true=y_val, y_pred=preds)
                if np.isnan(mse):
                    mse = 0
            except:
                mse = 0
            mses.append(mse)

            if mse <= min(mses):
                best_model = model
                best_hpos = {"alpha": alpha}

        f = open(f'../output/{data_set}_{mode}_hpos_{seed}.txt', 'a+')
        f.write(str(best_hpos))
        f.write("Validation MSEs," + ",".join([str(x) for x in mses]) + '\n')
        f.write(f'Avg,{sum(mses) / len(mses)}\n')
        f.write(f'Std,{np.std(mses, ddof=1)}\n')
        f.close()

        return best_model, best_hpos

    else:
        model = Lasso()
        model.fit(x_train_stat, np.ravel(y_train))

        return model


def train_ridge(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo, data_set):
    if hpo:
        best_model = ""
        best_hpos = ""
        mses = []

        for alpha in hpos["ridge"]["alpha"]:

            model = Ridge(alpha=alpha)
            model.fit(x_train_stat, np.ravel(y_train))
            preds = model.predict(x_val_stat)

            try:
                mse = metrics.mean_squared_error(y_true=y_val, y_pred=preds)
                if np.isnan(mse):
                    mse = 0
            except:
                mse = 0
            mses.append(mse)

            if mse <= min(mses):
                best_model = model
                best_hpos = {"alpha": alpha}

        f = open(f'../output/{data_set}_{mode}_hpos_{seed}.txt', 'a+')
        f.write(str(best_hpos))
        f.write("Validation MSEs," + ",".join([str(x) for x in mses]) + '\n')
        f.write(f'Avg,{sum(mses) / len(mses)}\n')
        f.write(f'Std,{np.std(mses, ddof=1)}\n')
        f.close()

        return best_model, best_hpos

    else:
        model = Ridge()
        model.fit(x_train_stat, np.ravel(y_train))

        return model


def train_dt(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo, data_set):
    if hpo:
        best_model = ""
        best_hpos = ""
        mses = []

        for max_depth in hpos["dt"]["max_depth"]:
            for min_samples_split in hpos["dt"]["min_samples_split"]:

                model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
                model.fit(x_train_stat, np.ravel(y_train))
                preds = model.predict(x_val_stat)
                try:
                    mse = metrics.mean_squared_error(y_true=y_val, y_pred=preds)
                    if np.isnan(mse):
                        mse = 0
                except:
                    mse = 0

                mses.append(mse)

                if mse <= min(mses):
                    best_model = model
                    best_hpos = {"max_depth": max_depth, "min_samples_split": min_samples_split}

        f = open(f'../output/{data_set}_{mode}_hpos_{seed}.txt', 'a+')
        f.write(str(best_hpos) + '\n')
        f.write("Validation MSEs," + ",".join([str(x) for x in mses]) + '\n')
        f.write(f'Avg,{sum(mses) / len(mses)}\n')
        f.write(f'Std,{np.std(mses, ddof=1)}\n')
        f.close()

        return best_model, best_hpos

    else:
        model = DecisionTreeRegressor()
        model.fit(x_train_stat, np.ravel(y_train))

        return model


def train_knn(x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, hpos, hpo, data_set):
    if hpo:
        best_model = ""
        best_hpos = ""
        mses = []

        for n_neighbors in hpos["knn"]["n_neighbors"]:
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
            model.fit(x_train_stat, np.ravel(y_train))
            preds = model.predict(x_val_stat)
            try:
                mse = metrics.mean_squared_error(y_true=y_val, y_pred=preds)
                if np.isnan(mse):
                    mse = 0
            except:
                mse = 0
            mses.append(mse)

            if mse <= min(mses):
                best_model = model
                best_hpos = {"n_eighbors": n_neighbors}

        f = open(f'../output/{data_set}_{mode}_hpos_{seed}.txt', 'a+')
        f.write(str(best_hpos) + '\n')
        f.write("Validation MSEs," + ",".join([str(x) for x in mses]) + '\n')
        f.write(f'Avg,{sum(mses) / len(mses)}\n')
        f.write(f'Std,{np.std(mses, ddof=1)}\n')
        f.close()

        return best_model, best_hpos

    else:
        model = KNeighborsRegressor()
        model.fit(x_train_stat, np.ravel(y_train))

        return model


def train_lstm(x_train_seq, x_train_stat, y_train, x_val_seq=False, x_val_stat=False, y_val=False, hpos=False,
               hpo=False, mode="pwn", data_set="sepsis"):
    max_case_len = x_train_seq.shape[1]
    num_features_seq = x_train_seq.shape[2]
    num_features_stat = x_train_stat.shape[1]

    if mode == "pwn":
        import torch
        import torch.nn as nn
        import torch.optim as optim

        if hpo:
            best_model = ""
            best_hpos = ""
            mses = []

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
                                            interactions_seq_itr=100,
                                            interactions_seq_best=inter_seq_best,
                                            interactions_seq_auto=False,
                                            input_sz_stat=num_features_stat,
                                            output_sz=1,
                                            only_static=False,
                                            masking=True,
                                            mlp_hidden_size=stat_feature_sz,
                                            x_seq=x_train_seq,
                                            x_stat=x_train_stat,
                                            y=y_train)

                                criterion = nn.MSELoss()
                                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                                idx = np.arange(len(x_train_seq))

                                import copy
                                best_val_loss = np.inf
                                patience = 2
                                epochs = 100
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
                                        out = model(x_train_seq[i * batch_size:(i + 1) * batch_size].float(),
                                                    x_train_stat[i * batch_size:(i + 1) * batch_size].float())
                                        loss = criterion(out, y_train[i * batch_size:(i + 1) * batch_size].float().reshape(-1,1))
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
                                                out = model(x_val_seq[i * batch_size: (i + 1) * batch_size].float(),
                                                            x_val_stat[i * batch_size: (i + 1) * batch_size].float())
                                                loss = loss_function(out, y_val[i * batch_size:(i + 1) * batch_size].float().reshape(-1,1))
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

                                    # torch.sigmoid(
                                    preds = model_best_es(x_val_seq_.float(), x_val_stat_.float())
                                    try:
                                        mse = metrics.mean_squared_error(y_true=y_val, y_pred=preds)
                                        if np.isnan(mse):
                                            mse = 0
                                    except:
                                        mse = 0
                                    mses.append(mse)

                                    if mse <= min(mses):
                                        best_model = copy.deepcopy(model_best_es)
                                        best_hpos = {"learning_rate": learning_rate, "batch_size": batch_size,
                                                     "seq_feature_sz": seq_feature_sz,
                                                     "stat_feature_sz": stat_feature_sz,
                                                     "inter_seq_best": inter_seq_best}

            f = open(f'../output/{data_set}_{mode}_hpos_{seed}.txt', 'a+')
            f.write(str(best_hpos) + '\n')
            f.write("Validation MSEs," + ",".join([str(x) for x in mses]) + '\n')
            f.write(f'Avg,{sum(mses) / len(mses)}\n')
            f.write(f'Std,{np.std(mses, ddof=1)}\n')
            f.close()

            return best_model, best_hpos
        else:
            pass


def create_np_arrays(X_seqs, X_statics, y, max_len):

    X_seqs_final = np.zeros((len(X_seqs), max_len, len(X_seqs[0][0])))
    X_statics_final = np.zeros((len(X_seqs), len(X_statics[0])))

    for i, x in enumerate(X_seqs):
        X_seqs_final[i, :len(x), :] = np.array(x)
        X_statics_final[i, :] = np.array(x_statics[i])
    y_final = np.array(y)

    return X_seqs_final, X_statics_final, y_final


def evaluate_on_cut(x_seqs, x_statics, y, mode, data_set, hpos, hpo, static_features, seed):
    k = 5
    results = {}
    id = -1

    skfold = KFold(n_splits=k, shuffle=True, random_state=seed)
    for train_index_, test_index in skfold.split(X=x_statics):

        id += 1
        train_index = train_index_[0: int(len(train_index_) * (1 - val_size))]
        val_index = train_index_[int(len(train_index_) * (1 - val_size)):]

        X_train_seq, X_train_stat, y_train = create_np_arrays(
            [x_seqs[x] for x in train_index],
            [x_statics[x] for x in train_index],
            [y[x] for x in train_index], max_len)

        X_val_seq, X_val_stat, y_val = create_np_arrays(
            [x_seqs[x] for x in val_index],
            [x_statics[x] for x in val_index],
            [y[x] for x in val_index], max_len)

        X_test_seq, X_test_stat, y_test = create_np_arrays(
            [x_seqs[x] for x in test_index],
            [x_statics[x] for x in test_index],
            [y[x] for x in test_index], max_len)

        if mode == "pwn":
            model, best_hpos = train_lstm(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                          y_val.reshape(-1, 1), hpos, hpo, mode, data_set)

            X_train_seq = torch.from_numpy(X_train_seq)
            X_train_stat = torch.from_numpy(X_train_stat)

            X_val_seq = torch.from_numpy(X_val_seq)
            X_val_stat = torch.from_numpy(X_val_stat)

            X_test_seq = torch.from_numpy(X_test_seq)
            X_test_stat = torch.from_numpy(X_test_stat)

            model.eval()
            with torch.no_grad():
                # torch.sigmoid(
                results['preds_train'] = model(X_train_seq.float(), X_train_stat.float())
                results['preds_val'] = model(X_val_seq.float(), X_val_stat.float())
                results['preds_test'] = model(X_test_seq.float(), X_test_stat.float())

        elif mode == "lasso":
            model, best_hpos = train_lasso(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                           y_val.reshape(-1, 1), hpos, hpo, data_set)

        elif mode == "ridge":
            model, best_hpos = train_ridge(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                        y_val.reshape(-1, 1), hpos, hpo, data_set)

        elif mode == "dt":
            model, best_hpos = train_dt(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                        y_val.reshape(-1, 1), hpos, hpo, data_set)

        elif mode == "knn":
            model, best_hpos = train_knn(X_train_seq, X_train_stat, y_train.reshape(-1, 1), X_val_seq, X_val_stat,
                                         y_val.reshape(-1, 1), hpos, hpo, data_set)

        if mode in ["dt", "knn", "ridge", "lasso"]:
            results['preds_train'] = model.predict(X_train_stat)
            results['preds_val'] = model.predict(X_val_stat)
            results['preds_test'] = model.predict(X_test_stat)

        results['gts_train'] = [int(y) for y in y_train]
        results['gts_val'] = [int(y) for y in y_val]
        results['gts_test'] = [int(y) for y in y_test]

        if mode == 'pwn':
            torch.save(model, os.path.join("../model", f"model_{id}_{seed}"))

        results_temp_train = pd.DataFrame(list(zip(results['preds_train'], results['gts_train'])), columns=['preds', 'gts'])
        results_temp_val = pd.DataFrame(list(zip(results['preds_val'], results['gts_val'])), columns=['preds', 'gts'])
        results_temp_test = pd.DataFrame(list(zip(results['preds_test'], results['gts_test'])), columns=['preds', 'gts'])

        if id == 0:
            results['mse_train'] = list()
            results['mse_val'] = list()
            results['mse_test'] = list()

        results['mse_train'].append(metrics.mean_squared_error(y_true=results_temp_train['gts'], y_pred=results_temp_train['preds']))
        results['mse_val'].append(metrics.mean_squared_error(y_true=results_temp_val['gts'], y_pred=results_temp_val['preds']))
        results['mse_test'].append(metrics.mean_squared_error(y_true=results_temp_test['gts'], y_pred=results_temp_test['preds']))

        metrics_ = ["mse_train", "mse_val", "mse_test"]

        for metric_ in metrics_:
            vals = []
            try:
                f = open(f'../output/{data_set}_{mode}_{seed}.txt', "a+")
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

    data_sets = ["sim"]

    hpos = {
        "pwn": {"seq_feature_sz": [4, 8], "stat_feature_sz": [4, 8], "learning_rate": [0.01, 0.001],
                "batch_size": [32, 128], "inter_seq_best": [1]},
        # "pwn": {"seq_feature_sz": [4, 8], "stat_feature_sz": [4, 8], "learning_rate": [0.001, 0.01], "batch_size": [32, 128], "inter_seq_best": [1]},
        "lasso": {"alpha": [pow(10, -3), pow(10, -2), pow(10, -1), pow(10, 0), pow(10, 1), pow(10, 2), pow(10, 3)]},
        "ridge": {"alpha": [pow(10, -3), pow(10, -2), pow(10, -1), pow(10, 0), pow(10, 1), pow(10, 2), pow(10, 3)]},
        "dt": {"max_depth": [2, 3, 4], "min_samples_split": [2]},
        "knn": {"n_neighbors": [3, 5, 10]}
    }

    data_set = "sim"

    for seed in [15]: # 37, 98, 137, 245]:
        for mode in ['pwn']:  # 'pwn', 'lasso', 'ridge', 'dt', 'knn'
            np.random.seed(seed=seed)
            torch.manual_seed(seed=seed)

            x_seqs, x_statics, y, _, seq_features, static_features = get_sim_data('Label', 'Simulation_data_1000.csv')

            x_seqs_train, x_statics_train, y_train, x_seqs_val, x_statics_val, y_val = \
                evaluate_on_cut(x_seqs, x_statics, y, mode, data_set, hpos, hpo, static_features, seed)
