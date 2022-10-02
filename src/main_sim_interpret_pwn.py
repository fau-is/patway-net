from src.data import get_sim_data
from src.interpret_LSTM import Net
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import os
import copy
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

seeds = [137] # 15, 37, 98, 137, 245]
results = {'mse_train': list(), 'mae_train': list(), 'r2_train': list()}
epochs = 1000  # 1000
batch_size = 32
lr = 0.001 # 0.001
patience = 100 # 100

for seed in seeds:
    torch.manual_seed(seed=seed)

    x_seqs, x_statics, y, _, seq_features, static_features = get_sim_data('Label', 'Simulation_data_1000.csv')

    x_seqs_final = np.zeros((len(x_seqs), 12, len(x_seqs[0][0])))
    x_statics_final = np.zeros((len(x_seqs), len(x_statics[0])))
    for i, x in enumerate(x_seqs):
        x_seqs_final[i, :len(x), :] = np.array(x)
        x_statics_final[i, :] = np.array(x_statics[i])
    y_final = np.array(y)

    x_seq_final = torch.from_numpy(x_seqs_final)
    x_stat_final = torch.from_numpy(x_statics_final)
    y_final = torch.from_numpy(y_final).reshape(-1)

    last_loss_all = np.inf
    trigger_times = 0

    model = Net(input_sz_seq=len(seq_features),
                hidden_per_seq_feat_sz=16,  # 16
                interactions_seq=[],
                interactions_seq_itr=100,
                interactions_seq_best=1,
                interactions_seq_auto=False,
                input_sz_stat=len(static_features),
                output_sz=1,
                masking=False,
                mlp_hidden_size=16,  # 16
                only_static=False,
                x_seq=x_seq_final,
                x_stat=x_stat_final,
                y=y_final)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    idx = np.arange(x_seq_final.shape[0])
    model_best_es = copy.deepcopy(model)

    for epoch in range(epochs):

        # np.random.shuffle(idx)
        # x_seq_final = x_seq_final[idx]
        # x_stat_final = x_stat_final[idx]
        # y_final = y_final[idx]

        loss_all = 0
        num_batches = x_seq_final.shape[0] // batch_size

        for i in range(num_batches):

            optimizer.zero_grad()
            out = model(x_seq_final[i * batch_size:(i + 1) * batch_size].float(),
                        x_stat_final[i * batch_size:(i + 1) * batch_size].float())
            loss = criterion(out, y_final[i * batch_size:(i + 1) * batch_size].float().reshape(-1, 1))
            loss.backward()
            optimizer.step()
            loss_all += float(loss)

        if loss_all >= last_loss_all:
            trigger_times += 1
            if trigger_times >= patience:
                break
        else:
            last_loss_all = loss_all
            trigger_times = 0
            model_best_es = copy.deepcopy(model)

        print(f'Epoch {epoch + 1}: {loss_all / num_batches}')

    torch.save(model_best_es, os.path.join("../model", f"model_sim"))

    model_best_es.eval()
    with torch.no_grad():
        y_pred = model_best_es(x_seq_final.float(), x_stat_final.float())
        # torch.sigmoid(

    results["mse_train"].append(mean_squared_error(y_true=y, y_pred=y_pred))
    results["mae_train"].append(mean_absolute_error(y_true=y, y_pred=y_pred))
    results["r2_train"].append(r2_score(y_true=y, y_pred=y_pred))

print(f'Train mse -- avg: {sum(results["mse_train"]) / len(results["mse_train"])} --sd: {np.std(results["mse_train"])} -- values: {results["mse_train"]}')
print(f'Train mae -- avg: {sum(results["mae_train"]) / len(results["mae_train"])} --sd: {np.std(results["mae_train"])} -- values: {results["mae_train"]}')
print(f'Train r2 -- avg: {sum(results["r2_train"]) / len(results["r2_train"])} --sd: {np.std(results["r2_train"])} -- values: {results["r2_train"]}')