from src.data import get_sim_data
from src.interpret_LSTM import Net
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import os
from sklearn.preprocessing import PowerTransformer

x_seqs, x_statics, y, _, seq_features, static_features = get_sim_data('Label', 'Simulation_data_5k.csv')

x_seq_final = np.zeros((len(x_seqs), 12, len(x_seqs[0][0])))
x_stat_final = np.zeros((len(x_seqs), len(x_statics[0])))
for i, x in enumerate(x_seqs):
    x_seq_final[i, :len(x), :] = np.array(x)
    x_stat_final[i, :] = np.array(x_statics[i])
y_final = np.array(y)  # .astype(np.int32)

x_seq_final = torch.from_numpy(x_seq_final)
x_stat_final = torch.from_numpy(x_stat_final)

# pt = PowerTransformer()
# y_final = pt.fit_transform(y_final.reshape(-1, 1))

y_final = torch.from_numpy(y_final).reshape(-1)

epochs = 100
batch_size = 64
lr = 0.001

model = Net(input_sz_seq=len(seq_features),
            hidden_per_seq_feat_sz=16,
            interactions_seq=[],
            interactions_seq_itr=10,
            interactions_seq_best=1,
            interactions_seq_auto=False,
            input_sz_stat=len(static_features),
            output_sz=1,
            masking=True,
            mlp_hidden_size=16,
            x_seq=x_seq_final,
            x_stat=x_stat_final,
            y=y_final)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

idx = np.arange(x_seq_final.shape[0])

for _ in range(epochs):
    np.random.shuffle(idx)

    x_seq_final = x_seq_final[idx]
    x_stat_final = x_stat_final[idx]
    y_final = y_final[idx]

    loss_all = 0
    for i in range(x_seq_final.shape[0] // batch_size):

        optimizer.zero_grad()  # a clean up step for PyTorch
        out = model(x_seq_final[i * batch_size:(i + 1) * batch_size].float(),
                    x_stat_final[i * batch_size:(i + 1) * batch_size].float())
        loss = criterion(out, y_final[i * batch_size:(i + 1) * batch_size].float())
        loss.backward()  # compute updates for each parameter
        optimizer.step()  # make the updates for each parameter
        loss_all += float(loss)

    print(loss_all)

torch.save(model, os.path.join("../model", f"model_sim"))

