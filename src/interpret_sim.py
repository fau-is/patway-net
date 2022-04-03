from src.data import get_sim_data
from src.interpret_LSTM import Net
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt

model = torch.load(os.path.join("../model", f"model_sim"))

x_seqs, x_statics, y, _, seq_features, static_features = get_sim_data('Label')
x_seq_final = np.zeros((len(x_seqs), 12, len(x_seqs[0][0])))
x_stat_final = np.zeros((len(x_seqs), len(x_statics[0])))
for i, x in enumerate(x_seqs):
    x_seq_final[i, :len(x), :] = np.array(x)
    x_stat_final[i, :] = np.array(x_statics[i])
y_final = np.array(y).astype(np.int32)

# x_seq_final = torch.from_numpy(x_seq_final)
# x_stat_final = torch.from_numpy(x_stat_final)
# y_final = torch.from_numpy(y_final).reshape(-1)

"""
# Print seq features (first time step)
t = 1
for idx in range(0, len(seq_features)):
    # x, out = model.plot_feat_seq_effect_custom(idx, -2, 2)
    x, out = model.plot_feat_seq_effect(idx, torch.from_numpy(x_seq_final[:, t, idx].reshape(-1, 1, 1)).float())
    x = x.detach().numpy().squeeze()
    out = out.detach().numpy()
    plt.scatter(x, out)
    # plt.plot(x, out)
    plt.xlabel("Feature value")
    plt.ylabel("Feature effect on model output")
    plt.title(f"Sequential feature:{seq_features[idx]}")
    plt.show()
    print(0)
"""

# Print stat features
for idx, value in enumerate(static_features):
    x, out = model.plot_feat_stat_effect(idx, torch.from_numpy(x_stat_final[:, idx].reshape(-1, 1)).float())
    x = x.detach().numpy().squeeze()
    out = out.detach().numpy()
    if value == "Age" or value == "BMI" or value == "Gender" or value == "Foreigner":
        plt.scatter(x, out)  # scatter plot
    else:
        plt.plot(x, out)  # line plot
    plt.xlabel("Feature value")
    plt.ylabel("Feature effect on model output")
    plt.title(f"Static feature:{static_features[idx]}")
    plt.show()