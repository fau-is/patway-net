from src.data import get_sim_data
from src.interpret_LSTM import Net
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
import seaborn as sns
import pandas as pd
from src.main import time_step_blow_up

model = torch.load(os.path.join("../model", f"model_sim"))

x_seqs, x_statics, y, _, seq_features, static_features = get_sim_data('Label', 'Simulation_data_1k_test4.csv')

# Create dataset with prefixes
# x_seq_final, x_stat_final, y_final = time_step_blow_up(x_seqs, x_statics, y, 12)


# Create dataset without prefixes
x_seq_final = np.zeros((len(x_seqs), 12, len(x_seqs[0][0])))
x_stat_final = np.zeros((len(x_seqs), len(x_statics[0])))
for i, x in enumerate(x_seqs):
    x_seq_final[i, :len(x), :] = np.array(x)
    x_stat_final[i, :] = np.array(x_statics[i])


""""
def slopee(x1,y1,x2,y2):
    eps = 0.01
    # x2 = 2; x1 = 1 --> +1 (fine)
    # x2 = 2; x1 = 2 --> 0 (break)

    x = (y2 - y1) / ((x2 - x1) + 1.0 + eps)
    return x

# seq_features = ["CRP"]
# Print seq features (t x to t y)
t_x = [1, 1, 5]
t_y = [5, 10, 10]

import seaborn as sns
import pandas as pd

for t in range(0, len(t_x)):
    for idx, value in enumerate(seq_features):
        x_x, out_x = model.plot_feat_seq_effect(idx, torch.from_numpy(x_seq_final[:, t_x[t], idx].reshape(-1, 1, 1)).float())
        x_x = x_x.detach().numpy().squeeze()
        out_x = out_x.detach().numpy()

        x_y, out_y = model.plot_feat_seq_effect(idx, torch.from_numpy(x_seq_final[:, t_y[t], idx].reshape(-1, 1, 1)).float())
        x_y = x_y.detach().numpy().squeeze()
        out_y = out_y.detach().numpy()

        z = slopee(x_x, out_x.squeeze(), x_y, out_y.squeeze())
        if sum(z) > 0:
            print(f"Feature {value} --- t_x ({t_x[t]}) to t_y ({t_y[t]}) --- found something!")

            data = np.column_stack([x_x, x_y, z])
            # data = pd.DataFrame({'x_x': data[:, 0], 'x_y': data[:, 1], 'z': data[:, 2]})

            plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap='magma', )
            plt.colorbar()
            plt.xlabel(f"Feature value t {t_x[t]}")
            plt.ylabel(f"Feature value t {t_y[t]}")
            plt.title(f"Sequential feature:{seq_features[idx]}")
            fig1 = plt.gcf()
            plt.show()
            plt.draw()
            # fig1.savefig(f'../plots/{value}.png', dpi=100)

            # sns.kdeplot(data["x_x"], data["x_y"])

# Print sequential features (local)
effect_feature_values = []

for t in range(0, 12):
    for idx, value in enumerate(seq_features):

        x, out, h_t, out_coef = model.plot_feat_seq_effect(idx, torch.from_numpy(x_seq_final[0, t, idx].reshape(1, 1, 1)).float())
        x = x.detach().numpy().squeeze()
        out = out.detach().numpy()
        h_t_list = [torch.mean(h_t[i,:]).item() for i in range (0, h_t.shape[0])]
        h_t_final = sum(h_t_list) / len(h_t_list)
        out_coef_final = torch.mean(out_coef.reshape(-1)).item()
        print(f"Time step {t} --- Feature {value} --- Effect Value {torch.mean(torch.from_numpy(out.squeeze()))} --- Hidden state {h_t_final} --- Coef {out_coef_final}")
"""

"""
# 1) Print sequential features (local)
effect_feature_values = []
case = 2
colors = ['blue', 'green', 'red', 'black', 'magenta']

for idx, value in enumerate(seq_features):
    effect_feature_values.append([])
    for t in range(0, 12):
        x, out, h_t, out_coef = model.plot_feat_seq_effect(idx, torch.from_numpy(
            x_seq_final[case, t, idx].reshape(1, 1, 1)).float())
        x = x.detach().numpy().squeeze()
        out = out.detach().numpy()
        effect_feature_values[-1].append(out[0][0])

    plt.plot(list(range(1, 13)), effect_feature_values[idx], label=value, color=colors[idx])
plt.xlabel("Time step")
plt.ylabel("Feature effect on model output")
plt.title(f"Change of sequential features for case: {case}")
fig1 = plt.gcf()
plt.legend()
plt.xticks(np.arange(1, 13, 1))
plt.show()
plt.draw()
fig1.savefig(f'../plots/seq_features_case_{case}.png', dpi=100)
"""

"""
# 2) Print sequential features (global)
cases = 100
effect_feature_values = []
t_values = []
feature_names = []
for idx, value in enumerate(seq_features):

    for t in range(0, 12):
        x, out, h_t, out_coef = model.plot_feat_seq_effect(idx, torch.from_numpy(x_seq_final[0:cases, t, idx].reshape(-1, 1, 1)).float())
        x = x.detach().numpy().squeeze()
        out = out.detach().numpy()
        effect_feature_values = effect_feature_values + list(out.squeeze())
        t_values = t_values + [t+1] * cases
        feature_names = feature_names + [value] * cases

df = pd.DataFrame(list(zip(effect_feature_values, t_values, feature_names)), columns=['effect', 't', 'feature'])

sns.stripplot(x="t", y="effect", hue="feature", data=df)
plt.xlabel("Time step")
plt.ylabel("Feature effect on model output")
plt.title(f"Change of sequential features for case: {cases}")
fig1 = plt.gcf()
plt.legend()
plt.xticks(np.arange(0, 12, 1))
plt.show()
plt.draw()
fig1.savefig(f'../plots/seq_features_case_{cases}.png', dpi=100)
"""


""""# 3) Print static features (global)
for idx, value in enumerate(static_features):
    # x, out = model.plot_feat_stat_effect_custom(idx, 0, 1)
    x, out = model.plot_feat_stat_effect(idx, torch.from_numpy(x_stat_final[:, idx].reshape(-1, 1)).float())
    x = x.detach().numpy().squeeze()
    out = out.detach().numpy()
    if value == "Age" or value == "BMI":  # or value == "Gender" or value == "Foreigner":
        plt.scatter(x, out)  # scatter plot
    elif value == "Gender" or value == "Foreigner":
        a, b = zip(set(x), set(np.squeeze(out)))
        x = [list(a)[0], list(b)[0]]
        out = [list(a)[1], list(b)[1]]
        plt.bar(x, out)  # bar plot
        plt.xticks(x, x)
    else:
        plt.plot(x, out)  # line plot
    plt.xlabel("Feature value")
    plt.ylabel("Feature effect on model output")
    plt.title(f"Static feature:{static_features[idx]}")
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(f'../plots/{value}.png', dpi=100)
"""

# 4) Print sequential feature over time with value range (global)
for t in range(0, 12):
    for idx, value in enumerate(seq_features):
        if value == "CRP":
            # x, out = model.plot_feat_seq_effect_custom(idx, -2, 2)
            x, out, h_t, out_coef = model.plot_feat_seq_effect(idx, torch.from_numpy(x_seq_final[2, t, idx].reshape(-1, 1, 1)).float())
            x = x.detach().numpy().squeeze()
            out = out.detach().numpy()

            if value == "CRP" or value == "LacticAcid" or value == "Start":
                plt.scatter(x, out)  # scatter plot
            elif value == "IVA" or value == "IVL":
                # todo: check bar plot
                a, b = zip(set(x), set(np.squeeze(out)))
                x = [list(a)[0], list(b)[0]]
                out = [list(a)[1], list(b)[1]]
                plt.bar(x, out)  # bar plot
                plt.xticks(x, x)
            else:
                plt.plot(x, out)  # line plot

            plt.xlabel("Feature value")
            plt.ylabel("Feature effect on model output")
            plt.title(f"Sequential feature:{seq_features[idx]}")
            fig1 = plt.gcf()
            plt.show()
            plt.draw()
            fig1.savefig(f'../plots/{value}_t{t}.png', dpi=100)


"""
# 5) Print sequential feature over time with value range (global, with history)
cases = 1
for t in range(0, 12):
    for idx, value in enumerate(seq_features):
        if value == "CRP":
            # x, out = model.plot_feat_seq_effect_custom(idx, -2, 2)
            x, out, h_t, out_coef = model.plot_feat_seq_effect(idx, torch.from_numpy(x_seq_final[0:cases, 0:t+1, idx].reshape(-1, t+1, 1)).float())

            # get last value
            x = x[:, -1, :]

            x = x.detach().numpy().squeeze()
            out = out.detach().numpy()

            if value == "CRP" or value == "LacticAcid" or value == "Start":
                plt.scatter(x, out)  # scatter plot
            elif value == "IVA" or value == "IVL":
                # todo: check bar plot
                a, b = zip(set(x), set(np.squeeze(out)))
                x = [list(a)[0], list(b)[0]]
                out = [list(a)[1], list(b)[1]]
                plt.bar(x, out)  # bar plot
                plt.xticks(x, x)
            else:
                plt.plot(x, out)  # line plot

            plt.xlabel("Feature value")
            plt.ylabel("Feature effect on model output")
            plt.title(f"Sequential feature:{seq_features[idx]}")
            fig1 = plt.gcf()
            plt.show()
            plt.draw()
            fig1.savefig(f'../plots/{value}_t{t}.png', dpi=100)
"""