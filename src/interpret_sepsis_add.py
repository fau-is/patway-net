import torch
import os
import matplotlib.pyplot as plt
import src.data as data
from src.main import time_step_blow_up
import numpy as np

repetition = 0
model = torch.load(os.path.join("../model", f"model_{repetition}"))
interactions_seq = model.get_number_interactions_seq()
number_interactions_seq = len(interactions_seq)

x_seqs, x_statics, y, x_time_vals_final, seq_features, static_features = data.get_sepsis_data('Admission IC', 30, 3)
x_seqs_final, x_statics_final, y_final = time_step_blow_up(x_seqs, x_statics, y, 30)


# (1) Sequential features (2 time steps, without history)
def delta(y2, y1):
    return y2 - y1


# Print seq features (t x to t y)
t_x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
t_y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

for t in range(0, 11):  # num of transmissions
    for idx, feature in enumerate(seq_features):

        x_x, out_x, _, _ = model.plot_feat_seq_effect(idx, torch.from_numpy(
            x_seqs_final[:, t_x[t], idx].reshape(-1, 1, 1)).float())
        x_x = x_x.detach().numpy().squeeze()
        out_x = out_x.detach().numpy()

        x_y, out_y, _, _ = model.plot_feat_seq_effect(idx, torch.from_numpy(
            x_seqs_final[:, t_y[t], idx].reshape(-1, 1, 1)).float())
        x_y = x_y.detach().numpy().squeeze()
        out_y = out_y.detach().numpy()

        z = delta(out_y.squeeze(), out_x.squeeze())

        data = np.column_stack([x_x, x_y, z])

        plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap='viridis')
        plt.colorbar(label='$\Delta$ Feature effect')
        if feature == 'CRP' and t == 0:
            plt.clim(0, 0.175)
        elif feature == 'CRP' and t > 0:
            plt.clim(-0.03, 0.035)
        elif feature == 'LacticAcid':
            plt.clim(-0.03, 0.03)
        elif feature == 'IVL':
            plt.clim(-0.2, 0.2)
        elif feature == 'IVA':
            plt.clim(-0.2, 0.2)
        elif feature == 'Start':
            plt.clim(-0.03, 0.03)
        else:
            plt.clim(-0.5, 0.5)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.plot([-0.5, 1.5], [-0.5, 1.5], color='grey', linewidth=0.6)
        plt.xlabel("Feature value $t_{%s}$" % str(t_x[t] + 1))
        plt.ylabel("Feature value $t_{%s}$" % str(t_y[t] + 1))
        plt.title(f"Sequential feature: {seq_features[idx]}")
        fig1 = plt.gcf()
        plt.show()
        plt.draw()
        fig1.savefig(f'../plots/{feature}_{t_x[t] + 1}-{t_y[t] + 1}.png', dpi=100)
        plt.close(fig1)

"""
# (2) Print static features (global)
for idx, value in enumerate(static_features):
    # x, out = model.plot_feat_stat_effect_custom(idx, 0, 1)
    x, out = model.plot_feat_stat_effect(idx, torch.from_numpy(x_statics_final[:, idx].reshape(-1, 1)).float())
    x = x.detach().numpy().squeeze()
    out = out.detach().numpy()
    plt.scatter(x, out, color='steelblue')
    if value == "Age" or value == "BMI":
        plt.scatter(x, out, color='steelblue')
        plt.ylim(0.19, 0.41)
    elif value == "Gender" or value == "Foreigner":
        a, b = zip(set(x), set(np.squeeze(out)))
        x = [list(a)[0], list(b)[0]]
        out = [list(a)[1], list(b)[1]]
        plt.bar(x, out, color='steelblue')
        plt.xticks(x, x)
    else:
        plt.plot(x, out, color='steelblue')
    plt.xlabel("Feature value")
    plt.ylabel("Feature effect on model output")
    plt.title(f"Static feature: {static_features[idx]}")
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(f'../plots/{value}.png', dpi=100)
    plt.close(fig1)

# (3) Print sequential feature over time with value range (global)
for t in range(0, 12):
    for idx, value in enumerate(seq_features):
        if value == "CRP":
            # x, out = model.plot_feat_seq_effect_custom(idx, -2, 2)
            x, out, h_t, out_coef = model.plot_feat_seq_effect(idx, torch.from_numpy(
                x_seqs_final[:, t, idx].reshape(-1, 1, 1)).float())
            x = x.detach().numpy().squeeze()
            out = out.detach().numpy()

            if value == "CRP" or value == "LacticAcid" or value == "Start":
                plt.scatter(x, out, color='steelblue')  # scatter plot
            elif value == "IVA" or value == "IVL":
                # todo: check bar plot
                a, b = zip(set(x), set(np.squeeze(out)))
                x = [list(a)[0], list(b)[0]]
                out = [list(a)[1], list(b)[1]]
                plt.bar(x, out, color='steelblue')
                plt.xticks(x, x)
            else:
                plt.plot(x, out, color='steelblue')

            # plt.xlim(-0.02, 1.02)
            # plt.ylim(-0.05, 0.23)
            plt.xlabel("Feature value")
            plt.ylabel("Feature effect on model output")
            plt.title("Sequential feature: %s ($t_{%s}$)" % (str(seq_features[idx]), str(t + 1)))
            fig1 = plt.gcf()
            plt.show()
            plt.draw()
            fig1.savefig(f'../plots/{value}_t{t + 1}.png', dpi=100)
            plt.close(fig1)

# (4) Print sequential features (local, no history)
effect_feature_values = []
case = 459
colors = ['olivedrab', 'lightskyblue', 'steelblue', 'crimson', 'orange']
plt.gca().set_prop_cycle(color=colors)

for idx, value in enumerate(seq_features):
    effect_feature_values.append([])
    for t in range(0, 12):
        x, out, h_t, out_coef = model.plot_feat_seq_effect(idx, torch.from_numpy(
            x_seqs_final[case, t, idx].reshape(1, 1, 1)).float())
        x = x.detach().numpy().squeeze()
        out = out.detach().numpy()
        effect_feature_values[-1].append(out[0][0])

    plt.ylim(-0.11, 0.21)
    plt.plot(list(range(1, 13)), effect_feature_values[idx], label=value, linestyle='dashed', marker='o', markersize=4)
plt.axhline(y=0, color='grey', linewidth=0.6)
plt.xlabel("Time step")
plt.ylabel("Feature effect on model output")
plt.title(f"Feature effect over time of patient pathway {case}")
fig1 = plt.gcf()
plt.legend(loc='upper right', title='Sequential feature')  # adjust based on plot
plt.xticks(np.arange(1, 13, 1))
plt.show()
plt.draw()
fig1.savefig(f'../plots/seq_features_case_{case}.png', dpi=100)
plt.close(fig1)


# Print seq interaction features (first time step
t = 7
if number_interactions_seq > 0:
    for idx in range(0, number_interactions_seq):

        a = torch.from_numpy(x_seqs_final[:, t, interactions_seq[idx][0]].reshape(-1, 1, 1))
        b = torch.from_numpy(x_seqs_final[:, t, interactions_seq[idx][1]].reshape(-1, 1, 1))
        x = torch.cat((a, b), dim=2)

        X_seq, out = model.plot_feat_seq_effect_inter(idx, x)
        X_seq = X_seq.detach().numpy().squeeze()
        out = out.detach().numpy()

        max_size = int(np.sqrt(len(X_seq))) ** 2
        out = out[0:max_size]
        # a_vals = len(set(X_seq[:,0]))
        # b_vals = len(set(X_seq[:,1]))
        # im = plt.imshow(out.reshape(int(np.sqrt(len(X_seq))), int(np.sqrt(len(X_seq)))).transpose())
        # todo:
        im = plt.imshow(out.reshape(int(np.sqrt(len(X_seq))), int(np.sqrt(len(X_seq)))).transpose())
                        # vmin=0, vmax=1)
        cbar = plt.colorbar(im)
        # cbar.set_label("")
        plt.title(f"Interaction:{seq_features[interactions_seq[idx][0]]} x {seq_features[interactions_seq[idx][1]]}")
        plt.xlabel(f"{seq_features[interactions_seq[idx][0]]}")
        plt.ylabel(f"{seq_features[interactions_seq[idx][1]]}")
        plt.show()
"""
