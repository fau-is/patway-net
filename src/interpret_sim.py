from src.data import get_sim_data
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as matplotlib

model = torch.load(os.path.join("../model", f"model_sim"))

x_seqs, x_statics, y, _, seq_features, static_features = get_sim_data('Label', 'Simulation_data_1k.csv')

# Create dataset without prefixes
x_seq_final = np.zeros((len(x_seqs), 12, len(x_seqs[0][0])))
x_stat_final = np.zeros((len(x_seqs), len(x_statics[0])))
for i, x in enumerate(x_seqs):
    x_seq_final[i, :len(x), :] = np.array(x)
    x_stat_final[i, :] = np.array(x_statics[i])


# (1) Sequential features (2 time steps, without history)
def delta(y2, y1):
    return y2 - y1

def slope(x1, y1, x2, y2):
    eps = 0.000000000000000000000000000000001
    return (y2 - y1) / ((x2 - x1) + eps)

# Print seq features (t x to t y)
t_x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
t_y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

for t in range(0, 11):  # num of transmissions
    for idx, feature in enumerate(seq_features):


        x_x, out_x, _, _ = model.plot_feat_seq_effect(idx, torch.from_numpy(x_seq_final[:, t_x[t], idx].reshape(-1, 1, 1)).float())
        x_x = x_x.detach().numpy().squeeze()
        out_x = out_x.detach().numpy()

        x_y, out_y, _, _ = model.plot_feat_seq_effect(idx, torch.from_numpy(x_seq_final[:, t_y[t], idx].reshape(-1, 1, 1)).float())
        x_y = x_y.detach().numpy().squeeze()
        out_y = out_y.detach().numpy()

        z = delta(out_y.squeeze(), out_x.squeeze())
        # if sum(z) > 0:
            # print(f"Feature {feature} --- t_x ({t_x[t]}) to t_y ({t_y[t]}) --- found something!")

        data = np.column_stack([x_x, x_y, z])

        plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap='viridis')
        plt.colorbar(label='$\Delta$ Feature effect')
        if feature == 'CRP' and t==0:
            plt.clim(0, 0.175)
        elif feature == 'CRP' and t>0:
            plt.clim(-0.03, 0.035)
        elif feature == 'LacticAcid':
            plt.clim(-0.03, 0.03)
        elif feature == 'IVL':
            plt.clim(-0.2, 0.2)
        elif feature == 'IVA':
            plt.clim (-0.2, 0.2)
        elif feature == 'Start':
            plt.clim(-0.03, 0.03)
        else:
            plt.clim(-0.5,0.5)

        plt.xlabel("Feature value $t_{%s}$" % str(t_x[t]+1))
        plt.ylabel("Feature value $t_{%s}$" % str(t_y[t]+1))
        plt.title(f"Sequential feature: {seq_features[idx]}")
        fig1 = plt.gcf()
        plt.show()
        plt.draw()
        fig1.savefig(f'../plots/{feature}_{t_x[t]+1}-{t_y[t]+1}.png', dpi=100)
        plt.close(fig1)


# (2) Print static features (global)
for idx, value in enumerate(static_features):
    # x, out = model.plot_feat_stat_effect_custom(idx, 0, 1)
    x, out = model.plot_feat_stat_effect(idx, torch.from_numpy(x_stat_final[:, idx].reshape(-1, 1)).float())
    x = x.detach().numpy().squeeze()
    out = out.detach().numpy()
    if value == "Age" or value == "BMI":
        plt.scatter(x, out, color = 'steelblue')
        plt.ylim(0.19, 0.41)
    elif value == "Gender" or value == "Foreigner":
        a, b = zip(set(x), set(np.squeeze(out)))
        x = [list(a)[0], list(b)[0]]
        out = [list(a)[1], list(b)[1]]
        plt.bar(x, out, color = 'steelblue')
        plt.xticks(x, x)
    else:
        plt.plot(x, out, color = 'steelblue')
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
            x, out, h_t, out_coef = model.plot_feat_seq_effect(idx, torch.from_numpy(x_seq_final[:, t, idx].reshape(-1, 1, 1)).float())
            x = x.detach().numpy().squeeze()
            out = out.detach().numpy()

            if value == "CRP" or value == "LacticAcid" or value == "Start":
                plt.scatter(x, out, color = 'steelblue')  # scatter plot
            elif value == "IVA" or value == "IVL":
                # todo: check bar plot
                a, b = zip(set(x), set(np.squeeze(out)))
                x = [list(a)[0], list(b)[0]]
                out = [list(a)[1], list(b)[1]]
                plt.bar(x, out, color = 'steelblue')
                plt.xticks(x, x)
            else:
                plt.plot(x, out, color = 'steelblue')

            plt.xlim(-0.02, 1.02)
            plt.ylim(-0.05, 0.23)
            plt.xlabel("Feature value")
            plt.ylabel("Feature effect on model output")
            plt.title("Sequential feature: %s ($t_{%s}$)" % (str(seq_features[idx]), str(t+1)))
            fig1 = plt.gcf()
            plt.show()
            plt.draw()
            fig1.savefig(f'../plots/{value}_t{t+1}.png', dpi=100)
            plt.close(fig1)

# (4) Print sequential features (local, no history)
effect_feature_values = []
case = 459
colors = ['olivedrab', 'lightskyblue', 'steelblue', 'crimson', 'orange']
plt.gca().set_prop_cycle(color=colors)


for idx, value in enumerate(seq_features):
    effect_feature_values.append([])
    for t in range(0, 12):
        x, out, h_t, out_coef = model.plot_feat_seq_effect(idx, torch.from_numpy(x_seq_final[case, t, idx].reshape(1, 1, 1)).float())
        x = x.detach().numpy().squeeze()
        out = out.detach().numpy()
        effect_feature_values[-1].append(out[0][0])

    plt.ylim(-0.11, 0.21)
    plt.plot(list(range(1, 13)), effect_feature_values[idx], label = value, linestyle = 'dashed', marker = 'o', markersize = 4)
plt.xlabel("Time step")
plt.ylabel("Feature effect on model output")
plt.title(f"Feature effect over time of patient pathway {case}")
fig1 = plt.gcf()
plt.legend(loc='upper right', title = 'Sequential feature') #adjust based on plot
plt.xticks(np.arange(1, 13, 1))
plt.show()
plt.draw()
fig1.savefig(f'../plots/seq_features_case_{case}.png', dpi=100)
plt.close(fig1)



