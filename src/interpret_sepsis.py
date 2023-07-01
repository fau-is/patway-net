import torch
import os
import matplotlib.pyplot as plt
import src.data as data
from src.main import time_step_blow_up
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

repetition = 0
# map location param is required as the model was trained on gpu
model = torch.load(os.path.join("../model", f"model_{repetition}"), map_location=torch.device('cpu'))
interactions_seq = model.get_number_interactions_seq()
number_interactions_seq = len(interactions_seq)

# Note: static features include binary features created from the diagnosis feature or not
x_seqs, x_statics, y, x_time_vals_final, seq_features, static_features = data.get_sepsis_data('Admission IC', 50, 3)
x_seqs_final, x_statics_final, y_final = time_step_blow_up(x_seqs, x_statics, y, 50)


# (1) Sequential features (2 time steps, without history)
def delta(y2, y1):
    return y2 - y1

t_x = list(range(0, 11))
t_y = list(range(1, 12))

plt.rcParams["figure.figsize"] = (7.5, 5)
for idx, feature in enumerate(seq_features):
    inputs = torch.linspace(-1, 1, 100).reshape(100, 1, 1).float()

    x_x, out_x, _, _ = model.plot_feat_seq_effect(idx, inputs)
    x_x = x_x.detach().numpy().squeeze()
    outputs = out_x.detach().numpy().squeeze()

    diffs = np.empty((len(inputs), len(inputs)))
    for i in range(100):
        for j in range(100):
            output1 = outputs[i]
            output2 = outputs[j]
            diffs[i, j] = float(output1) - float(output2)

    inputs = inputs.detach().numpy().squeeze()

    plt.rc('font', size=16)
    plt.rc('axes', titlesize=18)

    plt.imshow(diffs, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='$\Delta$ Feature effect')

    ticks = np.linspace(-1, 1, 5)
    tick_indices = np.linspace(0, len(inputs) - 1, len(ticks)).astype(int)

    plt.xticks(ticks=tick_indices, labels=ticks)
    plt.yticks(ticks=tick_indices, labels=ticks)

    # if feature == 'CRP' :
    #    plt.clim(-0.82, 0.82)

    # elif feature == 'LacticAcid':
    #    plt.clim(-0.03, 0.03)
    # else:
    #   plt.clim(-0.5, 0.5)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.clim(-1.3, 2.7)

    plt.plot([-0.5, 1.5], [-0.5, 1.5], color='grey', linewidth=0.6)
    plt.xlabel("Feature value $t$")
    plt.ylabel("Feature value $t+1$")
    plt.title(f"Sequential feature: {seq_features[idx]}")
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(f'../plots/{feature}_no_history_diffs.pdf', dpi=100)
    plt.close(fig1)

# (2) Print static features (global)
for idx, value in enumerate(static_features):
    plt.rcParams["figure.figsize"] = (7.5, 5)
    plt.rc('font', size=16)
    # x, out = model.plot_feat_stat_effect_custom(idx, 0, 1)
    x, out = model.plot_feat_stat_effect(idx, torch.from_numpy(x_statics_final[:, idx].reshape(-1, 1)).float())
    x = x.detach().numpy().squeeze()
    out = out.detach().numpy()
    if value == "Age":
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_out = out[sorted_indices]
        plt.plot(sorted_x, sorted_out + 1.6, linewidth=3, color='steelblue')
        plt.ylim(0, 1.7)
        plt.yticks(np.arange(0, 1.7, step=0.2))


    elif value == "Diagnose":
        plt.scatter(x, out, color='steelblue')
        # plt.ylim(0.19, 0.41)

    elif value == "Hypotensie":
        a, b = zip(set(x), set(np.squeeze(out)))
        x = [list(a)[0], list(b)[0]]
        out = [list(a)[1] + 1.3, list(b)[1] + 1.3]
        plt.bar(x, out, color='steelblue')
        plt.xticks(x, x)

    elif value == "DiagnosticBlood":
        a, b = zip(set(x), set(np.squeeze(out)))
        x = [list(a)[0], list(b)[0]]
        out = [list(a)[1] + 1.35, list(b)[1] + 1.35]
        plt.bar(x, out, color='steelblue')
        plt.xticks(x, x)

    else:
        try:
            a, b = zip(set(x), set(np.squeeze(out)))
            x = [list(a)[0], list(b)[0]]
            out = [list(a)[1], list(b)[1]]
            plt.bar(x, out, color='steelblue')
            plt.xticks(x, x)
        except:
            plt.scatter(x, out, color='steelblue')

    plt.xlabel("Feature value")
    plt.ylabel("Feature effect on model output")
    plt.title(f"Static feature: {static_features[idx]}")
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(f'../plots/{value}.pdf', dpi=100)
    plt.close(fig1)

# (3) Print sequential feature over time with value range (global)
for t in range(0, 11):
    for idx, value in enumerate(seq_features):
        plt.rcParams["figure.figsize"] = (7.5, 5)
        plt.rc('font', size=16)

        if value == "CRP":  # or value == "LacticAcid" or value == "CRP":
            # x, out = model.plot_feat_seq_effect_custom(idx, -2, 2)
            x, out, h_t, out_coef = model.plot_feat_seq_effect(idx, torch.from_numpy(
                x_seqs_final[:, t, idx].reshape(-1, 1, 1)).float())
            x = x.detach().numpy().squeeze()
            out = out.detach().numpy()
            sorted_indices = np.argsort(x)
            sorted_x = x[sorted_indices]
            sorted_out = out[sorted_indices]
            plt.plot(sorted_x, sorted_out + 0.75, linewidth=3, color='steelblue')

            plt.xlim(-0.02, 1.02)
            plt.ylim(-0.02, 3.02)
            plt.xlabel("Feature value")
            plt.ylabel("Feature effect on model output")
            plt.title("Sequential feature: %s ($t_{%s}$)" % (str(seq_features[idx]), str(t + 1)))
            fig1 = plt.gcf()
            plt.show()
            plt.draw()
            fig1.savefig(f'../plots/{value}_t{t + 1}.pdf', dpi=100)
            plt.close(fig1)

# (4) Print sequential features (local, no history)
effect_feature_values = []
case = 251
colors = ['steelblue', 'olivedrab', 'crimson', 'grey', 'yellow', 'lightskyblue', 'darkmagenta', 'darkorange']
plt.gca().set_prop_cycle(color=colors)

# seq_features=['Leucocytes', 'CRP', 'LacticAcid', 'IV Liquid', 'Admission NC']
seq_features_rel = ['Leucocytes', 'CRP', 'LacticAcid', 'ER Registration', 'ER Triage', 'ER Sepsis Triage',
                    'IV Liquid', 'IV Antibiotics']

for idx, value in enumerate(seq_features):
    effect_feature_values.append([])
    if value in seq_features_rel:
        for t in range(0, 11):
            x, out, h_t, out_coef = model.plot_feat_seq_effect(idx, torch.from_numpy(
                x_seqs_final[case, t, idx].reshape(1, 1, 1)).float())
            x = x.detach().numpy().squeeze()
            out = out.detach().numpy()

            if t == 0:
                correction_value = 0 - out[0][0]

            out_correction = out[0][0] + correction_value

            effect_feature_values[-1].append(out_correction)

        # plt.ylim(-0.11, 0.21)
        plt.plot(list(range(1, 12)), effect_feature_values[idx], label=value, linestyle='dashed', linewidth=3,
                 marker='o', markersize=6)
plt.axhline(y=0, color='grey', linewidth=0.6)
plt.xlabel("Time step", fontsize=16)
plt.ylabel("Feature effect on model output", fontsize=16)
plt.title(f"Feature effect over time of patient pathway 18", fontsize=16)
fig1 = plt.gcf()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=7, fontsize=16)
# plt.legend(loc='lower left', title='Sequential feature')  # adjust based on plot
plt.xticks(np.arange(1, 12, 1))
plt.rcParams["figure.figsize"] = (9, 9)
# plt.rc('axes', titlesize=20)
# plt.rc('axes', labelsize=17)
# plt.rc('xtick', labelsize=17)
# plt.rc('ytick', labelsize=17)
# plt.rc('legend', fontsize=19)
# plt.rc('legend', title_fontsize=19)
plt.show()
plt.draw()
fig1.savefig(f'../plots/seq_features_case_{case}.pdf', dpi=100)
plt.close(fig1)

# (5) Print sequential feature interactions (global, no history)
print(interactions_seq)

from scipy.interpolate import griddata

for t in range(1, 12):
    if number_interactions_seq > 0:
        for idx in range(0, number_interactions_seq):
            a = torch.from_numpy(x_seqs_final[:, t, interactions_seq[idx][0]].reshape(-1, 1, 1))
            b = torch.from_numpy(x_seqs_final[:, t, interactions_seq[idx][1]].reshape(-1, 1, 1))
            x = torch.cat((a, b), dim=2)

            X_seq, out = model.plot_feat_seq_effect_inter(idx, x)
            X_seq = X_seq.detach().numpy().squeeze()
            out = out.detach().numpy()

            a = a.detach().numpy().squeeze()
            b = b.detach().numpy().squeeze()

            grid_x, grid_y = np.mgrid[min(a):max(a):100j, min(b):max(b):100j]

            grid_z = griddata((a, b), out, (grid_x, grid_y))

            plt.imshow(grid_z.T.squeeze(), extent=(min(a), max(a), min(b), max(b)), origin='lower')

            # data = np.concatenate((a, b, out), axis=1)
            # plt.figure(figsize=(7.5, 5))
            # plt.rc('axes', titlesize=18)
            # plt.rc('axes', labelsize=16)
            # plt.rc('xtick', labelsize=16)
            # plt.rc('ytick', labelsize=16)

            # plt.pcolormesh(data[:, 0], data[:, 1], data[:, 2])

            # plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap='viridis')

            plt.colorbar(label='Interaction effect')

            plt.xlabel(f"Feature value {seq_features[interactions_seq[idx][0]]}")
            plt.ylabel(f"Feature value {seq_features[interactions_seq[idx][1]]}")
            plt.title("Interaction: %s x %s ($t_{%s}$)" % (str(seq_features[interactions_seq[idx][0]]),
                                                           str(seq_features[interactions_seq[idx][1]]), str(t + 1)))
            fig1 = plt.gcf()
            plt.show()
            plt.draw()
            fig1.savefig(f'../plots/interaction_{interactions_seq[idx][0]}-{interactions_seq[idx][1]}_{t}.pdf', dpi=100)
            plt.close(fig1)
