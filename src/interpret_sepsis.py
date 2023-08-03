import torch
import os
import matplotlib.pyplot as plt
import src.data as data
from src.main import time_step_blow_up
import numpy as np
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

name = "4_98"
model = torch.load(os.path.join("../model", f"model_{name}"), map_location=torch.device('cpu'))
interactions_seq = model.get_interactions_seq()
number_interactions_seq = len(interactions_seq)

x_seqs, x_statics, y, x_time_vals_final, seq_features, static_features = data.get_sepsis_data('Admission IC', 50, 3)
x_seqs_final, x_statics_final, y_final = time_step_blow_up(x_seqs, x_statics, y, 50)

"""
# (1) Sequential feature transition (2 time steps, no history)
plt.rcParams["figure.figsize"] = (7.5, 5)
for idx, feature in enumerate(seq_features):

    inputs = torch.linspace(0, 1, 200).reshape(200, 1, 1).float()
    x_x, out_x, _, _ = model.plot_feat_seq_effect(idx, inputs, history=False)
    x_x = x_x.detach().numpy().squeeze()
    outputs = out_x.detach().numpy().squeeze()
    diffs = np.empty((len(inputs), len(inputs)))

    for i in range(200):
        for j in range(200):
            output1 = outputs[i]
            output2 = outputs[j]
            diffs[i, j] =  float(output2) - float(output1)

    inputs = inputs.detach().numpy().squeeze()

    plt.rc('font', size=16)
    plt.rc('axes', titlesize=18)

    plt.imshow(diffs, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar(label='$\Delta$ Feature effect')

    ticks = np.linspace(0, 1, 5)
    tick_indices = np.linspace(0, len(inputs) - 1, len(ticks)).astype(int)

    plt.xticks(ticks=tick_indices, labels=ticks)
    plt.yticks(ticks=tick_indices, labels=ticks)

    # if feature == 'CRP' :
    #    plt.clim(-0.82, 0.82)

    # elif feature == 'LacticAcid':
    #    plt.clim(-0.03, 0.03)
    # else:
    #   plt.clim(-0.5, 0.5)
    #plt.xlim(-0.05, 1.05)
    #plt.ylim(-0.05, 1.05)
    #plt.clim(-1.3, 2.7)

    plt.xlabel("Feature value $t$")
    plt.ylabel("Feature value $t+1$")
    plt.title(f"Sequential feature: {seq_features[idx]}")
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(f'../plots/sepsis/seq_feat_diffs_{feature}.pdf', dpi=100, bbox_inches="tight")
    plt.close(fig1)


# (2) Print static features (global)
for idx, value in enumerate(static_features):
    plt.rcParams["figure.figsize"] = (7.5, 5)
    plt.rc('font', size=16)
    x, out = model.plot_feat_stat_effect(idx, torch.from_numpy(x_statics_final[:, idx].reshape(-1, 1)).float())
    x = x.detach().numpy().squeeze()
    out = out.detach().numpy()

    if value == "Age":
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_out = out[sorted_indices]
        plt.plot(sorted_x, sorted_out, linewidth=3, color='steelblue')  # + 1.6
        # plt.ylim(0, 1.7)
        # plt.yticks(np.arange(0, 1.7, step=0.2))

    elif value == "Diagnose":
        plt.scatter(x, out, color='steelblue')
        # plt.ylim(0.19, 0.41)

    elif value == "Hypotensie":
        a, b = zip(set(x), set(np.squeeze(out)))
        x = [list(a)[0], list(b)[0]]
        out = [list(a)[1], list(b)[1]]  # +1.3
        plt.bar(x, out, color='steelblue')
        plt.xticks(x, x)

    elif value == "Oligurie":
        a, b = zip(set(x), set(np.squeeze(out)))
        x = [list(a)[0], list(b)[0]]
        out = [list(a)[1] - list(a)[1] + 0.01, list(b)[1] - list(a)[1]]  # +1.3
        plt.bar(x, out, color='steelblue')
        plt.xticks(x, x)

    elif value == "DiagnosticBlood":
        a, b = zip(set(x), set(np.squeeze(out)))
        x = [list(a)[0], list(b)[0]]
        if list(b)[1] < 0:
            out = [list(a)[1] - list(b)[1], list(b)[1] - list(b)[1] - 0.001]   # + 1.35
        else:
            out = [list(a)[1] + list(b)[1], list(b)[1] + list(b)[1] - 0.001]
        plt.bar(x, out, color='steelblue')
        plt.ylim(-1.0, 0.01)
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
    fig1.savefig(f'../plots/sepsis/stat_feat_{value}.pdf', dpi=100, bbox_inches="tight")
    plt.close(fig1)

# (3) Print sequential features (global, no history)
for idx, value in enumerate(seq_features):
    plt.rcParams["figure.figsize"] = (7.5, 5)
    plt.rc('font', size=16)

    if value == "CRP" or value == "Leucocytes" or value == "LacticAcid":
        inputs = torch.linspace(0, 1, 200).reshape(200, 1, 1).float()
        x, out, h_t, out_coef = model.plot_feat_seq_effect(idx, inputs, history=False)
        x = x.detach().numpy().squeeze()
        out = out.detach().numpy()

        plt.plot(x, out, linewidth=3, color='steelblue')

        #plt.xlim(-0.02, 1.02)
        #plt.ylim(-0.02, 3.02)
        plt.xlabel("Feature value")
        plt.ylabel("Feature effect on model output")
        plt.title("Sequential feature: %s" % (str(seq_features[idx])))
        fig1 = plt.gcf()
        plt.show()
        plt.draw()
        fig1.savefig(f'../plots/sepsis/seq_feat_{value}.pdf', dpi=100, bbox_inches="tight")
        plt.close(fig1)
        

# (4) Print sequential feature interactions (global, no history)
print(interactions_seq)
plt.rcParams["figure.figsize"] = (7.5, 5)
from scipy.interpolate import griddata

if number_interactions_seq > 0:
    for idx in range(0, number_interactions_seq):
        fig, ax = plt.subplots()

        a = torch.linspace(0, 1, 200).reshape(200, 1, 1).float()
        b = torch.linspace(0, 1, 200).reshape(200, 1, 1).float()
        x = torch.cat((a, b), dim=2)

        X_seq, out = model.plot_feat_seq_effect_inter(idx, x)
        X_seq = X_seq.detach().numpy().squeeze()
        out = out.detach().numpy()

        a = a.detach().numpy().squeeze()
        b = b.detach().numpy().squeeze()

        grid_x, grid_y = np.mgrid[min(a):max(a):200j, min(b):max(b):200j]
        grid_z = griddata((a, b), out, (grid_x, grid_y), method="nearest")

        plt.rc('font', size=16)
        plt.rc('axes', titlesize=18)

        im = ax.imshow(grid_z.T.squeeze(), extent=(min(a), max(a), min(b), max(b)), origin='lower', cmap='magma')
        fig.colorbar(im, ax=ax, label='Interaction effect')

        plt.xlabel(f"Feature value {seq_features[interactions_seq[idx][0]]}")
        plt.ylabel(f"Feature value {seq_features[interactions_seq[idx][1]]}")
        plt.title("Interaction: %s x %s" % (
            str(seq_features[interactions_seq[idx][0]]),
            str(seq_features[interactions_seq[idx][1]])))

        fig1 = plt.gcf()
        plt.show()
        plt.draw()
        fig1.savefig(f'../plots/sepsis/seq_feat_inter_{interactions_seq[idx][0]}-{interactions_seq[idx][1]}.pdf',
                     dpi=100, bbox_inches="tight")
        plt.close(fig1)
"""

# (5) Print sequential feature (local, history)
plt.rcParams["figure.figsize"] = (20, 15)
plt.rc('font', size=16)
plt.rc('axes', titlesize=18)
case = 8067  # id of prefix == 8067 for case 581

seq_features_rel = ['Leucocytes', 'LacticAcid', 'CRP']  # 'CRP'

list_effect = []
list_value = []
list_time = []

for idx, value in enumerate(seq_features_rel):

    effect_feature_values = []
    data_feature_values = []

    if value in seq_features_rel:
        for t in range(0, 11):
            x, out, h_t, out_coef = model.plot_feat_seq_effect(
                idx, torch.from_numpy(x_seqs_final[case, 0:t + 1, idx].reshape(1, t + 1, 1)).float(), history=True)
            x = x.detach().numpy().squeeze()
            out = out.detach().numpy()

            out_correction = out[0][0]  # + correction_value

            effect_feature_values.append(out_correction)
            if t == 0:
                data_feature_values.append(x.item())
            else:
                data_feature_values.append(x[-1])

        steps = list(range(1, 12))
        for i in range(len(steps)):

            if i == 1:
                pass
            else:
                if effect_feature_values[i - 1] == effect_feature_values[i]:
                    pass
                else:
                    pass

        list_effect = list_effect + effect_feature_values
        list_value = list_value + data_feature_values
        list_time = list_time + list(range(1, 12))

df = pd.DataFrame({'x': list_time, 'y': list_value, 'z': list_effect})

clim_max = df['z'].max()
clim_min = df['z'].min()

fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
idx = 0

for ax in axes.flat:
    plt.sca(ax)
    effect_feature_values = []
    data_feature_values = []

    for t in range(0, 11):
        x, out, h_t, out_coef = model.plot_feat_seq_effect(
            idx, torch.from_numpy(x_seqs_final[case, 0:t + 1, idx].reshape(1, t + 1, 1)).float(), history=True)
        x = x.detach().numpy().squeeze()
        out = out.detach().numpy()

        out_correction = out[0][0]
        effect_feature_values.append(out_correction)
        if t == 0:
            data_feature_values.append(x.item())
        else:
            data_feature_values.append(x[-1])

    feature_name = seq_features_rel[idx]

    plt.plot(list(range(1, 12)), data_feature_values, linewidth=1, color='grey')
    plt.scatter(list(range(1, 12)), data_feature_values, c=effect_feature_values, cmap='viridis')
    # plt.ylim(-0.02, 0.85)
    plt.clim(clim_min, clim_max)
    plt.ylabel("Feature value of " + feature_name, fontsize=16)

    steps = list(range(1, 12))
    for i in range(len(steps)):
        plt.annotate(round(effect_feature_values[i], 3), (steps[i], data_feature_values[i]), xytext=(10, -5),
                     textcoords='offset points')

    plt.axhline(y=0, color='grey', linewidth=0.6)

    if idx == 2:
        plt.xlabel("Time step", fontsize=16)
        plt.xticks(np.arange(1, 12, 1))

    if idx == 0:
        plt.title(f"Sequential feature effect over time for individual patient pathway", fontsize=16)
    idx = idx + 1

plt.colorbar(ax=axes.ravel().tolist(), label='Feature effect on model output', location='right', shrink=0.8)
# plt.show()
fig.savefig(f'../plots/sepsis/seq_feat_case_{case}_single.with_hist.pdf', dpi=100, bbox_inches="tight")
plt.close(fig)
