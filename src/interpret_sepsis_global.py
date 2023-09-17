import pandas as pd
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import src.data as data
from src.main import time_step_blow_up

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
name = "4_98"
# Map location param is required as the model was trained on gpu
model = torch.load(os.path.join("../model", f"model_{name}"), map_location=torch.device('cpu'))
interactions_seq = model.get_interactions_seq()
number_interactions_seq = len(interactions_seq)
file_format="png"  # pdf, png

# Note: static features include binary features created from the diagnosis feature or not
x_seqs, x_statics, y, x_time_vals_final, seq_features, static_features = data.get_sepsis_data('Admission IC', 50, 3)
x_seqs_final, x_statics_final, y_final = time_step_blow_up(x_seqs, x_statics, y, 50)

case = -1

for t in range(1, 13):
    feat_imports = []
    feat_names = []

    # Static features
    for idx, value in enumerate(static_features):

        x, out = model.plot_feat_stat_effect(idx, torch.from_numpy(x_statics_final[:, idx].reshape(-1, 1)).float())
        x = x.detach().numpy().squeeze()
        out = out.detach().numpy()
        out = np.ravel(out)

        out_min = min(out)
        out_delta = 0 - out_min
        out = [x + out_delta for x in out]

        if len(set(out)) <= 2:
            feat_imports.append(max(out))
        else:
            from scipy import integrate

            sorted_index = np.argsort(np.array(x))
            out_sorted = np.array(out)[sorted_index]
            x_sorted = np.array(x)[sorted_index]
            feat_imports.append(integrate.trapz(y=out_sorted, x=x_sorted))

        feat_names.append(value)

    # Sequential features
    for idx, value in enumerate(seq_features):

        plt.figure(figsize=(5, 10))
        plt.rc('font', size=14)
        plt.rc('axes', titlesize=16)
        # plt.rc('xtick', labelsize=10)
        # plt.rc('ytick', labelsize=10)

        if t == 1:
            x_n = torch.linspace(0, 1, 200).reshape(200, 1, 1).float()
        else:
            # history of trace
            x_hist = torch.from_numpy(x_seqs_final[case, 0:t - 1, idx].reshape(1, t - 1, 1)).float()
            x_hist = x_hist.repeat(200, 1, 1)
            x_n = torch.linspace(0, 1, 200).reshape(200, 1, 1).float()
            x_n = torch.cat((x_hist, x_n), 1)

        x, out, _, _ = model.plot_feat_seq_effect(idx, x_n, history=True)
        x = x.detach().numpy().squeeze()
        out = out.detach().numpy()
        out = np.ravel(out)
        out = out.tolist()

        out_min = min(out)
        out_delta = 0 - out_min
        out = [o + out_delta for o in out]

        # last data points
        if t > 1:
            x = x[:, -1]

        if len(set(out)) <= 2:
            feat_imports.append(max(out))
        else:
            from scipy import integrate

            sorted_index = np.argsort(np.array(x))
            out_sorted = np.array(out)[sorted_index]
            x_sorted = np.array(x)[sorted_index]
            feat_imports.append(integrate.trapz(y=out_sorted, x=x_sorted))

        feat_names.append(value)

    sorted_index = np.argsort(np.array(feat_imports))
    feat_imports_sorted = np.array(feat_imports)[sorted_index]
    feat_names_sorted = np.array(feat_names)[sorted_index]
    feat_names_sorted = feat_names_sorted.tolist()
    feat_imports_sorted = feat_imports_sorted.tolist()

    # filter
    feat_names_sorted = feat_names_sorted[19:]  # total 39 features
    feat_imports_sorted = feat_imports_sorted[19:]
    y_pos = np.arange(len(feat_names_sorted))

    # colouring depending on bar length
    cmap = plt.get_cmap('Reds')
    norm = plt.Normalize(min(feat_imports_sorted), max(feat_imports_sorted))
    colors = cmap(norm(feat_imports_sorted))


    data = pd.DataFrame({'x': feat_names_sorted, 'y': y_pos})
    plot = plt.barh(y_pos, feat_imports_sorted, color=colors, zorder=2)

    # plt.yticks(y_pos, feat_names_sorted)
    plt.tick_params(left=False, labelleft=False)
    plt.xticks(np.arange(0, 1.41, step=0.2))
    plt.grid(True, zorder=0)

    def autolabel(plot):
        for idx, rect in enumerate(plot):
            if feat_imports_sorted[idx] > 0.7:
                plt.text(0.005, idx - 0.25, feat_names_sorted[idx], color='white')
            else:
                plt.text(0.005, idx - 0.25, feat_names_sorted[idx], color='black')

    autolabel(plot)

    plt.xlabel("Importance")
    plt.ylabel("Medical indicator")
    plt.title(f"Importance of medical indicators")
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(f'../plots/sepsis/global_feat_importance_{t}.{file_format}', dpi=100, bbox_inches="tight")
    plt.close(fig1)